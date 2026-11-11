"""Audit complete episodes and retain the evidence behind each score."""

import shutil
import subprocess
from collections import defaultdict
from statistics import mean

from .dataset import Dataset
from .io import digest, distinct, number, positive, read_jsonl, require_same, weights

QUALITY_WEIGHTS = {
    "integrity": 0.10,
    "timing": 0.25,
    "vision": 0.15,
    "action": 0.30,
    "language": 0.10,
    "provenance": 0.10,
}


def validate_protocol(config):
    if config.get("version") != 1 or not config.get("protocol_id"):
        raise ValueError("A version 1 evaluation protocol and protocol_id are required")
    if any(not config["robot"].get(k) for k in ("id", "action_space", "gripper_direction")):
        raise ValueError("Declare robot ID, action space and gripper direction")
    for field in ("action_dimensions", "state_dimensions"):
        dims = config[field]
        if not dims or len(distinct(dims, "name")) != len(dims):
            raise ValueError(f"Missing or duplicate dimensions: {field}")
        for dim in dims:
            if (
                not dim["unit"]
                or not number(dim["min"])
                or not number(dim["max"])
                or dim["min"] >= dim["max"]
            ):
                raise ValueError(f"Invalid units or bounds for {dim['name']}")
            if field == "action_dimensions":
                positive(dim["scale"], "error scale")
                positive(dim["max_step"], "action step limit")
    rules = config["quality"]
    positive(rules["grid_tolerance_s"], "grid tolerance")
    if type(rules["samples_per_camera"]) is not int or rules["samples_per_camera"] < 1:
        raise ValueError("samples_per_camera must be a positive integer")
    if type(rules["require_raw_timing"]) is not bool:
        raise ValueError("require_raw_timing must be a boolean")
    for name in ("age_ms", "camera_skew_ms", "command_delay_ms"):
        good, bad = rules["timing"][name]
        if not number(good) or not number(bad) or not 0 <= good < bad:
            raise ValueError(f"Invalid good/bad timing anchors for {name}")
    weights(config["task_weights"], "task weights")
    coverage = config["coverage"]
    weights({key: value["weight"] for key, value in coverage.items()}, "coverage weights")
    for cell in coverage.values():
        positive(cell["quota"], "coverage quota")
        if cell["task_id"] not in config["task_weights"]:
            raise ValueError("Coverage cell names a task outside task_weights")
    if type(config["min_training_seeds"]) is not int or config["min_training_seeds"] < 2:
        raise ValueError("min_training_seeds must be an integer >= 2")


def assignments(path, dataset, config):
    rows = distinct(read_jsonl(path), "episode_index")
    require_same(set(rows), set(dataset.episodes), "episode manifest coverage")
    group_splits = {}
    for index, row in rows.items():
        split, group = row["split"], row["source_group"]
        if (
            type(index) is not int
            or split not in {"train", "dev", "test"}
            or not isinstance(group, str)
            or not group.strip()
        ):
            raise ValueError(f"Invalid split or source group for episode {index}")
        if group_splits.setdefault(group, split) != split:
            raise ValueError(f"Source group {group} crosses splits")
        cell = config["coverage"][row["coverage_cell"]]
        if cell["task_id"] != row["task_id"]:
            raise ValueError(f"Coverage cell and task disagree for episode {index}")
        for name in ("action_verified", "language_verified"):
            if type(row.get(name)) is not bool:
                raise ValueError(f"Episode {index}: {name} must be true or false")
    return rows


def _gray_frame(*, file=None, offset=0, blob=None):
    args = ["ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file,pipe"]
    if file is not None:
        args += ["-ss", f"{offset:.6f}", "-i", str(file)]
    else:
        args += ["-i", "pipe:0"]
    args += ["-frames:v", "1", "-vf", "scale=32:32,format=gray", "-threads", "1", "-f", "rawvideo", "pipe:1"]
    result = subprocess.run(args, input=blob, capture_output=True, timeout=30)
    if result.returncode or len(result.stdout) != 1024:
        raise ValueError("frame could not be decoded")
    return result.stdout


def _video_duration(path, cache):
    if path not in cache:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-protocol_whitelist",
                "file,pipe",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=duration",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode:
            raise ValueError("ffprobe failed")
        cache[path] = positive(float(result.stdout.strip()), "video duration")
    return cache[path]


def _vision(ds, index, frames, keys, rules, cache, issue):
    meta = ds.episodes[index]
    scores = []
    sample_count = min(len(frames), rules["samples_per_camera"])
    slots = [(2 * i + 1) * len(frames) // (2 * sample_count) for i in range(sample_count)]
    for key in keys:
        previous = None
        for slot in slots:
            frame = frames[slot]
            try:
                if ds.cameras[key] == "video":
                    prefix = f"videos/{key}"
                    path = ds.video_map[index, key]
                    start, end = meta[f"{prefix}/from_timestamp"], meta[f"{prefix}/to_timestamp"]
                    duration = _video_duration(path, cache)
                    if not 0 <= start < end <= duration + 1 / ds.fps:
                        raise ValueError("episode video offsets exceed the file duration")
                    offset = start + frame["timestamp"]
                    if not start <= offset < min(end, duration):
                        raise ValueError("requested frame is outside the episode video segment")
                    pixels = _gray_frame(file=path, offset=offset)
                else:
                    data = frame[key]
                    if not data or not data.get("bytes"):
                        raise ValueError("embedded image bytes are missing")
                    pixels = _gray_frame(blob=data["bytes"])
            except (KeyError, TypeError, ValueError, OSError, subprocess.TimeoutExpired) as exc:
                issue("MEDIA_DECODE", "block", slot, f"{key}: {exc}")
                continue
            ok = mean(pixels) >= rules["black_mean_threshold"]
            if not ok:
                issue("DARK_FRAME", "review", slot, key)
            if previous and pixels == previous[0] and frame["observation.state"] != previous[1]:
                issue("POSSIBLE_FROZEN_FRAME", "review", slot, key)
                ok = False
            scores.append(ok)
            previous = pixels, frame["observation.state"]
    return 100 * mean(scores) if len(scores) == sample_count * len(keys) else None


def _raw_timing(timing, frames, index, keys, rules, issue, fps=None):
    maxima = {"age_ms": 0.0, "camera_skew_ms": 0.0, "command_delay_ms": 0.0}
    last_decision = -1
    max_gap = 0.0
    missing = False
    for slot in range(len(frames)):
        row = timing.get((index, slot))
        if row is None:
            issue("RAW_TIMING_MISSING", "block" if rules["require_raw_timing"] else "review", slot)
            missing = True
            continue
        try:
            decision = row["decision_ns"]
            state, state_received = row["state_sample_ns"], row["state_receive_ns"]
            issue_ns, apply_ns = row["command_issue_ns"], row["command_apply_ns"]
            captures = [row["image_capture_ns"][key] for key in keys]
            receipts = [row["image_receive_ns"][key] for key in keys]
            values = [decision, state, state_received, issue_ns, apply_ns, *captures, *receipts]
            if not all(type(v) is int and v >= 0 for v in values):
                raise ValueError("timestamps must be nonnegative integer nanoseconds")
            if not (
                last_decision < decision
                and state <= state_received <= decision <= issue_ns <= apply_ns
                and all(c <= r <= decision for c, r in zip(captures, receipts, strict=True))
            ):
                raise ValueError("noncausal timestamps or a clock reset")
        except (KeyError, TypeError, ValueError) as exc:
            issue("RAW_TIMING_INVALID", "block", slot, str(exc))
            return None, {}
        if last_decision >= 0:
            gap = (decision - last_decision) / 1e6
            max_gap = max(max_gap, gap)
            if fps is not None and gap > 2000 / fps:
                issue("RAW_DECISION_GAP", "review", slot)
        last_decision = decision
        metrics = {
            "age_ms": (decision - min(state, *captures)) / 1e6,
            "camera_skew_ms": (max(captures) - min(captures)) / 1e6,
            "command_delay_ms": (apply_ns - decision) / 1e6,
        }
        for name, value in metrics.items():
            maxima[name] = max(maxima[name], value)
            if value > rules["timing"][name][1]:
                issue("TIMING_BUDGET", "review", slot, name)
    scores = []
    for name, value in maxima.items():
        good, bad = rules["timing"][name]
        scores.append(100 * max(0, min(1, (bad - value) / (bad - good))))
    return (None if missing else mean(scores)), {**maxima, "max_decision_gap_ms": max_gap}


def _vector(values, dims):
    return isinstance(values, (list, tuple)) and len(values) == len(dims) and all(number(v) for v in values)


def audit(root, config, episode_manifest, raw_timing=None):
    validate_protocol(config)
    ds = Dataset(root)
    ds.check_dimensions(config)
    rows = assignments(episode_manifest, ds, config)
    timing = {}
    for row in read_jsonl(raw_timing) if raw_timing else ():
        key = row["episode_index"], row["frame_index"]
        if (
            key in timing
            or type(key[0]) is not int
            or key[0] not in rows
            or type(key[1]) is not int
            or not 0 <= key[1] < ds.episodes[key[0]]["length"]
        ):
            raise ValueError(f"Invalid or duplicate timing key: {key}")
        timing[key] = row
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise ValueError("The media audit requires ffmpeg and ffprobe")
    keys, rules = config["camera_keys"], config["quality"]
    act_dims, state_dims = config["action_dimensions"], config["state_dimensions"]
    extras = [k for k in keys if ds.cameras[k] == "image"]
    results, review, durations, exact = [], [], {}, defaultdict(list)
    intervals = []
    total_frames = 0
    for index, frames in ds.frames(extras):
        total_frames += len(frames)
        meta, entry = ds.episodes[index], rows[index]
        problems = {}

        def issue(code, level, slot=None, detail=None, *, problems=problems, index=index):
            if code not in problems:
                problems[code] = {
                    "code": code,
                    "severity": level,
                    "count": 0,
                    "episode_index": index,
                    "frame_index": slot,
                    "detail": detail,
                }
            problems[code]["count"] += 1

        if not entry["action_verified"]:
            issue("ACTION_SEMANTICS_UNVERIFIED", "block")
        if not entry["language_verified"]:
            issue("LANGUAGE_UNVERIFIED", "block")
        start = meta["dataset_from_index"]
        intervals.append((start, meta["dataset_to_index"]))
        if len(frames) != meta["length"] or meta["dataset_to_index"] != start + len(frames):
            issue("EPISODE_OFFSETS", "block")
        valid, clean_steps, previous = 0, [], None
        task_ids = set()
        for slot, row in enumerate(frames):
            time = row["timestamp"]
            task_ids.add(row["task_index"])
            if (
                row["index"] != start + slot
                or row["frame_index"] != slot
                or not number(time)
                or abs(time - slot / ds.fps) > rules["grid_tolerance_s"]
                or (slot and number(frames[slot - 1]["timestamp"]) and time <= frames[slot - 1]["timestamp"])
            ):
                issue("FRAME_GRID", "block", slot)
            action, state = row["action"], row["observation.state"]
            if not _vector(action, act_dims) or not _vector(state, state_dims):
                issue("INVALID_VECTOR", "block", slot)
                previous = None
                continue
            valid += 1
            for label, values, dims in (("action", action, act_dims), ("state", state, state_dims)):
                if any(not d["min"] <= v <= d["max"] for v, d in zip(values, dims, strict=True)):
                    issue("OUT_OF_BOUNDS", "block", slot, label)
            if previous is not None:
                ok = not any(
                    abs(v - old) > d["max_step"] for v, old, d in zip(action, previous, act_dims, strict=True)
                )
                clean_steps.append(ok)
                if not ok:
                    issue("ACTION_JUMP", "review", slot)
            previous = action
        task_texts = [ds.task_text(t) for t in sorted(task_ids, key=str)]
        if any(not isinstance(t, str) or not t.strip() for t in task_texts):
            issue("TASK_TEXT_MISSING", "block")
        timing_score, timing_metrics = _raw_timing(timing, frames, index, keys, rules, issue, fps=ds.fps)
        vision = _vision(ds, index, frames, keys, rules, durations, issue)
        if len(frames) < 2:
            issue("NO_ACTION_TRANSITION", "block")
        dimensions = {
            "integrity": 100 * valid / len(frames),
            "timing": timing_score,
            "vision": vision,
            "action": 100 * mean(clean_steps) if clean_steps else None,
            "language": 100 if entry["language_verified"] else None,
            "provenance": 100,
        }
        eligible = not any(item["severity"] == "block" for item in problems.values())
        quality = (
            sum(dimensions[k] * w for k, w in QUALITY_WEIGHTS.items())
            if eligible and all(v is not None for v in dimensions.values())
            else None
        )
        result = {
            **entry,
            "n_frames": len(frames),
            "task_texts": task_texts,
            "eligible": eligible,
            "quality": quality,
            "dimensions": dimensions,
            "recommended_action": "quarantine" if not eligible else "review" if problems else "retain",
            "timing": timing_metrics,
            "reason_codes": sorted(problems),
        }
        results.append(result)
        review.extend(problems.values())
        if eligible and any(f["observation.state"] != frames[0]["observation.state"] for f in frames):
            fingerprint = digest([task_texts, [(f["action"], f["observation.state"]) for f in frames]])
            exact[fingerprint].append(result)
    intervals.sort()
    if total_frames != ds.info["total_frames"] or any(
        start != (intervals[i - 1][1] if i else 0) for i, (start, _) in enumerate(intervals)
    ):
        raise ValueError("Dataset global frame count or episode index ranges do not match metadata")
    for group in exact.values():
        if len({r["split"] for r in group}) > 1:
            raise ValueError(f"Exact moving trajectory crosses splits: {[r['episode_index'] for r in group]}")
        if len(group) > 1:
            for row in group:
                row["reason_codes"].append("EXACT_TRAJECTORY_DUPLICATE")
                row["recommended_action"] = "review"
                review.append(
                    {
                        "episode_index": row["episode_index"],
                        "code": "EXACT_TRAJECTORY_DUPLICATE",
                        "severity": "review",
                        "count": 1,
                    }
                )
    return {
        "version": 1,
        "protocol_id": config["protocol_id"],
        "protocol_sha": digest(config),
        "dataset_sha": ds.fingerprint(),
        "fps": ds.fps,
        "episodes": sorted(results, key=lambda r: r["episode_index"]),
        "review": review,
    }
