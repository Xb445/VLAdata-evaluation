"""Generate offline metrics and collect case-level closed-loop evidence."""

import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .dataset import Dataset
from .experiments import checkpoint_hash, option_args, validate_run
from .io import digest, distinct, file_hash, number, read_json, read_jsonl, require_same, write_json

OUTCOMES = {"success", "failure", "timeout", "intervention", "not_started", "infra_invalid"}


def _check_checkpoint(run, plan):
    validate_run(run, plan)
    require_same(checkpoint_hash(run["checkpoint"]), run["checkpoint_sha"], "evaluation checkpoint")


def action_errors(prediction, target, padding, scales, dimensions):
    if not len(target) or len(prediction) != len(target) or len(padding) != len(target):
        raise ValueError("Predicted action horizon, target horizon, and padding mask disagree")
    if any(len(row) != len(scales) for row in [*prediction, *target]):
        raise ValueError("Predicted and target action dimensions must match the robot contract")
    errors = []
    jumps = {"prediction": 0, "label": 0}
    limit_steps = {"prediction": 0, "label": 0}
    transitions, previous = 0, None
    for pred, label, pad in zip(prediction, target, padding, strict=True):
        if pad:
            previous = None
            continue
        if not all(number(v) for v in [*pred, *label]):
            raise ValueError("Non-finite action in an unpadded horizon step")
        errors.append([abs(a - b) for a, b in zip(pred, label, strict=True)])
        for name, action in (("prediction", pred), ("label", label)):
            limit_steps[name] += any(
                not d["min"] <= v <= d["max"] for v, d in zip(action, dimensions, strict=True)
            )
        if previous is not None:
            transitions += 1
            for name, action, prior in (("prediction", pred, previous[0]), ("label", label, previous[1])):
                jumps[name] += any(
                    abs(v - old) > d["max_step"] for v, old, d in zip(action, prior, dimensions, strict=True)
                )
        previous = pred, label
    if not errors:
        raise ValueError("No valid action steps in the window")
    return {
        "valid_steps": len(errors),
        "mae": [mean(col) for col in zip(*errors, strict=True)],
        "scaled_mae": mean(e / s for row in errors for e, s in zip(row, scales, strict=True)),
        "first_step_mae": errors[0],
        "valid_transitions": transitions,
        "chunk_jump_steps": jumps,
        "outside_bounds_steps": limit_steps,
    }


def offline(root, plan, config, run, *, split="dev", device="cpu", stride=1, draws=1):
    import torch

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    _check_checkpoint(run, plan)
    require_same(digest(config), plan["protocol_sha"], "offline protocol")
    source = Dataset(root)
    require_same(source.fingerprint(), plan["dataset_sha"], "offline reference dataset")
    source.check_dimensions(config)
    ids = plan["reference_episodes"][split]
    if not ids or stride < 1 or draws < 1:
        raise ValueError("Offline evaluation needs reference episodes, positive stride and draws")
    repo_id = plan["recipe"]["repo_id"]
    checkpoint = run["checkpoint"]
    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint)
    if policy_cfg.type not in {"act", "pi05", "smolvla"}:
        raise ValueError(f"Unsupported action-chunk policy for offline evaluation: {policy_cfg.type}")
    policy_cfg.device = device
    if hasattr(policy_cfg, "compile_model"):
        policy_cfg.compile_model = False
    if getattr(policy_cfg, "rtc_config", None) is not None:
        policy_cfg.rtc_config.enabled = False
    meta = LeRobotDatasetMetadata(repo_id, root=source.root)
    offsets = resolve_delta_timestamps(policy_cfg, meta)
    if not offsets or "action" not in offsets:
        raise ValueError("Checkpoint does not declare an action prediction horizon")
    dataset = LeRobotDataset(
        repo_id, root=source.root, episodes=ids, delta_timestamps=offsets, video_backend="pyav"
    )
    policy = get_policy_class(policy_cfg.type).from_pretrained(checkpoint, config=policy_cfg)
    policy = policy.to(device).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    scales = [d["scale"] for d in config["action_dimensions"]]
    rows = []
    # Sample on each episode's own frame grid, independent of selected-episode concatenation.
    selected_indices = [
        int(i) for i, frame in enumerate(dataset.hf_dataset["frame_index"]) if int(frame) % stride == 0
    ]
    with torch.inference_mode():
        for i in selected_indices:
            item = dataset[i]
            raw_target = item["action"].tolist()
            padding = item["action_is_pad"].tolist()
            observation = {
                key: value for key, value in item.items() if key.startswith("observation.") or key == "task"
            }
            episode, frame = int(item["episode_index"]), int(item["frame_index"])
            for draw in range(draws):
                # The same sample receives the same noise seed across candidate checkpoints.
                noise_seed = int(digest([episode, frame, draw])[:8], 16)
                torch.manual_seed(noise_seed)
                policy.reset()
                pre.reset()
                post.reset()
                prediction = policy.predict_action_chunk(pre(dict(observation)))
                if prediction.ndim != 3 or prediction.shape[0] != 1:
                    raise ValueError(f"Expected [1, horizon, action_dim], got {tuple(prediction.shape)}")
                physical = [
                    post(prediction[:, h]).detach().cpu()[0].tolist() for h in range(prediction.shape[1])
                ]
                metrics = action_errors(physical, raw_target, padding, scales, config["action_dimensions"])
                rows.append(
                    dict(
                        episode_index=episode, frame_index=frame, draw=draw, noise_seed=noise_seed, **metrics
                    )
                )
    if not rows:
        raise ValueError("No reference windows evaluated")
    by_episode = defaultdict(list)
    for row in rows:
        by_episode[row["episode_index"]].append(row["scaled_mae"])
    return {
        "version": 1,
        "run": run,
        "split": split,
        "stride": stride,
        "draws": draws,
        "dimensions": [{"name": d["name"], "unit": d["unit"]} for d in config["action_dimensions"]],
        "episode_macro_scaled_mae": mean(mean(v) for v in by_episode.values()),
        "windows": rows,
    }


def _cases(plan):
    cases = plan.get("cases")
    if cases is None:
        raise ValueError("Freeze a case manifest when creating the plan")
    return cases, distinct(cases["cases"], "case_id")


def _attempts(run, plan, records, source):
    cases, expected = _cases(plan)
    attempts = defaultdict(list)
    for row in records:
        case = expected.get(row["case_id"])
        if case is None or row.get("outcome") not in OUTCOMES:
            raise ValueError(f"Unknown case or outcome: {row}")
        require_same(
            (row["scenario_id"], row["case_seed"]),
            (case["scenario_id"], case["seed"]),
            "case scenario and seed",
        )
        if type(row["attempt"]) is not int or row["attempt"] < 1:
            raise ValueError("attempt must be a positive integer")
        if row["outcome"] == "infra_invalid" and not row.get("reason"):
            raise ValueError("An infrastructure exclusion needs a recorded reason")
        attempts[row["case_id"]].append(row)
    for case, rows in attempts.items():
        rows.sort(key=lambda r: r["attempt"])
        if [row["attempt"] for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError(f"Invalid attempt numbering for {case}")
        if any(row["outcome"] != "infra_invalid" for row in rows[:-1]):
            raise ValueError(f"Completed case {case} cannot be replaced by another attempt")
    return {
        "version": 1,
        "run": run,
        "cases_sha": digest(cases),
        "environment": cases["environment"],
        "split": cases["split"],
        "source": source,
        "attempts": [row for key in sorted(attempts) for row in attempts[key]],
    }


def import_rollouts(plan, run, path):
    _check_checkpoint(run, plan)
    rows = list(read_jsonl(path))
    for row in rows:
        require_same(row["checkpoint_sha"], run["checkpoint_sha"], "recorded checkpoint")
        if row.get("outcome") != "infra_invalid" and (
            row.get("outcome_verified") is not True
            or not isinstance(row.get("evidence"), str)
            or not row["evidence"].strip()
        ):
            raise ValueError("Imported outcomes require confirmation and an evidence reference")
    return _attempts(run, plan, rows, {"kind": "imported", "sha256": file_hash(path)})


def _simulation_groups(cases):
    groups = defaultdict(list)
    for case in cases["cases"]:
        if not isinstance(case["suite"], str) or type(case["task_index"]) is not int:
            raise ValueError("Simulation cases require suite and integer task_index")
        groups[case["suite"], case["task_index"]].append(case)
    if not groups:
        raise ValueError("No simulation cases")
    expected = None
    for key, group in groups.items():
        group.sort(key=lambda c: c["seed"])
        seeds = [c["seed"] for c in group]
        if seeds != list(range(seeds[0], seeds[0] + len(seeds))) or (
            expected is not None and seeds != expected
        ):
            raise ValueError(f"{key}: native lerobot-eval needs the same consecutive seeds per task")
        expected = seeds
    return groups, expected


def parse_simulation(info, plan, run):
    cases, _ = _cases(plan)
    groups, _ = _simulation_groups(cases)
    records = []
    observed = set()
    for task in info["per_task"]:
        key = task["task_group"], task["task_id"]
        if key in observed or key not in groups:
            raise ValueError(f"Unexpected or duplicate task in eval_info.json: {key}")
        observed.add(key)
        metrics = task["metrics"]
        successes = metrics["successes"]
        require_same(len(successes), len(groups[key]), f"{key} case count")
        rewards = metrics.get("sum_rewards", [None] * len(successes))
        require_same(len(rewards), len(successes), f"{key} reward count")
        for case, ok, reward in zip(groups[key], successes, rewards, strict=True):
            if type(ok) is not bool:
                raise ValueError("lerobot-eval successes must be booleans")
            records.append(
                {
                    "case_id": case["case_id"],
                    "scenario_id": case["scenario_id"],
                    "case_seed": case["seed"],
                    "attempt": 1,
                    "outcome": "success" if ok else "failure",
                    "sum_reward": reward if number(reward) else None,
                }
            )
    require_same(observed, set(groups), "simulator task list")
    return _attempts(run, plan, records, {"kind": "lerobot-eval"})


def simulate(plan, run, output):
    _check_checkpoint(run, plan)
    cases, _ = _cases(plan)
    if cases["environment"] != "simulation":
        raise ValueError("Use import-rollouts for a real-robot case manifest")
    _, seeds = _simulation_groups(cases)
    options = cases["simulation_args"]
    if not options.get("env.type"):
        raise ValueError("simulation_args must include env.type")
    output = Path(output).resolve()
    if output.is_relative_to(Path(run["checkpoint"]).resolve()):
        raise ValueError("Simulation output must be outside the checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_eval",
        f"--policy.path={run['checkpoint']}",
        f"--output_dir={output / 'eval'}",
        f"--seed={seeds[0]}",
        f"--eval.n_episodes={len(seeds)}",
        "--eval.batch_size=1",
        "--eval.use_async_envs=false",
        "--trust_remote_code=false",
        "--env.max_parallel_tasks=1",
        *option_args(
            options,
            reserved={"policy.path", "env.max_parallel_tasks"},
            allowed_prefixes=("env.", "policy.", "rename_map"),
        ),
    ]
    write_json(output / "command.json", command)
    with (output / "eval.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "HF_HUB_OFFLINE": "1"}
        )
    if result.returncode:
        raise ValueError(f"Simulation failed; see {output / 'eval.log'}")
    info = output / "eval/eval_info.json"
    bundle = parse_simulation(read_json(info), plan, run)
    bundle["source"]["sha256"] = file_hash(info)
    write_json(output / "rollouts.json", bundle)
    return bundle
