"""Aggregate technical quality, coverage, policy performance and paired effects."""

import random
from collections import defaultdict
from math import sqrt
from statistics import mean

from .evaluate import _attempts
from .experiments import validate_run
from .io import digest, number, require_same, weights, write_json, write_jsonl, write_text


def wilson(successes, n):
    if not n:
        return None
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    half = z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return [100 * max(0, center - half), 100 * min(1, center + half)]


def _percentile(values, q):
    ordered = sorted(values)
    index = q * (len(ordered) - 1)
    lo = int(index)
    return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (index - lo)


def _quality(episodes, config):
    known = [row for row in episodes if number(row["quality"])]
    groups = defaultdict(set)
    for row in episodes:
        groups[row["coverage_cell"]].add(row["source_group"])
    cells = config["coverage"]
    cell_weights = weights({name: c["weight"] for name, c in cells.items()}, "coverage weights")
    by_cell = {
        name: {
            "source_groups": len(groups[name]),
            "quota": c["quota"],
            "attainment": min(1, len(groups[name]) / c["quota"]),
        }
        for name, c in cells.items()
    }
    coverage = 100 * sum(cell_weights[name] * cell["attainment"] for name, cell in by_cell.items())
    return {
        "q": mean(r["quality"] for r in known) if len(known) == len(episodes) else None,
        "q_known_episodes": len(known),
        "q_p10_known": _percentile([r["quality"] for r in known], 0.10) if known else None,
        "q_frame_weighted_known": (
            sum(r["quality"] * r["n_frames"] for r in known) / sum(r["n_frames"] for r in known)
        )
        if known
        else None,
        "c": coverage,
        "coverage_cells": by_cell,
    }


def _bundle_index(bundles, plan):
    out = {}
    for bundle in bundles:
        run = bundle["run"]
        validate_run(run, plan)
        key = run["variant"], run["seed"]
        if key in out:
            raise ValueError(f"Duplicate evidence for {key}")
        out[key] = bundle
    return out


def _policy_results(variant, plan, bundles, task_weights):
    seeds = plan["recipe"]["seeds"]
    cases = plan.get("cases")
    if cases is None:
        return None, {}, {}
    task_cases = defaultdict(list)
    for case in cases["cases"]:
        task_cases[case["task_id"]].append(case["case_id"])
    seeds_report = {}
    completed = {}
    for seed in seeds:
        bundle = bundles.get((variant, seed))
        if bundle is None:
            continue
        require_same(bundle["cases_sha"], digest(cases), "rollout cases")
        require_same(
            (bundle["environment"], bundle["split"]),
            (cases["environment"], cases["split"]),
            "evaluation environment and split",
        )
        records = _attempts(bundle["run"], plan, bundle["attempts"], bundle["source"])["attempts"]
        latest = {r["case_id"]: r for r in records}
        resolved = {
            k: r["outcome"] == "success" for k, r in latest.items() if r["outcome"] != "infra_invalid"
        }
        tasks = {}
        for task, ids in task_cases.items():
            valid = [resolved[i] for i in ids if i in resolved]
            k = sum(valid)
            tasks[task] = {
                "successes": k,
                "n": len(valid),
                "expected": len(ids),
                "success_pct": 100 * k / len(valid) if valid else None,
                "wilson95_pct": wilson(k, len(valid)),
            }
        seeds_report[str(seed)] = {
            "tasks": tasks,
            "attempts": len(records),
            "infra_invalid": sum(r["outcome"] == "infra_invalid" for r in records),
            "pending": sorted({c["case_id"] for c in cases["cases"]} - resolved.keys()),
        }
        if not seeds_report[str(seed)]["pending"]:
            completed[seed] = resolved
    p = None
    if set(completed) == set(seeds):
        p = mean(
            sum(task_weights[t] * 100 * mean(rows[i] for i in ids) for t, ids in task_cases.items())
            for rows in completed.values()
        )
    return p, seeds_report, completed


def _paired_interval(baseline, candidate, cases, task_weights, *, draws=1000):
    """Resample matched training seeds and case clusters within task strata."""
    seeds = sorted(baseline)
    strata = defaultdict(lambda: defaultdict(list))
    for case in cases["cases"]:
        strata[case["task_id"]][case["cluster_id"]].append(case["case_id"])
    if len(seeds) < 2 or any(len(groups) < 2 for groups in strata.values()):
        return None
    clusters = {task: list(groups.values()) for task, groups in strata.items()}
    delta = {
        seed: {
            case["case_id"]: int(candidate[seed][case["case_id"]]) - int(baseline[seed][case["case_id"]])
            for case in cases["cases"]
        }
        for seed in seeds
    }
    rng = random.Random(2026)
    samples = []
    for _ in range(draws):
        selected_seeds = rng.choices(seeds, k=len(seeds))
        total = 0.0
        for task, groups in clusters.items():
            ids = [i for group in rng.choices(groups, k=len(groups)) for i in group]
            total += task_weights[task] * mean(delta[s][i] for s in selected_seeds for i in ids)
        samples.append(100 * total)
    return [_percentile(samples, 0.025), _percentile(samples, 0.975)]


def report(audit, config, plan, offline_bundles, rollout_bundles):
    require_same(digest(audit), plan["audit_sha"], "audit result")
    require_same(digest(config), plan["protocol_sha"], "report protocol")
    offline = _bundle_index(offline_bundles, plan)
    rollouts = _bundle_index(rollout_bundles, plan)
    runtimes = {
        digest(bundle["run"]["training_runtime"]) for bundle in [*offline.values(), *rollouts.values()]
    }
    if len(runtimes) > 1:
        raise ValueError("Evidence combines different training runtimes")
    windows_fingerprint = None
    for key, result in offline.items():
        if key in rollouts:
            require_same(
                result["run"]["checkpoint_sha"],
                rollouts[key]["run"]["checkpoint_sha"],
                "offline/rollout checkpoint",
            )
        windows = result["windows"]
        ids = {(r["episode_index"], r["frame_index"], r["draw"]) for r in windows}
        if len(ids) != len(windows) or not windows:
            raise ValueError("Offline windows must be nonempty and unique")
        require_same(
            {r[0] for r in ids},
            set(plan["reference_episodes"][result["split"]]),
            "offline reference episodes",
        )
        scope = digest([result["split"], result["stride"], result["draws"], sorted(ids)])
        if windows_fingerprint is None:
            windows_fingerprint = scope
        require_same(scope, windows_fingerprint, "offline window protocol")

    task_weights = weights(config["task_weights"], "task weights")
    episodes = {r["episode_index"]: r for r in audit["episodes"]}
    all_results = {}
    closed = {}
    seeds = plan["recipe"]["seeds"]
    train_count = sum(r["split"] == "train" for r in episodes.values())
    for name, definition in plan["variants"].items():
        selected = [episodes[i] for i in definition["episode_indices"]]
        if any(not r["eligible"] or r["split"] != "train" for r in selected):
            raise ValueError("Plan includes quarantined or held-out episodes")
        result = _quality(selected, config)
        p, seed_details, completed = _policy_results(name, plan, rollouts, task_weights)
        closed[name] = completed
        local_offline = {
            str(s): offline[name, s]["episode_macro_scaled_mae"] for s in seeds if (name, s) in offline
        }
        has_offline = len(local_offline) == len(seeds)
        evidence = "E2" if has_offline and p is not None else "E1" if has_offline else "E0"
        q, c = result["q"], result["c"]
        result.update(
            {
                "episodes": len(selected),
                "frames": definition["frames"],
                "retained_train_fraction": len(selected) / train_count,
                "p": p,
                "u_pp": None,
                "u_ci95_pp": None,
                "composite": 0.25 * q + 0.15 * c + 0.60 * p
                if q is not None and p is not None and has_offline
                else None,
                "offline_scaled_mae_by_seed": local_offline,
                "closed_loop_by_seed": seed_details,
                "evidence_level": evidence,
            }
        )
        all_results[name] = result

    baseline = all_results["baseline"]
    cases = plan.get("cases")
    for name, result in all_results.items():
        if name == "baseline" or result["p"] is None or baseline["p"] is None:
            continue
        result["u_pp"] = result["p"] - baseline["p"]
        result["u_ci95_pp"] = _paired_interval(closed["baseline"], closed[name], cases, task_weights)
        if (
            cases["split"] == "test"
            and result["u_ci95_pp"] is not None
            and len(seeds) >= config["min_training_seeds"]
            and result["evidence_level"] == baseline["evidence_level"] == "E2"
        ):
            result["evidence_level"] = "E3"

    clean, matched = all_results["clean"], all_results["matched_random"]
    selection_effect = None
    distinct_random_control = (
        plan["variants"]["clean"]["episode_indices"] != plan["variants"]["matched_random"]["episode_indices"]
    )
    if distinct_random_control and clean["p"] is not None and matched["p"] is not None:
        selection_effect = {
            "delta_pp": clean["p"] - matched["p"],
            "ci95_pp": _paired_interval(closed["matched_random"], closed["clean"], cases, task_weights),
        }
    return {
        "version": 1,
        "protocol_id": config["protocol_id"],
        "protocol_sha": plan["protocol_sha"],
        "plan_sha": digest(plan),
        "dataset_sha": plan["dataset_sha"],
        "environment": cases["environment"] if cases else None,
        "closed_loop_split": cases["split"] if cases else None,
        "frozen_training_seeds": seeds,
        "normalization": plan["normalization"],
        "variants": all_results,
        "clean_minus_matched_random": selection_effect,
        "distinct_random_control": distinct_random_control,
        "n_episodes": len(episodes),
        "n_quarantined": sum(not r["eligible"] for r in episodes.values()),
    }


def write_report(output, result, audit):
    def show(value):
        return "未评估" if value is None else f"{value:.2f}"

    lines = [
        "# VLA 训练数据评估",
        "",
        f"评估协议 `{result['protocol_id']}`",
        "",
        f"案例环境 {result['environment'] or '未配置'} · 阶段 {result['closed_loop_split'] or '未配置'}",
        "",
        f"数据共 {result['n_episodes']} 条轨迹，隔离 {result['n_quarantined']} 条。",
        "",
        "| 数据版本 | 轨迹数 | Q | C | P % | 相对基线 U pp | 总分 | 证据 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, r in result["variants"].items():
        lines.append(
            f"| {name} | {r['episodes']} | {show(r['q'])} | {show(r['c'])} | "
            f"{show(r['p'])} | {show(r['u_pp'])} | {show(r['composite'])} | {r['evidence_level']} |"
        )
    lines += [
        "",
        "各训练种子的逐任务成功次数、Wilson 区间及配对差值区间见 score_report.json。",
        "查看 review_queue.jsonl 可定位异常轨迹和首个异常帧。",
        "",
    ]
    write_json(output / "score_report.json", result)
    write_text(output / "report.md", "\n".join(lines))
    write_jsonl(output / "episode_metrics.jsonl", audit["episodes"])
    write_jsonl(output / "review_queue.jsonl", audit["review"])
