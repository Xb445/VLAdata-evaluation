"""Freeze candidate subsets and run matched LeRobot training jobs."""

import os
import random
import re
import subprocess
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from .audit import validate_protocol
from .dataset import Dataset
from .io import digest, distinct, file_hash, positive, read_json, require_same, tree_hash, write_json

RUN_ID = re.compile(r"[A-Za-z0-9_-]+\Z")
TRAIN_RESERVED = {
    "steps",
    "batch_size",
    "seed",
    "output_dir",
    "resume",
    "save_checkpoint",
    "save_freq",
    "eval_freq",
    "config_path",
    "policy.path",
    "policy.pretrained_path",
    "policy.type",
    "policy.push_to_hub",
    "wandb.enable",
}


def option_args(options, *, reserved, allowed_prefixes):
    args = []
    for key, value in sorted(options.items()):
        if (
            not re.fullmatch(r"[A-Za-z0-9_.]+", key)
            or key in reserved
            or not key.startswith(allowed_prefixes)
        ):
            raise ValueError(f"Option is controlled by the protocol or unsupported: {key}")
        if isinstance(value, str):
            rendered = value
        else:
            import json

            rendered = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        args.append(f"--{key}={rendered}")
    return args


def checkpoint_hash(path):
    root = Path(path).resolve()
    files = [p for p in root.rglob("*") if p.is_file() and not p.name.startswith(".")]
    if not (root / "config.json").is_file() or not any(p.name == "model.safetensors" for p in files):
        raise ValueError(f"Expected a local full LeRobot checkpoint: {root}")
    return tree_hash(root, files)


def validate_recipe(recipe):
    for name in ("steps", "batch_size"):
        if type(recipe[name]) is not int:
            raise ValueError(f"{name} must be an integer")
        positive(recipe[name], name)
    seeds = recipe["seeds"]
    if (
        not seeds
        or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Training seeds must be distinct nonnegative integers")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", recipe["repo_id"]):
        raise ValueError("Use an owner/name dataset repo_id, even for a local dataset")
    option_args(
        recipe.get("train_args", {}),
        reserved=TRAIN_RESERVED,
        allowed_prefixes=("policy.", "num_workers", "log_freq", "tolerance_s", "rename_map"),
    )


def validate_cases(cases, audit, config):
    if cases["split"] not in {"dev", "test"} or cases["environment"] not in {"simulation", "real"}:
        raise ValueError("Cases must declare a dev/test split and simulation/real environment")
    if not cases.get("execution_protocol") or not cases.get("success_criteria"):
        raise ValueError("Declare execution protocol and success criteria")
    indexed = distinct(cases["cases"], "case_id")
    require_same(
        {row["task_id"] for row in indexed.values()}, set(config["task_weights"]), "task coverage in cases"
    )
    forbidden_splits = {"train", "dev"} if cases["split"] == "test" else {"train"}
    forbidden_groups = {r["source_group"] for r in audit["episodes"] if r["split"] in forbidden_splits}
    distinct(({"key": (c["task_id"], c["scenario_id"], c["seed"])} for c in indexed.values()), "key")
    for case in indexed.values():
        if any(
            not isinstance(case.get(k), str) or not case[k].strip()
            for k in ("case_id", "cluster_id", "scenario_id", "source_group")
        ):
            raise ValueError("Each case requires IDs for case, cluster, scenario and source group")
        if case["source_group"] in forbidden_groups:
            raise ValueError(f"Case {case['case_id']} shares a source group with an earlier split")
        if type(case["seed"]) is not int or not 0 <= case["seed"] < 2**32:
            raise ValueError("Each case needs an integer seed")
    return indexed


def make_plan(audit, config, recipe, cases=None):
    validate_protocol(config)
    validate_recipe(recipe)
    recipe = {**recipe, "pretrained_path": str(Path(recipe["pretrained_path"]).resolve())}
    require_same(audit["protocol_sha"], digest(config), "audit protocol")
    if cases is not None:
        validate_cases(cases, audit, config)
    eligible = [r for r in audit["episodes"] if r["split"] == "train" and r["eligible"]]
    excluded = set(config["clean_exclude_codes"])
    clean = [r for r in eligible if not excluded.intersection(r["reason_codes"])]
    if not eligible or not clean:
        raise ValueError("Baseline and cleaned subsets need eligible training episodes")
    rng = random.Random(recipe.get("selection_seed", 0))
    quota = Counter(r["coverage_cell"] for r in clean)
    pools = {cell: [r for r in eligible if r["coverage_cell"] == cell] for cell in quota}
    clean_ids = {r["episode_index"] for r in clean}
    alternative_exists = any(len(pools[cell]) > n for cell, n in quota.items())
    # Conditional random draw with identical coverage quotas and a distinct episode list.
    for _ in range(128):
        matched = [r for cell, n in sorted(quota.items()) for r in rng.sample(pools[cell], n)]
        if not alternative_exists or {r["episode_index"] for r in matched} != clean_ids:
            break
    else:
        raise ValueError("Choose another selection_seed to generate a distinct random control")
    variants = {"baseline": eligible, "clean": clean, "matched_random": matched}
    for name, groups in recipe.get("exclude_source_groups", {}).items():
        if name in variants or not RUN_ID.fullmatch(name) or not isinstance(groups, list):
            raise ValueError(f"Invalid source ablation name or groups: {name}")
        subset = [r for r in eligible if r["source_group"] not in groups]
        if not subset or len(subset) == len(eligible):
            raise ValueError(f"Ablation {name} must remove some and keep some training data")
        variants[name] = subset
    if not any(r["eligible"] and r["split"] == "dev" for r in audit["episodes"]):
        raise ValueError("A frozen eligible dev episode is required for offline diagnostics")
    plan = {
        "version": 1,
        "audit_sha": digest(audit),
        "protocol_sha": digest(config),
        "dataset_sha": audit["dataset_sha"],
        "recipe": recipe,
        "cases": cases,
        "initial_checkpoint_sha": checkpoint_hash(recipe["pretrained_path"]),
        "normalization": "aggregate_selected_training_episodes",
        "variants": {},
        "reference_episodes": {
            split: [r["episode_index"] for r in audit["episodes"] if r["eligible"] and r["split"] == split]
            for split in ("dev", "test")
        },
    }
    for name, subset in variants.items():
        plan["variants"][name] = {
            "episode_indices": sorted(r["episode_index"] for r in subset),
            "frames": sum(r["n_frames"] for r in subset),
            "coverage_cells": dict(Counter(r["coverage_cell"] for r in subset)),
        }
    return plan


def _snapshot(dataset, plan, variant, output):
    from lerobot.datasets.dataset_tools import split_dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    subset_dir = output / "datasets" / variant
    manifest_path = output / "datasets" / f"{variant}.json"
    source_ids = plan["variants"][variant]["episode_indices"]
    repo_id = plan["recipe"]["repo_id"]
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        require_same(manifest["plan_sha"], digest(plan), "materialized subset plan")
        require_same(manifest["source_episode_indices"], source_ids, "materialized episode list")
        require_same(Dataset(subset_dir).fingerprint(), manifest["dataset_sha"], "materialized dataset")
    else:
        if subset_dir.exists():
            raise ValueError(f"Incomplete materialization at {subset_dir}; choose a new output directory")
        source = LeRobotDataset(repo_id, root=dataset.root, episodes=source_ids, video_backend="pyav")
        split_dataset(source, {variant: source_ids}, output_dir=subset_dir.parent)
        stats = read_json(subset_dir / "meta/stats.json")
        for key in ("action", "observation.state"):
            if not stats.get(key):
                raise ValueError(f"Missing selected-episode statistics for {key}")
        manifest = {
            "plan_sha": digest(plan),
            "source_episode_indices": source_ids,
            "dataset_sha": Dataset(subset_dir).fingerprint(),
        }
        write_json(manifest_path, manifest)
    return subset_dir, f"{repo_id}_{variant}", manifest["dataset_sha"]


def training_command(plan, seed, subset, repo_id, output):
    recipe = plan["recipe"]
    return [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--dataset.root={subset}",
        f"--dataset.repo_id={repo_id}",
        "--dataset.video_backend=pyav",
        "--dataset.use_imagenet_stats=false",
        f"--policy.path={Path(recipe['pretrained_path']).resolve()}",
        f"--steps={recipe['steps']}",
        f"--batch_size={recipe['batch_size']}",
        f"--seed={seed}",
        f"--output_dir={output}",
        "--resume=false",
        "--save_checkpoint=true",
        f"--save_freq={recipe['steps']}",
        "--eval_freq=0",
        "--policy.push_to_hub=false",
        "--wandb.enable=false",
        *option_args(
            recipe.get("train_args", {}),
            reserved=TRAIN_RESERVED,
            allowed_prefixes=("policy.", "num_workers", "log_freq", "tolerance_s", "rename_map"),
        ),
    ]


def validate_run(run, plan):
    require_same(run["plan_sha"], digest(plan), "run plan")
    if run["variant"] not in plan["variants"] or run["seed"] not in plan["recipe"]["seeds"]:
        raise ValueError("Run variant or seed is outside the frozen plan")
    require_same(run["training_steps"], plan["recipe"]["steps"], "run training steps")


def train(root, plan, output, variants=None, seeds=None):
    dataset = Dataset(root)
    require_same(dataset.fingerprint(), plan["dataset_sha"], "audited dataset")
    recipe = plan["recipe"]
    validate_recipe(recipe)
    require_same(
        checkpoint_hash(recipe["pretrained_path"]), plan["initial_checkpoint_sha"], "initial checkpoint"
    )
    output = Path(output).resolve()
    if output.is_relative_to(dataset.root) or output.is_relative_to(
        Path(recipe["pretrained_path"]).resolve()
    ):
        raise ValueError("Training output must be outside the input dataset and checkpoint")
    variant_names = variants or list(plan["variants"])
    seed_values = seeds or recipe["seeds"]
    if not set(variant_names) <= plan["variants"].keys() or not set(seed_values) <= set(recipe["seeds"]):
        raise ValueError("Selected variants or seeds are outside the plan")
    from importlib.util import find_spec

    source_root = Path(find_spec("lerobot").origin).resolve().parent
    runtime = {key: version(key) for key in ("lerobot", "torch", "torchvision", "datasets", "pyarrow", "av")}
    runtime["lerobot_source_sha"] = tree_hash(source_root, list(source_root.rglob("*.py")))
    manifests = []
    for variant in variant_names:
        subset, repo_id, subset_sha = _snapshot(dataset, plan, variant, output)
        for seed in seed_values:
            run_dir = output / "runs" / variant / str(seed)
            path = run_dir / "run.json"
            if path.is_file():
                previous = read_json(path)
                validate_run(previous, plan)
                require_same((previous["variant"], previous["seed"]), (variant, seed), "run identity")
                require_same(previous["training_runtime"], runtime, "existing training runtime")
                require_same(
                    checkpoint_hash(previous["checkpoint"]), previous["checkpoint_sha"], "existing checkpoint"
                )
                manifests.append(path)
                continue
            run_dir.mkdir(parents=True, exist_ok=False)
            command = training_command(plan, seed, subset, repo_id, run_dir / "training")
            write_json(run_dir / "command.json", command)
            with (run_dir / "train.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "HF_HUB_OFFLINE": "1"}
                )
            if result.returncode:
                raise ValueError(f"Training failed for {variant}/{seed}; see {run_dir / 'train.log'}")
            final = run_dir / "training/checkpoints/last/pretrained_model"
            checkpoint = final.resolve()
            saved = read_json(checkpoint / "train_config.json")
            require_same(
                (saved["steps"], saved["batch_size"], saved["seed"]),
                (recipe["steps"], recipe["batch_size"], seed),
                "saved training budget",
            )
            run = {
                "version": 1,
                "plan_sha": digest(plan),
                "variant": variant,
                "seed": seed,
                "training_steps": recipe["steps"],
                "checkpoint": str(checkpoint),
                "checkpoint_sha": checkpoint_hash(checkpoint),
                "subset_sha": subset_sha,
                "stats_sha": file_hash(subset / "meta/stats.json"),
                "training_runtime": runtime,
            }
            write_json(path, run)
            manifests.append(path)
    return manifests
