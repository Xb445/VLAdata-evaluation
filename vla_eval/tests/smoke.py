"""Exercise the CLI with a tiny native v3 dataset, ACT training and PushT simulation."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from vla_eval.io import read_json, write_json, write_jsonl


def fixture(output):
    import numpy as np
    import torch

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    torch.manual_seed(14)
    camera = "observation.image"
    features = {
        "action": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
        camera: {"dtype": "video", "shape": (64, 64, 3), "names": ["height", "width", "channel"]},
    }
    root = output / "dataset"
    dataset = LeRobotDataset.create(
        "local/vla_eval_smoke",
        root=root,
        fps=10,
        features=features,
        robot_type="pusht",
        vcodec="h264",
        video_backend="pyav",
    )
    assignments, timings = [], []
    for index in range(7):
        split = "train" if index < 5 else "dev" if index == 5 else "test"
        assignments.append(
            {
                "episode_index": index,
                "source_group": f"capture-{index}",
                "split": split,
                "task_id": "push",
                "coverage_cell": "push/easy",
                "action_verified": index != 4,
                "language_verified": True,
            }
        )
        for frame in range(6):
            image = np.full((64, 64, 3), 70, dtype=np.uint8)
            image[12:28, 3 + frame * 4 : 20 + frame * 4] = (180, 135, 220)
            state = np.array([160 + 4 * frame + index, 160 + index], dtype=np.float32)
            action = state + np.array([10, 12], dtype=np.float32)
            if index == 3 and frame == 3:
                action[0] += 150
            dataset.add_frame(
                {
                    "observation.state": state,
                    "action": action,
                    camera: image,
                    "task": "Push the T block toward the goal.",
                }
            )
            t = (index * 10 + 1) * 10**9 + frame * 10**8
            timings.append(
                {
                    "episode_index": index,
                    "frame_index": frame,
                    "decision_ns": t,
                    "state_sample_ns": t - 2_000_000,
                    "state_receive_ns": t - 1_000_000,
                    "command_issue_ns": t + 1_000_000,
                    "command_apply_ns": t + 3_000_000,
                    "image_capture_ns": {camera: t - 5_000_000},
                    "image_receive_ns": {camera: t - 2_000_000},
                }
            )
        dataset.save_episode()
    dataset.finalize()
    dims = [{"name": n, "unit": "px", "min": 0, "max": 512} for n in ("x", "y")]
    protocol = {
        "version": 1,
        "protocol_id": "cpu-smoke-v1",
        "camera_keys": [camera],
        "robot": {"id": "pusht", "action_space": "absolute_xy", "gripper_direction": "not_applicable"},
        "action_dimensions": [dict(d, scale=100, max_step=35) for d in dims],
        "state_dimensions": dims,
        "coverage": {"push/easy": {"task_id": "push", "quota": 4, "weight": 1}},
        "task_weights": {"push": 1},
        "clean_exclude_codes": ["ACTION_JUMP"],
        "min_training_seeds": 2,
        "quality": {
            "require_raw_timing": True,
            "grid_tolerance_s": 0.001,
            "black_mean_threshold": 10,
            "samples_per_camera": 2,
            "timing": {"age_ms": [10, 45], "camera_skew_ms": [5, 20], "command_delay_ms": [10, 50]},
        },
    }
    write_json(output / "protocol.json", protocol)
    write_jsonl(output / "episodes.jsonl", assignments)
    write_jsonl(output / "timing.jsonl", timings)
    checkpoint = output / "initial"
    policy_features = dataset_to_policy_features(dataset.meta.features)
    config = ACTConfig(
        device="cpu",
        chunk_size=3,
        n_action_steps=2,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        use_vae=False,
        pretrained_backbone_weights=None,
        input_features={k: policy_features[k] for k in (camera, "observation.state")},
        output_features={"action": policy_features["action"]},
        push_to_hub=False,
    )
    ACTPolicy(config).save_pretrained(checkpoint)
    pre, post = make_pre_post_processors(config)
    pre.save_pretrained(checkpoint)
    post.save_pretrained(checkpoint)
    write_json(
        output / "recipe.json",
        {
            "repo_id": "local/vla_eval_smoke",
            "pretrained_path": str(checkpoint),
            "steps": 2,
            "batch_size": 2,
            "seeds": [11, 12],
            "selection_seed": 7,
            "train_args": {"policy.device": "cpu", "num_workers": 0, "log_freq": 1},
        },
    )
    write_json(
        output / "cases.json",
        {
            "environment": "simulation",
            "split": "test",
            "execution_protocol": "PushT 4 steps, CPU, 384px render",
            "success_criteria": "gym-pusht is_success",
            "simulation_args": {
                "env.type": "pusht",
                "env.episode_length": 4,
                "env.visualization_width": 384,
                "env.visualization_height": 384,
                "env.observation_width": 64,
                "env.observation_height": 64,
                "policy.device": "cpu",
            },
            "cases": [
                {
                    "case_id": f"case-{seed}",
                    "task_id": "push",
                    "suite": "pusht",
                    "task_index": 0,
                    "seed": seed,
                    "scenario_id": f"PushT-v0:{seed}",
                    "cluster_id": f"reset-{seed}",
                    "source_group": f"simulation-{seed}",
                }
                for seed in (100, 101)
            ],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "SDL_VIDEODRIVER": "dummy",
        "SDL_AUDIODRIVER": "dummy",
    }

    def cli(*arguments):
        subprocess.run([sys.executable, "-m", "vla_eval", *map(str, arguments)], check=True, env=env)

    fixture(out)
    audit, protocol, plan = out / "audit.json", out / "protocol.json", out / "plan.json"
    cli(
        "audit",
        "--dataset",
        out / "dataset",
        "--protocol",
        protocol,
        "--episodes",
        out / "episodes.jsonl",
        "--raw-timing",
        out / "timing.jsonl",
        "--out",
        audit,
    )
    cli(
        "plan",
        "--audit",
        audit,
        "--protocol",
        protocol,
        "--recipe",
        out / "recipe.json",
        "--cases",
        out / "cases.json",
        "--out",
        plan,
    )
    experiment = out / "experiment"
    cli("train", "--dataset", out / "dataset", "--plan", plan, "--out", experiment)
    offline_files, rollout_files = [], []
    for manifest in sorted((experiment / "runs").glob("*/*/run.json")):
        offline = manifest.with_name("offline.json")
        sim = manifest.with_name("simulation")
        cli(
            "offline",
            "--dataset",
            out / "dataset",
            "--plan",
            plan,
            "--protocol",
            protocol,
            "--run",
            manifest,
            "--stride",
            5,
            "--out",
            offline,
        )
        cli("sim", "--plan", plan, "--run", manifest, "--out", sim)
        offline_files.append(offline)
        rollout_files.append(sim / "rollouts.json")
    cli(
        "report",
        "--audit",
        audit,
        "--protocol",
        protocol,
        "--plan",
        plan,
        "--offline",
        *offline_files,
        "--rollouts",
        *rollout_files,
        "--out",
        out / "report",
    )
    result = read_json(out / "report/score_report.json")
    assert result["n_quarantined"] == 1
    assert result["variants"]["baseline"]["episodes"] == 4
    assert result["variants"]["clean"]["episodes"] == result["variants"]["matched_random"]["episodes"] == 3
    assert result["distinct_random_control"]
    assert len(offline_files) == len(rollout_files) == 6
    assert all(v["p"] is not None for v in result["variants"].values())
    assert result["variants"]["clean"]["evidence_level"] == "E3"
    print(f"CPU integration smoke passed: {out / 'report/report.md'}")


if __name__ == "__main__":
    main()
