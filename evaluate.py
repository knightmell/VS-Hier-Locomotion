"""Evaluate Full and Base View with identical environment/policy seed sets."""
from pathlib import Path
import argparse
import json
from omegaconf import OmegaConf
from p6_train import _make_locomotion_environment
from p6_evaluation import evaluate_exact_episodes
from policy_loader import load_base_policy
from stable_baselines3.dsrl.hierarchical_rfs_dsrl import HierarchicalRFSDSRL

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["hopper", "walker", "halfcheetah"])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--asset-dir", type=Path, default=ROOT / "assets/pretrained")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["full", "base", "both"], default="both")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("episodes must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = OmegaConf.load(ROOT / "cfg/gym" / (args.task + ".yaml"))
    cfg.asset_dir = str(args.asset_dir.resolve())
    cfg.device = args.device
    OmegaConf.resolve(cfg)
    model = HierarchicalRFSDSRL.load(str(args.checkpoint.resolve()), device=args.device,
                                   diffusion_policy=load_base_policy(cfg))
    modes = ["full", "base"] if args.mode == "both" else [args.mode]
    result = {}
    for mode in modes:
        result[mode] = evaluate_exact_episodes(
            model=model,
            make_environment=lambda: _make_locomotion_environment(cfg, Path(cfg.normalization_path)),
            environment_seeds=list(range(10000, 10000 + args.episodes)),
            policy_seed_start=20000, deterministic=bool(cfg.deterministic_eval),
            action_chunk=int(cfg.act_steps), max_episode_primitive_steps=int(cfg.env.max_episode_steps),
            chunk_transitions=int(model.num_timesteps),
            nominal_primitive_steps=int(model.num_timesteps) * int(cfg.act_steps),
            evaluation_mode="current_full_hierarchy" if mode == "full" else "current_base_only")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v["summary"] for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    main()
