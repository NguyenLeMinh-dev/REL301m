"""Supervised mean-action MSE using SB3's native SAC actor and checkpoint."""
from common import device, git_sha, package_versions, sha256, verify_stack, write_json
from dataset import load_dataset, split_episodes
from env import make_lift_env, environment_contract

import argparse
import csv
import hashlib
import shutil
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import SAC
from tqdm import tqdm


def state_hash(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode()); digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def make_model(env, seed=0, learning_rate=3e-4, model_device="cpu"):
    # Native SAC policy/optimizers/checkpoint. BC never calls learn() or fills replay.
    return SAC("MlpPolicy", env, policy_kwargs={"net_arch": [256, 256]}, seed=seed,
               learning_rate=learning_rate, buffer_size=100_000, batch_size=256,
               device=model_device, verbose=0)


def batch_metrics(model, observations, actions, indices, batch_size, *, train):
    model.actor.set_training_mode(train)
    total = np.zeros(3, dtype=np.float64)
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start+batch_size]
        obs = torch.as_tensor(observations[batch], device=model.device)
        targets = torch.as_tensor(actions[batch], device=model.device)
        with torch.set_grad_enabled(train):
            pred = model.actor(obs, deterministic=True)
            squared = (pred-targets).square()
            loss = squared.mean()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite BC loss")
            if train:
                model.actor.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.actor.parameters()):
                    raise ValueError("Non-finite BC gradient")
                model.actor.optimizer.step()
        total += np.array([squared.detach().sum().item(), squared[:, :6].detach().sum().item(), squared[:, 6].detach().sum().item()])
    count = len(indices)
    return {"loss": float(total[0]/(count*7)), "arm_mse": float(total[1]/(count*6)), "gripper_mse": float(total[2]/count)}


def train(args):
    verify_stack()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.patience < 0:
        raise ValueError("Invalid BC hyperparameters")
    if args.run_dir.exists():
        raise FileExistsError(f"Use a new run directory: {args.run_dir}")
    data, metadata = load_dataset(args.dataset)
    train_eps, val_eps, train_indices, val_indices = split_episodes(data, args.seed)
    env = make_lift_env(args.seed, assist_distance=metadata["assist_distance"])
    try:
        contract = environment_contract(env, assist_distance=metadata["assist_distance"])
        if metadata["environment"] != contract:
            raise ValueError("Dataset/live environment contract mismatch")
        model = make_model(env, args.seed, args.learning_rate, device(args.device))
        if not isinstance(model.actor.optimizer, torch.optim.Adam):
            raise TypeError("SAC actor optimizer is not Adam")
        critic_before = state_hash(model.critic)
        args.run_dir.mkdir(parents=True)
        checkpoints = args.run_dir / "checkpoints"; checkpoints.mkdir()
        summary = {"status": "training", "algorithm": "BC MSE on native SB3 SAC deterministic actor",
                   "number_of_demonstrations": metadata["number_of_successful_episodes"],
                   "number_of_transitions": metadata["number_of_transitions"],
                   "source_policy_checkpoint": metadata["source_model_path"], "source_policy_sha256": metadata["source_model_sha256"],
                   "source_policy_success_rate_during_collection": metadata["source_policy_success_rate"],
                   "dataset_path": str(args.dataset.resolve()), "dataset_sha256": sha256(args.dataset),
                   "collection_metadata": metadata, "environment": contract,
                   "train_episode_ids": train_eps.tolist(), "validation_episode_ids": val_eps.tolist(),
                   "train_transitions": len(train_indices), "validation_transitions": len(val_indices),
                   "split_seed": args.seed, "split_unit": "episode", "requested_train_fraction": 0.8,
                   "hyperparameters": {"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                                       "net_arch": [256, 256], "seed": args.seed, "patience": args.patience, "optimizer": "Adam"},
                   "git_commit": git_sha(), "package_versions": package_versions(), "device": str(model.device),
                   "evaluation_success": None, "critic_trained": False,
                   "notes": "Critics/temperature are untouched. Std head is not directly supervised; hidden-feature changes can alter std outputs. No RL fine-tuning run."}
        model.bc_bootstrap_metadata = summary
        write_json(args.run_dir/"summary.json", summary)
        rng = np.random.default_rng(args.seed)
        best, best_epoch, stale = float("inf"), 0, 0
        fields = ["epoch", "train_loss", "val_loss", "train_arm_mse", "val_arm_mse", "train_gripper_mse", "val_gripper_mse"]
        with (args.run_dir/"history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            with tqdm(range(1, args.epochs+1), desc="BC", unit="epoch", dynamic_ncols=True) as progress:
                for epoch in progress:
                    shuffled = rng.permutation(train_indices)
                    tr = batch_metrics(model, data["observations"], data["actions"], shuffled, args.batch_size, train=True)
                    val = batch_metrics(model, data["observations"], data["actions"], val_indices, args.batch_size, train=False)
                    row = {"epoch": epoch, "train_loss": tr["loss"], "val_loss": val["loss"],
                           "train_arm_mse": tr["arm_mse"], "val_arm_mse": val["arm_mse"],
                           "train_gripper_mse": tr["gripper_mse"], "val_gripper_mse": val["gripper_mse"]}
                    writer.writerow(row); stream.flush()
                    improved = val["loss"] < best
                    if improved:
                        best, best_epoch, stale = val["loss"], epoch, 0
                    else:
                        stale += 1
                    summary.update(best_validation_loss=best, best_epoch=best_epoch, completed_epochs=epoch)
                    if improved:
                        model.save(checkpoints/"best_bc.zip")
                    model.save(checkpoints/"last_bc.zip")
                    write_json(args.run_dir/"summary.json", summary)
                    progress.set_postfix(train_loss=f"{tr['loss']:.6f}", val_loss=f"{val['loss']:.6f}", arm_mse=f"{val['arm_mse']:.6f}", grip_mse=f"{val['gripper_mse']:.6f}")
                    if args.patience and stale >= args.patience:
                        tqdm.write(f"EARLY_STOP epoch={epoch} patience={args.patience}")
                        break
        if state_hash(model.critic) != critic_before or any(p.grad is not None for p in model.critic.parameters()):
            raise RuntimeError("BC unexpectedly changed or differentiated the critic")
        # Check same-device deterministic action preservation separately from
        # any stochastic policy/teacher equivalence claim.
        obs, _ = env.reset()
        last_expected, _ = model.predict(obs, deterministic=True)
        last_restored = SAC.load(checkpoints/"last_bc.zip", env=env, device=model.device)
        last_actual, _ = last_restored.predict(obs, deterministic=True)
        np.testing.assert_array_equal(last_actual, last_expected)
        best_model = SAC.load(checkpoints/"best_bc.zip", env=env, device=model.device)
        shutil.copyfile(checkpoints/"best_bc.zip", checkpoints/"bc_sac_warmstart.zip")
        torch.save(best_model.actor.state_dict(), checkpoints/"bc_actor_state_dict.pt")
        obs, _ = env.reset()
        predicted, _ = best_model.predict(obs, deterministic=True)
        if predicted.dtype != np.float32 or not env.action_space.contains(predicted):
            raise ValueError("Reloaded BC action outside live space")
        summary.update(status="completed", critic_unchanged=True, deterministic_save_reload_preserved=True, deterministic_save_reload_max_abs_error=float(np.max(np.abs(last_actual-last_expected))), checkpoint_format="Stable-Baselines3 SAC .zip",
                       best_checkpoint_sha256=sha256(checkpoints/"best_bc.zip"))
        write_json(args.run_dir/"summary.json", summary)
        print(f"BC_TRAIN=PASS epochs={summary['completed_epochs']} best_epoch={best_epoch} best_val_loss={best:.6f} train_episodes={len(train_eps)} val_episodes={len(val_eps)}")
        print(f"BC_CHECKPOINT={checkpoints/'bc_sac_warmstart.zip'} critic_trained=False")
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=20, help="0 disables early stopping")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--run-dir", type=Path, default=Path("runs/bc_lift_50demo_seed0"))
    train(p.parse_args())


if __name__ == "__main__":
    main()
