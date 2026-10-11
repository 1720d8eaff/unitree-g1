"""CEM search, held-out validation, and contact/transfer negative controls."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import csv
import hashlib
import json
import time
import numpy as np
from task import PickPlace, ROOT, LOW, HIGH, BASELINE, PARAM_NAMES


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def evaluate(env, params, seeds, **kwargs):
    records = [dict(seed=seed, **env.episode(params, seed=seed, **kwargs)) for seed in seeds]
    return dict(successes=sum(r["success"] for r in records), episodes=len(records),
                mean_reward=float(np.mean([r["reward"] for r in records])),
                mean_target_error_m=float(np.mean([r["target_error_m"] for r in records])), records=records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    if args.generations < 1 or args.population < 4:
        parser.error("Use at least 1 generation and a population of 4.")
    env = PickPlace()
    rng = np.random.default_rng(args.seed)
    mean = BASELINE.copy()
    mean[12] = (LOW[12] + HIGH[12]) / 2  # Start landing compensation at its search midpoint.
    std = (HIGH - LOW) * 0.32
    best = BASELINE.copy()
    best_reward = -1e10
    pool, history = [], []
    started = time.perf_counter()
    try:
        for generation in range(args.generations):
            candidates = np.clip(rng.normal(mean, std, (args.population, len(mean))), LOW, HIGH)
            candidates[0], candidates[1] = best, mean
            results = []
            for i, candidate in enumerate(candidates):
                seed = generation * args.population + i
                result = env.episode(candidate, seed=seed)
                results.append(result)
                history.append(dict(generation=generation + 1, candidate=i, seed=seed, **result,
                                    **dict(zip(PARAM_NAMES, candidate.tolist()))))
                if result["reward"] > best_reward:
                    best_reward, best = result["reward"], candidate.copy()
                write_json(ROOT / "progress.json", dict(status="training", generation=generation + 1,
                           generations=args.generations, candidate=i + 1, population=args.population,
                           best_reward=best_reward, elapsed_s=round(time.perf_counter() - started, 1)))
            order = np.argsort([r["reward"] for r in results])[::-1]
            elite = candidates[order[:max(3, args.population // 4)]]
            mean = 0.25 * mean + 0.75 * elite.mean(axis=0)
            std = np.maximum(0.30 * std + 0.70 * elite.std(axis=0), (HIGH - LOW) * 0.045)
            pool.append(candidates[order[0]].copy())
            print(f"Generation {generation + 1}/{args.generations}: "
                  f"placed {sum(r['success'] for r in results)}/{args.population}, "
                  f"best reward {best_reward:.2f}", flush=True)
        with (ROOT / "training.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        print("Selecting on separate development scenes...", flush=True)
        pool.extend([best, mean])
        unique = []
        for p in pool:
            if not any(np.allclose(p, other) for other in unique):
                unique.append(p)
        selections = [(evaluate(env, p, range(600, 603)), p) for p in unique]
        dev, learned = max(selections, key=lambda x: (x[0]["successes"], x[0]["mean_reward"]))
        write_json(ROOT / "progress.json", dict(status="validating", generation=args.generations,
                   generations=args.generations, elapsed_s=round(time.perf_counter() - started, 1)))
        print("Comparing baseline and policy on 20 unseen scenes...", flush=True)
        baseline = evaluate(env, BASELINE, range(3000, 3020))
        validation = evaluate(env, learned, range(3000, 3020))
        controls = {name: evaluate(env, learned, [4000, 4001], **flag) for name, flag in
                    [("no_grip", dict(no_grip=True)), ("no_palm_contact", dict(no_contact=True)),
                     ("no_transfer", dict(no_transfer=True))]}
        definition = ("lift >= 0.10 m, dual contact >= 0.5 s, then release on table within 0.035 m of target, "
                      "upright > 0.96, speed < 0.035 m/s and no palm contacts for >= 1.0 s; no fall")
        report = dict(method="cross-entropy trajectory parameter search (CEM)", seed=args.seed,
                      generations=args.generations, population=args.population, training_episodes=len(history),
                      selection=dev, baseline=baseline, learned=validation, negative_controls=controls,
                      elapsed_s=round(time.perf_counter() - started, 2), success_definition=definition)
        write_json(ROOT / "evaluation.json", report)
        if validation["successes"] < 17 or any(r["successes"] for r in controls.values()):
            write_json(ROOT / "progress.json", dict(status="validation_failed", **report))
            raise RuntimeError("Validation failed; previous policy preserved. See evaluation.json.")
        policy = dict(version=1, method=report["method"],
                      model_sha256=hashlib.sha256((ROOT / "scene.mjb").read_bytes()).hexdigest(),
                      parameters=dict(zip(PARAM_NAMES, learned.tolist())), validation_successes=validation["successes"],
                      validation_episodes=validation["episodes"], training_episodes=len(history))
        write_json(ROOT / "policy.json", policy)
        write_json(ROOT / "progress.json", dict(status="complete", **policy, elapsed_s=report["elapsed_s"]))
        print(json.dumps(dict(baseline=f"{baseline['successes']}/20", learned=f"{validation['successes']}/20",
                              target_error_cm=round(validation["mean_target_error_m"] * 100, 2),
                              negative_controls={k: f"{v['successes']}/2" for k, v in controls.items()},
                              elapsed_s=report["elapsed_s"], parameters=policy["parameters"]), indent=2), flush=True)
    except KeyboardInterrupt:
        write_json(ROOT / "progress.json", dict(status="cancelled"))
        raise SystemExit(130)


if __name__ == "__main__":
    main()
