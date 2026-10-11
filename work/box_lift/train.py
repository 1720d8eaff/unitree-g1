"""Train a compact trajectory policy with cross-entropy optimization on CPU."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import csv
import hashlib
import json
import time
import numpy as np
from task import BoxLift, ROOT, LOW, HIGH, BASELINE, PARAM_NAMES


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def evaluate(env, params, seeds, **kwargs):
    records = [dict(seed=seed, **env.episode(params, seed=seed, **kwargs)) for seed in seeds]
    return dict(successes=sum(r["success"] for r in records), episodes=len(records),
                mean_reward=float(np.mean([r["reward"] for r in records])),
                mean_lift_m=float(np.mean([r["max_lift_m"] for r in records])),
                records=records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.generations < 1 or args.population < 4:
        parser.error("Use at least 1 generation and a population of 4.")
    env = BoxLift()
    rng = np.random.default_rng(args.seed)
    mean = BASELINE.copy()
    std = (HIGH - LOW) * 0.38
    best = BASELINE.copy()
    best_reward = -1e10
    pool = []
    history = []
    started = time.perf_counter()
    write_json(ROOT / "progress.json", dict(status="training", generation=0, generations=args.generations))
    try:
        for generation in range(args.generations):
            params = np.clip(rng.normal(mean, std, (args.population, len(mean))), LOW, HIGH)
            params[0] = best
            params[1] = mean
            results = []
            for i, candidate in enumerate(params):
                # Training seeds are disjoint from validation seeds below.
                seed = generation * args.population + i
                result = env.episode(candidate, seed=seed)
                results.append(result)
                history.append(dict(generation=generation + 1, candidate=i, seed=seed,
                                    **result, **dict(zip(PARAM_NAMES, candidate.tolist()))))
                if result["reward"] > best_reward:
                    best_reward, best = result["reward"], candidate.copy()
                write_json(ROOT / "progress.json", dict(status="training", generation=generation + 1,
                           generations=args.generations, candidate=i + 1, population=args.population,
                           best_reward=best_reward, elapsed_s=round(time.perf_counter() - started, 1)))
            order = np.argsort([r["reward"] for r in results])[::-1]
            elite = params[order[:max(3, args.population // 4)]]
            mean = 0.25 * mean + 0.75 * elite.mean(axis=0)
            std = np.maximum(0.30 * std + 0.70 * elite.std(axis=0), (HIGH - LOW) * 0.045)
            pool.append(params[order[0]].copy())
            print(f"Generation {generation + 1}/{args.generations}: "
                  f"success {sum(r['success'] for r in results)}/{args.population}, "
                  f"best reward {best_reward:.2f}", flush=True)
        with (ROOT / "training.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        print("Selecting policy on 3 separate development episodes...", flush=True)
        pool.extend([best, mean])
        candidates = []
        for p in pool:
            if not any(np.allclose(p, q) for q in candidates):
                candidates.append(p)
        selections = [(evaluate(env, p, range(500, 503)), p) for p in candidates]
        dev, learned = max(selections, key=lambda pair: (pair[0]["successes"], pair[0]["mean_reward"]))
        print("Validating baseline and learned policy on 12 unseen episodes...", flush=True)
        seeds = list(range(1000, 1012))
        baseline = evaluate(env, BASELINE, seeds)
        validation = evaluate(env, learned, seeds)
        controls = dict(no_grip=evaluate(env, learned, [2000, 2001], no_grip=True),
                        no_palm_contact=evaluate(env, learned, [2000, 2001], no_contact=True))
        report = dict(method="cross-entropy trajectory policy search (CEM)",
                      seed=args.seed, generations=args.generations, population=args.population,
                      training_episodes=len(history), selection=dev, baseline=baseline,
                      learned=validation, negative_controls=controls,
                      elapsed_s=round(time.perf_counter() - started, 2),
                      success_definition="box rises >= 0.10 m with both palm contacts continuously >= 3.0 s; no fall")
        write_json(ROOT / "evaluation.json", report)
        if validation["successes"] < 9 or any(c["successes"] for c in controls.values()):
            write_json(ROOT / "progress.json", dict(status="validation_failed", **report))
            raise RuntimeError("Validation failed; evaluation.json contains details. Existing policy was preserved.")
        policy = dict(version=1, method=report["method"],
                      model_sha256=hashlib.sha256((ROOT / "scene.mjb").read_bytes()).hexdigest(),
                      parameters=dict(zip(PARAM_NAMES, learned.tolist())),
                      validation_successes=validation["successes"], validation_episodes=validation["episodes"],
                      training_episodes=len(history))
        write_json(ROOT / "policy.json", policy)
        write_json(ROOT / "progress.json", dict(status="complete", **policy, elapsed_s=report["elapsed_s"]))
        print(json.dumps(dict(baseline=f"{baseline['successes']}/12", learned=f"{validation['successes']}/12",
                              negative_controls={k: f"{v['successes']}/{v['episodes']}" for k, v in controls.items()},
                              elapsed_s=report["elapsed_s"], parameters=policy["parameters"]), indent=2), flush=True)
    except KeyboardInterrupt:
        write_json(ROOT / "progress.json", dict(status="cancelled"))
        print("Training cancelled.", flush=True)


if __name__ == "__main__":
    main()
