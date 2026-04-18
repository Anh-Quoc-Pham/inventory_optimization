from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in Python path when running from scripts/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.chapter3_benchmark import run_chapter3_benchmark
from src.config import (
    GA_CROSSOVER_PROB,
    GA_GENERATIONS,
    GA_MUTATION_PROB,
    GA_POPULATION_SIZE,
    MIN_INVENTORY_DAYS,
    MAX_INVENTORY_DAYS,
    RANDOM_SEED,
    SIMPLEX_DUMMY_EXCESS_COST,
    SIMPLEX_DUMMY_SHORTAGE_COST,
    SIMPLEX_MAX_ITERATIONS,
    SIMPLEX_PIVOT_RULE,
    SIMPLEX_TOLERANCE,
    SIMPLEX_USE_BLAND,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Chapter 3 benchmark pipeline for inventory_optimization"
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--results-dir", type=str, default="results/chapter3_benchmark")
    parser.add_argument(
        "--vis-dir",
        type=str,
        default="visualizations/chapter3_benchmark",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)

    parser.add_argument("--min-days", type=int, default=MIN_INVENTORY_DAYS)
    parser.add_argument("--max-days", type=int, default=MAX_INVENTORY_DAYS)

    parser.add_argument("--ga-population", type=int, default=GA_POPULATION_SIZE)
    parser.add_argument("--ga-generations", type=int, default=GA_GENERATIONS)
    parser.add_argument("--ga-crossover", type=float, default=GA_CROSSOVER_PROB)
    parser.add_argument("--ga-mutation", type=float, default=GA_MUTATION_PROB)

    parser.add_argument(
        "--simplex-max-iterations",
        type=int,
        default=SIMPLEX_MAX_ITERATIONS,
    )
    parser.add_argument("--simplex-tolerance", type=float, default=SIMPLEX_TOLERANCE)
    parser.add_argument(
        "--simplex-pivot-rule",
        type=str,
        default=SIMPLEX_PIVOT_RULE,
        choices=["dantzig", "bland"],
    )
    parser.add_argument(
        "--simplex-use-bland",
        action="store_true",
        default=SIMPLEX_USE_BLAND,
    )
    parser.add_argument(
        "--simplex-no-bland",
        action="store_false",
        dest="simplex_use_bland",
    )
    parser.add_argument("--simplex-keep-history", action="store_true")
    parser.add_argument(
        "--simplex-dummy-excess-cost",
        type=float,
        default=SIMPLEX_DUMMY_EXCESS_COST,
    )
    parser.add_argument(
        "--simplex-dummy-shortage-cost",
        type=float,
        default=SIMPLEX_DUMMY_SHORTAGE_COST,
    )

    args = parser.parse_args()

    result = run_chapter3_benchmark(
        data_dir=Path(args.data_dir),
        results_dir=Path(args.results_dir),
        vis_dir=Path(args.vis_dir),
        seed=args.seed,
        min_days=args.min_days,
        max_days=args.max_days,
        ga_population=args.ga_population,
        ga_generations=args.ga_generations,
        ga_crossover=args.ga_crossover,
        ga_mutation=args.ga_mutation,
        simplex_max_iterations=args.simplex_max_iterations,
        simplex_tolerance=args.simplex_tolerance,
        simplex_pivot_rule=args.simplex_pivot_rule,
        simplex_use_bland=args.simplex_use_bland,
        simplex_keep_history=args.simplex_keep_history,
        simplex_dummy_excess_cost=args.simplex_dummy_excess_cost,
        simplex_dummy_shortage_cost=args.simplex_dummy_shortage_cost,
    )

    print("\nChapter 3 benchmark completed")
    print(f"Best algorithm: {result['best_algorithm']}")
    print("Artifacts:")
    for name, path in result["artifact_paths"].items():
        print(f"  - {name}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
