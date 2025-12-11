"""
Iterative strategy discovery via constraint-based abstraction for Nim.
Discovers decision rules by finding pile value constraints that uniquely determine optimal actions.
"""

import operator
import numpy as np
from typing import Any, List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from itertools import combinations, product

from nim import Nim
from nim_oracle import NimOracle


@dataclass
class StateActionPair:
    state: Tuple[int, ...]
    optimal_actions: List[Tuple[int, int]]

    def action(self) -> Tuple[int, int]:
        return self.optimal_actions[0]


@dataclass
class DiscoveredStrategy:
    constraint: Dict[int, int]  # pile_index -> value
    action: Tuple[int, int]  # (pile_index, count)
    coverage: int

    def matches(self, state: Tuple[int, ...]) -> bool:
        for pile_idx, value in self.constraint.items():
            if state[pile_idx] != value:
                return False
        return True

    def __str__(self) -> str:
        pile_strs = [f"p{idx}={val}" for idx, val in sorted(self.constraint.items())]
        return f"({', '.join(pile_strs)}) -> remove {self.action[1]} from pile {self.action[0]}"


class NimGameGraphMiner:
    """Mine all P1 states where we can win."""

    def __init__(self, initial_piles: Tuple[int, ...] = (3, 4, 5)) -> None:
        self.initial_piles = initial_piles
        self.n_piles = len(initial_piles)
        self.oracle = NimOracle(initial_piles)

    def get_all_states(self, verbose: bool = True) -> List[StateActionPair]:
        if verbose:
            print("Phase 1: Mining All P1 Winning States")

        all_states = []
        for p0 in range(self.initial_piles[0] + 1):
            for p1 in range(self.initial_piles[1] + 1):
                for p2 in range(self.initial_piles[2] + 1):
                    state = (p0, p1, p2)
                    if not all(p == 0 for p in state):
                        all_states.append(state)

        result = []
        for state in all_states:
            val, _ = self.oracle.get_outcome(state, player=1)
            if val > 0:
                optimal = self.oracle.get_optimal_actions(state, player=1)
                if optimal:
                    result.append(StateActionPair(state, optimal))

        if verbose:
            print(
                f"Found {len(result)} P1 winning states (out of {len(all_states)} non-terminal states)."
            )

        return result


class NimIterativeStrategyDiscovery:
    """Discover strategies by finding pile value constraints that uniquely determine optimal actions."""

    def __init__(
        self,
        initial_piles: Tuple[int, ...] = (3, 4, 5),
        max_constraint_size: int = 3,
        verbose: bool = True,
    ) -> None:
        self.initial_piles = initial_piles
        self.n_piles = len(initial_piles)
        self.max_constraint_size = min(max_constraint_size, self.n_piles)
        self.verbose = verbose
        self.miner = NimGameGraphMiner(initial_piles)
        self.strategies: List[DiscoveredStrategy] = []

    def run(self) -> List[DiscoveredStrategy]:
        states = self.miner.get_all_states(verbose=self.verbose)
        uncovered = set(range(len(states)))

        if self.verbose:
            print(f"\nTotal states to cover: {len(states)}")
            print("\n" + "=" * 60)
            print("Iterative Strategy Discovery")
            print("=" * 60)

        for constraint_size in range(1, self.max_constraint_size + 1):
            if not uncovered:
                break

            if self.verbose:
                print(
                    f"\n--- Trying {constraint_size}-pile constraints ({len(uncovered)} uncovered) ---"
                )

            strategies_found = self._find_strategies_of_size(
                states, uncovered, constraint_size
            )

            if strategies_found:
                strategies_found.sort(key=lambda s: -s.coverage)

                for strategy in strategies_found:
                    covered_by_this = set()
                    for idx in uncovered:
                        if strategy.matches(states[idx].state):
                            if strategy.action in states[idx].optimal_actions:
                                covered_by_this.add(idx)

                    if covered_by_this:
                        self.strategies.append(strategy)
                        uncovered -= covered_by_this

                        if self.verbose:
                            print(f"  SELECTED: {strategy}")
                            print(
                                f"    Covers: {len(covered_by_this)} states, {len(uncovered)} remaining"
                            )

                        if not uncovered:
                            break

        if self.verbose:
            print("\n" + "=" * 60)
            print("FINAL STRATEGY LIST (Decision List)")
            print("=" * 60)
            for i, strategy in enumerate(self.strategies):
                print(f"  {i+1}. {strategy} (covers {strategy.coverage})")
            print(f"\nTotal covered: {len(states) - len(uncovered)}/{len(states)}")
            if uncovered:
                print(f"Uncovered: {len(uncovered)}")

        return self.strategies

    def _find_strategies_of_size(
        self, states: List[StateActionPair], uncovered: Set[int], constraint_size: int
    ) -> List[DiscoveredStrategy]:
        strategies: List[DiscoveredStrategy] = []
        pile_combinations = list(combinations(range(self.n_piles), constraint_size))

        for pile_indices in pile_combinations:
            value_ranges = [range(self.initial_piles[i] + 1) for i in pile_indices]
            value_assignments = list(product(*value_ranges))

            for values in value_assignments:
                constraint = dict(zip(pile_indices, values))

                matching_indices = []
                for idx in uncovered:
                    state = states[idx].state
                    if all(state[p] == v for p, v in constraint.items()):
                        matching_indices.append(idx)

                if not matching_indices:
                    continue

                common_actions: Optional[Set[Tuple[int, int]]] = None
                for idx in matching_indices:
                    optimal = set(states[idx].optimal_actions)
                    if common_actions is None:
                        common_actions = optimal
                    else:
                        common_actions &= optimal
                    if not common_actions:
                        break

                if common_actions:
                    action = min(common_actions)
                    valid = all(
                        states[idx].state[action[0]] >= action[1]
                        for idx in matching_indices
                    )

                    if valid:
                        strategies.append(
                            DiscoveredStrategy(
                                constraint=constraint,
                                action=action,
                                coverage=len(matching_indices),
                            )
                        )

        return strategies

    def get_policy_function(self) -> Any:
        def predict(state: Tuple[int, ...]) -> Tuple[int, int]:
            for strategy in self.strategies:
                if strategy.matches(state):
                    pile_idx, count = strategy.action
                    if state[pile_idx] >= count:
                        return strategy.action
            # Fallback: take 1 from first non-empty pile
            for pile_idx in range(len(state)):
                if state[pile_idx] > 0:
                    return (pile_idx, 1)
            return (0, 0)

        return predict


# Stitch compression

try:
    import stitch_core

    STITCH_AVAILABLE = True
except ImportError:
    STITCH_AVAILABLE = False


def strategy_to_sexpr(strategy: DiscoveredStrategy) -> str:
    piles = [f"(pile {idx} {val})" for idx, val in sorted(strategy.constraint.items())]
    pile_idx, count = strategy.action
    return f"(rule {' '.join(piles)} (remove {pile_idx} {count}))"


def strategy_to_canonical_orderings(strategy: DiscoveredStrategy) -> List[str]:
    pile_idx, count = strategy.action
    piles = list(strategy.constraint.items())

    piles_by_index = sorted(piles, key=operator.itemgetter(0, 1))
    ordering1 = [f"(pile {p[0]} {p[1]})" for p in piles_by_index]

    piles_by_value = sorted(piles, key=operator.itemgetter(1, 0))
    ordering2 = [f"(pile {p[0]} {p[1]})" for p in piles_by_value]

    result = [f"(rule {' '.join(ordering1)} (remove {pile_idx} {count}))"]
    if ordering1 != ordering2:
        result.append(f"(rule {' '.join(ordering2)} (remove {pile_idx} {count}))")
    return result


@dataclass
class AbstractStrategyBlock:
    abstraction: str
    strategies: List[DiscoveredStrategy]
    start_idx: int
    end_idx: int


def is_meaningful_abstraction(abstraction_body: str, min_piles: int = 1) -> bool:
    if "remove" not in abstraction_body:
        return False
    return abstraction_body.count("(pile") >= min_piles


def greedy_sliding_window_abstraction(
    strategies: List[DiscoveredStrategy],
    stitch_iterations: int = 20,
    min_window_size: int = 2,
    verbose: bool = True,
) -> List[AbstractStrategyBlock]:
    blocks: List[AbstractStrategyBlock] = []
    n = len(strategies)
    left = 0

    while left < n:
        if verbose:
            print(f"\nScanning from rule {left+1}...")

        best_r = -1
        best_abstraction: Optional[str] = None
        right = left + min_window_size - 1

        while right < n:
            window = strategies[left : right + 1]
            all_orderings: List[str] = []
            strategy_indices = []

            for s in window:
                perms = strategy_to_canonical_orderings(s)
                start_idx = len(all_orderings)
                all_orderings.extend(perms)
                strategy_indices.append(range(start_idx, start_idx + len(perms)))

            valid_abstraction = None

            try:
                res = stitch_core.compress(
                    all_orderings, iterations=stitch_iterations, max_arity=4
                )

                if res.abstractions:
                    derived_from = {ab.name: {ab.name} for ab in res.abstractions}
                    for ab in res.abstractions:
                        for other in res.abstractions:
                            if ab.name in other.body:
                                derived_from[ab.name].add(other.name)

                    for ab in res.abstractions:
                        if not is_meaningful_abstraction(ab.body, min_piles=1):
                            continue

                        family = derived_from[ab.name]
                        window_fully_covered = True

                        for idx_range in strategy_indices:
                            strategy_covered = any(
                                any(name in res.rewritten[i] for name in family)
                                for i in idx_range
                            )
                            if not strategy_covered:
                                window_fully_covered = False
                                break

                        if window_fully_covered:
                            valid_abstraction = ab
                            break

            except Exception as e:
                if verbose:
                    print(f"  Stitch error: {e}")

            if valid_abstraction:
                best_r = right
                best_abstraction = valid_abstraction.body
                right += 1
            else:
                break

        if best_r >= left and best_abstraction is not None:
            block = AbstractStrategyBlock(
                abstraction=best_abstraction,
                strategies=strategies[left : best_r + 1],
                start_idx=left + 1,
                end_idx=best_r + 1,
            )
            blocks.append(block)
            if verbose:
                print(
                    f"  BLOCK: rules {left+1}-{best_r+1}, abstraction: {best_abstraction}"
                )
            left = best_r + 1
        else:
            block = AbstractStrategyBlock(
                abstraction=strategy_to_sexpr(strategies[left]),
                strategies=[strategies[left]],
                start_idx=left + 1,
                end_idx=left + 1,
            )
            blocks.append(block)
            if verbose:
                print(f"  SINGLETON: rule {left+1}")
            left += 1

    return blocks


def evaluate_policy(
    predict_fn: Any,
    initial_piles: Tuple[int, ...] = (3, 4, 5),
    n_games: int = 500,
    verbose: bool = True,
) -> Dict[str, Dict[str, int]]:
    oracle = NimOracle(initial_piles)
    results: Dict[str, Dict[str, int]] = {
        "vs_random": {"win": 0, "loss": 0},
        "vs_optimal": {"win": 0, "loss": 0},
    }

    for _ in range(n_games):
        game = Nim(initial_piles)
        state = game.reset()

        while not game.done:
            player = game.current_player
            if player == 1:
                action = predict_fn(state)
            else:
                valid = game.get_valid_actions()
                action = valid[np.random.randint(len(valid))] if valid else (0, 1)
            state, done, winner = game.step(action)

        if game.winner == 1:
            results["vs_random"]["win"] += 1
        else:
            results["vs_random"]["loss"] += 1

    for _ in range(n_games):
        game = Nim(initial_piles)
        state = game.reset()

        while not game.done:
            player = game.current_player
            if player == 1:
                action = predict_fn(state)
            else:
                action = oracle.get_action(state, player=-1)
            state, done, winner = game.step(action)

        if game.winner == 1:
            results["vs_optimal"]["win"] += 1
        else:
            results["vs_optimal"]["loss"] += 1

    if verbose:
        print("\nEvaluation Results:")
        print(f"  vs Random (n={n_games}):")
        print(f"    Win:  {results['vs_random']['win'] / n_games * 100:.1f}%")
        print(f"    Loss: {results['vs_random']['loss'] / n_games * 100:.1f}%")
        print(f"  vs Optimal (n={n_games}):")
        print(f"    Win:  {results['vs_optimal']['win'] / n_games * 100:.1f}%")
        print(f"    Loss: {results['vs_optimal']['loss'] / n_games * 100:.1f}%")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Iterative Strategy Discovery for Nim")
    parser.add_argument(
        "--piles",
        type=str,
        default="3,4,5",
        help="Initial pile sizes (comma-separated)",
    )
    parser.add_argument(
        "--max-size", type=int, default=3, help="Max constraint size (piles)"
    )
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    parser.add_argument(
        "--no-compress", action="store_true", help="Skip Stitch compression"
    )
    parser.add_argument(
        "--stitch-iterations",
        type=int,
        default=50,
        help="Stitch compression iterations",
    )
    args = parser.parse_args()

    initial_piles = tuple(int(x) for x in args.piles.split(","))

    print("=" * 70)
    print(f"Iterative Strategy Discovery for Nim {initial_piles}")
    print("=" * 70)

    discoverer = NimIterativeStrategyDiscovery(
        initial_piles=initial_piles, max_constraint_size=args.max_size, verbose=True
    )
    discovered_strategies = discoverer.run()

    if not args.no_compress and STITCH_AVAILABLE:
        print("\n" + "=" * 60)
        print("Phase 2: Greedy Sliding Window Abstraction")
        print("=" * 60)
        print("Finding maximal blocks where ONE abstraction covers all strategies...")

        abstract_blocks = greedy_sliding_window_abstraction(
            discovered_strategies,
            stitch_iterations=args.stitch_iterations,
            verbose=True,
        )

        print("\n" + "=" * 60)
        print("ABSTRACT DECISION LIST")
        print("=" * 60)
        print(
            f"# {len(discovered_strategies)} strategies compressed into {len(abstract_blocks)} abstract blocks\n"
        )

        for block_idx, abstract_block in enumerate(abstract_blocks, 1):
            print("=" * 60)
            print(
                f"BLOCK {block_idx}: Rules {abstract_block.start_idx}-{abstract_block.end_idx} ({len(abstract_block.strategies)} strategies)"
            )
            print(f"ABSTRACTION: {abstract_block.abstraction}")
            print("=" * 60)

            for i, strategy in enumerate(abstract_block.strategies):
                rule_num = abstract_block.start_idx + i
                constraint_parts = [
                    f"p{idx}={val}" for idx, val in sorted(strategy.constraint.items())
                ]
                constraint_str = ", ".join(constraint_parts)
                print(
                    f"  {rule_num:3}. ({constraint_str}) -> remove {strategy.action[1]} from pile {strategy.action[0]}"
                )

            print()

        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Total strategies: {len(discovered_strategies)}")
        print(f"  Abstract blocks: {len(abstract_blocks)}")
        if abstract_blocks:
            block_sizes = [len(b.strategies) for b in abstract_blocks]
            print(
                f"  Block sizes: min={min(block_sizes)}, max={max(block_sizes)}, avg={sum(block_sizes)/len(block_sizes):.1f}"
            )

    if not args.no_eval:
        print("\n" + "=" * 60)
        print("Phase 3: Evaluating Discovered Policy")
        print("=" * 60)

        predict_fn = discoverer.get_policy_function()
        eval_results = evaluate_policy(
            predict_fn, initial_piles=initial_piles, n_games=500, verbose=True
        )
