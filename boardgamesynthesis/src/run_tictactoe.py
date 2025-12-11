"""
Iterative strategy discovery via constraint-based abstraction for Tic-Tac-Toe.
Discovers decision rules by finding cell constraints that uniquely determine optimal actions.
"""

import operator
import numpy as np
import collections
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass
from itertools import combinations, product
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tictactoe import TicTacToe
from tictactoe_oracle import get_current_player


@dataclass
class StateActionPair:
    board: np.ndarray
    optimal_actions: List[int]

    def action(self) -> int:
        return self.optimal_actions[0]


@dataclass
class DiscoveredStrategy:
    constraint: Dict[int, int]  # cell_index -> value (1=X, -1=O, 0=Empty)
    action: int
    coverage: int

    def matches(self, board: np.ndarray) -> bool:
        for cell_idx, value in self.constraint.items():
            if board[cell_idx] != value:
                return False
        return True

    def __str__(self) -> str:
        cell_strs = []
        for idx, val in sorted(self.constraint.items()):
            symbol = {1: "X", -1: "O", 0: "E"}[val]
            cell_strs.append(f"c{idx}={symbol}")
        return f"({', '.join(cell_strs)}) -> play {self.action}"


class GameGraphMiner:
    """Mine all P1 states where we can win or draw."""

    def __init__(self) -> None:
        self.memo_dist: Dict[Tuple[int, ...], Tuple[int, int]] = {}

    def get_all_states(self, verbose: bool = True) -> List[StateActionPair]:
        if verbose:
            print("Phase 1: Mining All P1 States (Win/Draw)")

        initial_board = np.zeros(9, dtype=int)
        queue = collections.deque([tuple(initial_board)])
        visited = {tuple(initial_board)}
        collected_states: List[Tuple[int, ...]] = []

        while queue:
            board_tuple = queue.popleft()
            board = np.array(board_tuple)
            player = get_current_player(board)

            env = TicTacToe()
            env.board = board
            if env.check_win(1) or env.check_win(-1) or np.all(board != 0):
                continue

            if player == 1:
                collected_states.append(board_tuple)

            valid_actions = [i for i in range(9) if board[i] == 0]
            for action in valid_actions:
                env_next = TicTacToe()
                env_next.board = board.copy()
                env_next.current_player = player
                env_next.step(action)
                next_tuple = tuple(env_next.board)
                if next_tuple not in visited:
                    visited.add(next_tuple)
                    queue.append(next_tuple)

        result = []
        for state_tuple in collected_states:
            board = np.array(state_tuple)
            val, _ = self._get_outcome(board)
            if val >= 0:
                optimal = self._get_optimal_actions(board)
                if optimal:
                    result.append(StateActionPair(board, optimal))

        if verbose:
            print(f"Found {len(result)} valid P1 states.")

        return result

    def _get_outcome(self, board: np.ndarray) -> Tuple[int, int]:
        """Minimax: returns (value, depth). Value: 1=win, 0=draw, -1=loss."""
        board_tuple = tuple(board)
        if board_tuple in self.memo_dist:
            return self.memo_dist[board_tuple]

        env = TicTacToe()
        env.board = board

        if env.check_win(1):
            return (1, 0)
        if env.check_win(-1):
            return (-1, 0)
        if np.all(board != 0):
            return (0, 0)

        player = get_current_player(board)
        outcomes = []
        for a in [i for i in range(9) if board[i] == 0]:
            env_next = TicTacToe()
            env_next.board = board.copy()
            env_next.current_player = player
            env_next.step(a)
            v, d = self._get_outcome(env_next.board)
            outcomes.append((v, d + 1))

        if player == 1:
            best = max(outcomes, key=lambda x: (x[0], -x[1]))
        else:
            best = min(outcomes, key=lambda x: (x[0], -x[1]))

        self.memo_dist[board_tuple] = best
        return best

    def _get_optimal_actions(self, board: np.ndarray) -> List[int]:
        target_val, target_depth = self._get_outcome(board)
        actions = []
        for a in [i for i in range(9) if board[i] == 0]:
            env = TicTacToe()
            env.board = board.copy()
            env.current_player = 1
            env.step(a)
            v, d = self._get_outcome(env.board)
            if v == target_val and d + 1 == target_depth:
                actions.append(a)
        return actions


class IterativeStrategyDiscovery:
    """Discover strategies by finding cell constraints that uniquely determine optimal actions."""

    def __init__(self, max_constraint_size: int = 7, verbose: bool = True) -> None:
        self.max_constraint_size = max_constraint_size
        self.verbose = verbose
        self.miner = GameGraphMiner()
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
                    f"\n--- Trying {constraint_size}-cell constraints ({len(uncovered)} uncovered) ---"
                )

            strategies_found = self._find_strategies_of_size(
                states, uncovered, constraint_size
            )

            if strategies_found:
                strategies_found.sort(key=lambda s: -s.coverage)

                for strategy in strategies_found:
                    covered_by_this = set()
                    for idx in uncovered:
                        if strategy.matches(states[idx].board):
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
        cell_combinations = list(combinations(range(9), constraint_size))
        value_assignments = list(product((1, -1, 0), repeat=constraint_size))

        if self.verbose and constraint_size <= 3:
            print(
                f"  Checking {len(cell_combinations) * len(value_assignments)} constraint combinations..."
            )

        for cell_indices in cell_combinations:
            for values in value_assignments:
                constraint = dict(zip(cell_indices, values))

                matching_indices = []
                for idx in uncovered:
                    board = states[idx].board
                    if all(board[c] == v for c, v in constraint.items()):
                        matching_indices.append(idx)

                if not matching_indices:
                    continue

                common_actions: Optional[Set[int]] = None
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
                        states[idx].board[action] == 0 for idx in matching_indices
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
        def predict(board: np.ndarray) -> int:
            for strategy in self.strategies:
                if strategy.matches(board):
                    if board[strategy.action] == 0:
                        return strategy.action
            # Fallback: center > corners > edges
            if board[4] == 0:
                return 4
            for idx in (0, 2, 6, 8, 1, 3, 5, 7):
                if board[idx] == 0:
                    return idx
            return 0

        return predict


def evaluate_policy(
    predict_fn: Any, n_games: int = 500, verbose: bool = True
) -> Dict[str, Dict[str, int]]:
    from tictactoe_oracle import Oracle

    oracle = Oracle()
    results: Dict[str, Dict[str, int]] = {
        "vs_random": {"win": 0, "loss": 0, "draw": 0},
        "vs_optimal": {"win": 0, "loss": 0, "draw": 0},
    }

    env = TicTacToe()

    for _ in range(n_games):
        state = env.reset()
        done = False
        while not done:
            player = get_current_player(state)
            if player == 1:
                action = predict_fn(state)
            else:
                valid = [i for i in range(9) if state[i] == 0]
                action = np.random.choice(valid) if valid else 0
            state, done, _ = env.step(action)

        if env.check_win(1):
            results["vs_random"]["win"] += 1
        elif env.check_win(-1):
            results["vs_random"]["loss"] += 1
        else:
            results["vs_random"]["draw"] += 1

    for _ in range(n_games):
        state = env.reset()
        done = False
        while not done:
            player = get_current_player(state)
            if player == 1:
                action = predict_fn(state)
            else:
                action = oracle.get_action(state)
            state, done, _ = env.step(action)

        if env.check_win(1):
            results["vs_optimal"]["win"] += 1
        elif env.check_win(-1):
            results["vs_optimal"]["loss"] += 1
        else:
            results["vs_optimal"]["draw"] += 1

    if verbose:
        print("\nEvaluation Results:")
        print(f"  vs Random (n={n_games}):")
        print(f"    Win:  {results['vs_random']['win'] / n_games * 100:.1f}%")
        print(f"    Loss: {results['vs_random']['loss'] / n_games * 100:.1f}%")
        print(f"    Draw: {results['vs_random']['draw'] / n_games * 100:.1f}%")
        print(f"  vs Optimal (n={n_games}):")
        print(f"    Win:  {results['vs_optimal']['win'] / n_games * 100:.1f}%")
        print(f"    Loss: {results['vs_optimal']['loss'] / n_games * 100:.1f}%")
        print(f"    Draw: {results['vs_optimal']['draw'] / n_games * 100:.1f}%")

    return results


# Stitch compression

try:
    import stitch_core

    STITCH_AVAILABLE = True
except ImportError:
    STITCH_AVAILABLE = False


def strategy_to_sexpr(strategy: DiscoveredStrategy, use_structured: bool = True) -> str:
    action = strategy.action

    if use_structured:
        cells_curr = []
        for idx, val in sorted(strategy.constraint.items()):
            symbol = {1: "X", -1: "O", 0: "E"}[val]
            cells_curr.append(f"(cell {idx} {symbol})")
        return f"(rule {' '.join(cells_curr)} (play {action}))"

    ar, ac = action // 3, action % 3
    cells: List[Tuple[str, int, int]] = []
    for idx, val in strategy.constraint.items():
        symbol = {1: "X", -1: "O", 0: "E"}[val]
        r, c = idx // 3, idx % 3
        dr, dc = r - ar, c - ac
        cells.append((symbol, dr, dc))
    cells.sort(key=operator.itemgetter(1, 2))
    cell_strs = [f"({sym} {dr} {dc})" for sym, dr, dc in cells]
    return f"(pattern {' '.join(cell_strs)})"


def strategy_to_all_permutations(strategy: DiscoveredStrategy) -> List[str]:
    from itertools import permutations

    action = strategy.action
    cells = []
    for idx, val in strategy.constraint.items():
        symbol = {1: "X", -1: "O", 0: "E"}[val]
        cells.append(f"(cell {idx} {symbol})")

    return [f"(rule {' '.join(perm)} (play {action}))" for perm in permutations(cells)]


def strategy_to_canonical_orderings(strategy: DiscoveredStrategy) -> List[str]:
    action = strategy.action
    cells: List[Tuple[int, int, str, int]] = []
    val_order = {1: 0, 0: 1, -1: 2}
    for idx, val in strategy.constraint.items():
        symbol = {1: "X", -1: "O", 0: "E"}[val]
        cells.append((idx, val, symbol, val_order[val]))

    cells_by_value = sorted(cells, key=operator.itemgetter(3, 0))
    ordering1 = [f"(cell {c[0]} {c[2]})" for c in cells_by_value]

    cells_by_index = sorted(cells, key=operator.itemgetter(0, 3))
    ordering2 = [f"(cell {c[0]} {c[2]})" for c in cells_by_index]

    result = [f"(rule {' '.join(ordering1)} (play {action}))"]
    if ordering1 != ordering2:
        result.append(f"(rule {' '.join(ordering2)} (play {action}))")
    return result


@dataclass
class AbstractStrategyBlock:
    abstraction: str
    strategies: List[DiscoveredStrategy]
    start_idx: int
    end_idx: int


def is_meaningful_abstraction(abstraction_body: str, min_cells: int = 2) -> bool:
    if "play" not in abstraction_body:
        return False
    cell_count = abstraction_body.count("(cell")
    return cell_count >= min_cells


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
                        if not is_meaningful_abstraction(ab.body, min_cells=2):
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


@dataclass
class CompressionResult:
    abstractions: List[Any]
    rewritten: List[str]
    original: List[str]
    strategies: List[DiscoveredStrategy]


@dataclass
class GlobalCompressionResult:
    abstractions: List[Any]
    rewritten: List[str]
    original: List[str]


def compress_strategies_global(
    strategies: List[DiscoveredStrategy],
    iterations: int = 500,
    max_arity: int = 5,
    verbose: bool = True,
) -> Optional[GlobalCompressionResult]:
    if not STITCH_AVAILABLE:
        print("Stitch not available - skipping compression")
        return None

    programs = [strategy_to_sexpr(s) for s in strategies]

    if verbose:
        print(f"\nCompressing {len(programs)} strategies globally...")
        print("  Sample programs:")
        for i in (0, 1, 24, 25, -1):
            if abs(i) < len(programs):
                print(f"    [{i}]: {programs[i]}")

    try:
        result = stitch_core.compress(
            programs, iterations=iterations, max_arity=max_arity
        )

        if result.abstractions:
            if verbose:
                print(f"\n  Found {len(result.abstractions)} global abstractions:")
                for ab in result.abstractions:
                    print(f"    {ab.name}: {ab.body}")

                orig_size = sum(len(p) for p in programs)
                rewritten_size = sum(len(p) for p in result.rewritten)
                abs_size = sum(len(ab.body) for ab in result.abstractions)
                print(
                    f"\n  Compression: {orig_size} -> {rewritten_size} + {abs_size} = {rewritten_size + abs_size}"
                )
                print(f"  Ratio: {orig_size / (rewritten_size + abs_size):.2f}x")

            return GlobalCompressionResult(
                abstractions=result.abstractions,
                rewritten=result.rewritten,
                original=programs,
            )
        if verbose:
            print("  No abstractions found")
        return None

    except Exception as e:
        if verbose:
            print(f"  Compression failed: {e}")
        return None


def get_top_level_abstraction(rewritten: str) -> Optional[str]:
    rewritten = rewritten.strip()
    if rewritten.startswith("("):
        inner = rewritten[1:].strip()
        first_space = inner.find(" ")
        first_paren = inner.find(")")
        if first_space == -1:
            first_space = len(inner)
        if first_paren == -1:
            first_paren = len(inner)
        end = min(first_space, first_paren)
        token = inner[:end]
        if token.startswith("fn_"):
            return token
    elif rewritten.startswith("fn_"):
        return rewritten.split()[0] if " " in rewritten else rewritten
    return None


@dataclass
class StrategyBlock:
    abstraction: Optional[str]
    strategies: List[DiscoveredStrategy]
    rewritten: List[str]
    start_idx: int


def cluster_by_abstraction(
    strategies: List[DiscoveredStrategy], rewritten: List[str]
) -> List[StrategyBlock]:
    blocks: List[StrategyBlock] = []
    current_block: Optional[StrategyBlock] = None

    for i, (strategy, rewr) in enumerate(zip(strategies, rewritten)):
        top_abs = get_top_level_abstraction(rewr)

        if current_block is None:
            current_block = StrategyBlock(
                abstraction=top_abs,
                strategies=[strategy],
                rewritten=[rewr],
                start_idx=i + 1,
            )
        elif current_block.abstraction == top_abs:
            current_block.strategies.append(strategy)
            current_block.rewritten.append(rewr)
        else:
            blocks.append(current_block)
            current_block = StrategyBlock(
                abstraction=top_abs,
                strategies=[strategy],
                rewritten=[rewr],
                start_idx=i + 1,
            )

    if current_block is not None:
        blocks.append(current_block)

    return blocks


def interpret_abstraction(body: str) -> str:
    import re

    x_count = body.count(" X)")
    o_count = body.count(" O)")
    e_count = body.count(" E)")

    lines = [
        {0, 1, 2},
        {3, 4, 5},
        {6, 7, 8},
        {0, 3, 6},
        {1, 4, 7},
        {2, 5, 8},
        {0, 4, 8},
        {2, 4, 6},
    ]

    cells = {int(m.group(1)) for m in re.finditer(r"\(c(\d+)", body)}
    is_line = any(cells >= line for line in lines)

    if x_count == 2 and e_count == 1 and is_line:
        return "WIN: Complete own line (2 X's + empty)"
    if o_count == 2 and e_count == 1 and is_line:
        return "BLOCK: Block opponent's line (2 O's + empty)"
    if x_count == 2 and e_count >= 1:
        return "DEVELOP: Extend own presence"
    if o_count == 2 and e_count >= 1:
        return "DEFEND: Counter opponent's threat"
    if x_count == 1 and e_count >= 2:
        return "OPEN: Early game development"
    if e_count >= 3:
        return "OPENING: Empty board strategy"
    return f"PATTERN: {x_count}X, {o_count}O, {e_count}E"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Iterative Strategy Discovery")
    parser.add_argument(
        "--max-size", type=int, default=5, help="Max constraint size (cells)"
    )
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    parser.add_argument(
        "--no-compress", action="store_true", help="Skip Stitch compression"
    )
    parser.add_argument(
        "--stitch-iterations",
        type=int,
        default=500,
        help="Stitch compression iterations",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Iterative Strategy Discovery via Constraint-Based Abstraction")
    print("=" * 70)

    discoverer = IterativeStrategyDiscovery(
        max_constraint_size=args.max_size, verbose=True
    )
    discovered_strategies = discoverer.run()

    if not args.no_compress and STITCH_AVAILABLE:
        print("\n" + "=" * 60)
        print("Phase 2: Greedy Sliding Window Abstraction")
        print("=" * 60)
        print("Finding maximal blocks where ONE abstraction covers all strategies...")
        print("(Using 2 canonical orderings: sort-by-value vs sort-by-index)")

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

        CHAR_MAP = {1: "X", -1: "O", 0: "E"}

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
                    f"c{idx}={CHAR_MAP[val]}"
                    for idx, val in sorted(strategy.constraint.items())
                ]
                constraint_str = ", ".join(constraint_parts)
                print(f"  {rule_num:3}. ({constraint_str}) -> play {strategy.action}")

            print()

        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Total strategies: {len(discovered_strategies)}")
        print(f"  Abstract blocks: {len(abstract_blocks)}")
        block_sizes = [len(b.strategies) for b in abstract_blocks]
        print(
            f"  Block sizes: min={min(block_sizes)}, max={max(block_sizes)}, avg={sum(block_sizes)/len(block_sizes):.1f}"
        )

    if not args.no_eval:
        print("\n" + "=" * 60)
        print("Phase 3: Evaluating Discovered Policy")
        print("=" * 60)

        predict_fn = discoverer.get_policy_function()
        eval_results = evaluate_policy(predict_fn, n_games=500, verbose=True)
