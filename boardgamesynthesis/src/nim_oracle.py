"""Nim oracle using minimax with memoization."""

import numpy as np
from typing import Any, Dict, List, Tuple, Union, cast
from nim import compute_nim_sum


class NimOracle:
    def __init__(self, initial_piles: Tuple[int, ...] = (3, 4, 5)) -> None:
        self.initial_piles = initial_piles
        self.n_piles = len(initial_piles)
        self.cache: Dict[Tuple[Tuple[int, ...], int], Tuple[int, int]] = {}
        self.minimax_cache: Dict[Any, Union[float, Tuple[float, int]]] = {}

    def get_action(self, state: Tuple[int, ...], player: int = 1) -> Tuple[int, int]:
        cache_key = (state, player)
        if cache_key in self.cache:
            return self.cache[cache_key]

        valid_actions = self._get_valid_actions(state)
        best_action = valid_actions[0]
        best_score = -np.inf if player == 1 else np.inf

        for action in valid_actions:
            new_state = self._apply_action(state, action)
            score = self._minimax(new_state, -player)

            if player == 1:
                if score > best_score:
                    best_score = score
                    best_action = action
            else:
                if score < best_score:
                    best_score = score
                    best_action = action

        self.cache[cache_key] = best_action
        return best_action

    def get_q_values(
        self, state: Tuple[int, ...], player: int = 1
    ) -> Dict[Tuple[int, int], float]:
        valid_actions = self._get_valid_actions(state)
        q_values = {}

        for action in valid_actions:
            new_state = self._apply_action(state, action)
            score = self._minimax(new_state, -player)
            q_values[action] = score

        return q_values

    def get_optimal_actions(
        self, state: Tuple[int, ...], player: int = 1
    ) -> List[Tuple[int, int]]:
        q_values = self.get_q_values(state, player)
        if not q_values:
            return []

        best_value = max(q_values.values()) if player == 1 else min(q_values.values())
        return [a for a, v in q_values.items() if v == best_value]

    def _get_valid_actions(self, state: Tuple[int, ...]) -> List[Tuple[int, int]]:
        actions = []
        for pile_idx in range(self.n_piles):
            for count in range(1, state[pile_idx] + 1):
                actions.append((pile_idx, count))
        return actions

    def _apply_action(
        self, state: Tuple[int, ...], action: Tuple[int, int]
    ) -> Tuple[int, ...]:
        pile_idx, count = action
        new_state = list(state)
        new_state[pile_idx] -= count
        return tuple(new_state)

    def _minimax(self, state: Tuple[int, ...], player: int) -> float:
        cache_key = (state, player)
        if cache_key in self.minimax_cache:
            return cast(float, self.minimax_cache[cache_key])

        # Terminal: previous player took last object and won
        if all(p == 0 for p in state):
            result = -player
            self.minimax_cache[cache_key] = result
            return result

        valid_actions = self._get_valid_actions(state)

        if player == 1:
            best_score = -np.inf
            for action in valid_actions:
                new_state = self._apply_action(state, action)
                score = self._minimax(new_state, -player)
                best_score = max(best_score, score)
            self.minimax_cache[cache_key] = best_score
            return best_score
        else:
            best_score = np.inf
            for action in valid_actions:
                new_state = self._apply_action(state, action)
                score = self._minimax(new_state, -player)
                best_score = min(best_score, score)
            self.minimax_cache[cache_key] = best_score
            return best_score

    def get_outcome(self, state: Tuple[int, ...], player: int = 1) -> Tuple[float, int]:
        """Returns (value, depth) with optimal play."""
        return self._get_outcome_helper(state, player)

    def _get_outcome_helper(
        self, state: Tuple[int, ...], player: int
    ) -> Tuple[float, int]:
        depth_key = ("depth", state, player)
        if depth_key in self.minimax_cache:
            return cast(Tuple[float, int], self.minimax_cache[depth_key])

        if all(p == 0 for p in state):
            result = (-player, 0)
            self.minimax_cache[depth_key] = result
            return result

        valid_actions = self._get_valid_actions(state)
        outcomes = []
        for action in valid_actions:
            new_state = self._apply_action(state, action)
            v, d = self._get_outcome_helper(new_state, -player)
            outcomes.append((v, d + 1))

        if player == 1:
            best = max(outcomes, key=lambda x: (x[0], -x[1]))
        else:
            best = min(outcomes, key=lambda x: (x[0], -x[1]))

        self.minimax_cache[depth_key] = best
        return best


if __name__ == "__main__":
    print("Testing NimOracle on (3, 4, 5)...")
    oracle = NimOracle((3, 4, 5))
    state = (3, 4, 5)

    print(f"\nInitial state: {state}")
    print(f"Nim-sum: {compute_nim_sum(state)}")

    action = oracle.get_action(state, player=1)
    print(f"Optimal action: remove {action[1]} from pile {action[0]}")

    print("\nVerification against Nim theory:")
    test_states = [(3, 4, 5), (1, 2, 3), (0, 0, 1), (1, 1, 0), (2, 2, 0)]
    for s in test_states:
        ns = compute_nim_sum(s)
        value, depth = oracle.get_outcome(s, player=1)
        theory_value = 1 if ns != 0 else -1
        match = "OK" if value == theory_value else "WRONG"
        print(f"  {s}: nim-sum={ns}, oracle={value}, theory={theory_value} [{match}]")
