"""
Nim game implementation.
Player who takes the last object wins (normal play convention).
"""

from typing import List, Tuple, Optional


class Nim:
    def __init__(self, piles: Tuple[int, ...] = (3, 4, 5)) -> None:
        self.initial_piles = piles
        self.n_piles = len(piles)
        self.max_pile_sizes = piles
        self.piles = list(piles)
        self.current_player = 1
        self.done = False
        self.winner: Optional[int] = None

    def reset(self) -> Tuple[int, ...]:
        self.piles = list(self.initial_piles)
        self.current_player = 1
        self.done = False
        self.winner = None
        return tuple(self.piles)

    def get_state(self) -> Tuple[int, ...]:
        return tuple(self.piles)

    def get_valid_actions(self) -> List[Tuple[int, int]]:
        actions = []
        for pile_idx in range(self.n_piles):
            for count in range(1, self.piles[pile_idx] + 1):
                actions.append((pile_idx, count))
        return actions

    def action_to_index(self, action: Tuple[int, int]) -> int:
        pile_idx, count = action
        idx = sum(self.max_pile_sizes[i] for i in range(pile_idx))
        idx += count - 1
        return idx

    def index_to_action(self, idx: int) -> Tuple[int, int]:
        cumsum = 0
        for pile_idx in range(self.n_piles):
            if idx < cumsum + self.max_pile_sizes[pile_idx]:
                count = idx - cumsum + 1
                return (pile_idx, count)
            cumsum += self.max_pile_sizes[pile_idx]
        raise ValueError(f"Invalid action index: {idx}")

    def step(
        self, action: Tuple[int, int]
    ) -> Tuple[Tuple[int, ...], bool, Optional[int]]:
        if self.done:
            raise ValueError("Game is over")

        pile_idx, count = action
        if pile_idx < 0 or pile_idx >= self.n_piles:
            raise ValueError(f"Invalid pile index: {pile_idx}")
        if count < 1 or count > self.piles[pile_idx]:
            raise ValueError(
                f"Invalid count {count} for pile {pile_idx} with {self.piles[pile_idx]} objects"
            )

        self.piles[pile_idx] -= count

        if all(p == 0 for p in self.piles):
            self.done = True
            self.winner = self.current_player
        else:
            self.current_player *= -1

        return tuple(self.piles), self.done, self.winner

    def copy(self) -> "Nim":
        new_game = Nim(self.initial_piles)
        new_game.piles = self.piles.copy()
        new_game.current_player = self.current_player
        new_game.done = self.done
        new_game.winner = self.winner
        return new_game

    def render(self) -> str:
        lines = [
            f"Pile {i}: {'|' * count} ({count})" for i, count in enumerate(self.piles)
        ]
        lines.append(f"Player {1 if self.current_player == 1 else 2}'s turn")
        return "\n".join(lines)


def get_current_player_nim(
    state: Tuple[int, ...], initial_piles: Tuple[int, ...] = (3, 4, 5)
) -> int:
    """Determine current player from state by counting moves made."""
    total_initial = sum(initial_piles)
    total_current = sum(state)
    moves_made = total_initial - total_current
    return 1 if moves_made % 2 == 0 else -1


def compute_nim_sum(state: Tuple[int, ...]) -> int:
    """Compute nim-sum (XOR of all pile sizes)."""
    result = 0
    for pile in state:
        result ^= pile
    return result


if __name__ == "__main__":
    game = Nim((3, 4, 5))
    print("Initial state:")
    print(game.render())
    print()

    print("Valid actions:", game.get_valid_actions())
    print()

    print("Nim-sum examples:")
    test_states = [(3, 4, 5), (1, 2, 3), (0, 0, 1), (1, 1, 0), (2, 2, 0)]
    for state in test_states:
        ns = compute_nim_sum(state)
        position = "Losing" if ns == 0 else "Winning"
        print(f"  {state}: nim-sum={ns} ({position})")
