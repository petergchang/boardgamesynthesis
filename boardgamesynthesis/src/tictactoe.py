import numpy as np


class TicTacToe:
    def __init__(self):
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1  # 1 for X, -1 for O
        self.done = False
        self.winner = None

    def reset(self):
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.board.copy()

    def get_valid_actions(self):
        return [i for i in range(9) if self.board[i] == 0]

    def step(self, action):
        if self.done:
            raise ValueError("Game is over")
        if self.board[action] != 0:
            raise ValueError(f"Invalid action {action}")

        self.board[action] = self.current_player

        if self.check_win(self.current_player):
            self.done = True
            self.winner = self.current_player
        elif np.all(self.board != 0):
            self.done = True
            self.winner = 0
        else:
            self.current_player *= -1

        return self.board.copy(), self.done, self.winner

    def check_win(self, player):
        b = self.board.reshape(3, 3)
        if np.any(np.all(b == player, axis=0)) or np.any(np.all(b == player, axis=1)):
            return True
        if np.all(np.diag(b) == player) or np.all(np.diag(np.fliplr(b)) == player):
            return True
        return False

    def get_state(self):
        return self.board.copy()
