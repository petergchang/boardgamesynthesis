import numpy as np
from tictactoe import TicTacToe


def get_current_player(board):
    """Determine current player from board state. X (1) goes first."""
    count_x = np.sum(board == 1)
    count_o = np.sum(board == -1)
    return 1 if count_x == count_o else -1


class Oracle:
    def __init__(self):
        self.cache = {}
        self.minimax_cache = {}

    def get_action(self, board):
        board_tuple = tuple(board)
        if board_tuple in self.cache:
            return self.cache[board_tuple]

        player = get_current_player(board)
        best_score = -np.inf if player == 1 else np.inf
        best_action = -1

        valid_actions = [i for i in range(9) if board[i] == 0]

        for action in valid_actions:
            env = TicTacToe()
            env.board = np.array(board)
            env.current_player = player
            env.done = False
            env.step(action)

            score = self.minimax(env.board, -player)

            if player == 1:
                if score > best_score:
                    best_score = score
                    best_action = action
            else:
                if score < best_score:
                    best_score = score
                    best_action = action

        self.cache[board_tuple] = best_action
        return best_action

    def get_q_values(self, board):
        board_tuple = tuple(board)
        cache_key = (board_tuple, "qvalues")
        if cache_key in self.minimax_cache:
            return self.minimax_cache[cache_key]

        player = get_current_player(board)
        valid_actions = [i for i in range(9) if board[i] == 0]
        q_values = {}

        for action in valid_actions:
            new_board = np.array(board)
            new_board[action] = player
            score = self.minimax(new_board, -player)
            q_values[action] = score

        self.minimax_cache[cache_key] = q_values
        return q_values

    def minimax(self, board, player):
        board_tuple = tuple(board)
        cache_key = (board_tuple, player)
        if cache_key in self.minimax_cache:
            return self.minimax_cache[cache_key]

        env = TicTacToe()
        env.board = np.array(board)

        if env.check_win(1):
            self.minimax_cache[cache_key] = 1
            return 1
        if env.check_win(-1):
            self.minimax_cache[cache_key] = -1
            return -1
        if np.all(board != 0):
            self.minimax_cache[cache_key] = 0
            return 0

        best_score = -np.inf if player == 1 else np.inf
        valid_actions = [i for i in range(9) if board[i] == 0]

        for action in valid_actions:
            new_board = np.array(board)
            new_board[action] = player
            score = self.minimax(new_board, -player)

            if player == 1:
                best_score = max(best_score, score)
            else:
                best_score = min(best_score, score)

        self.minimax_cache[cache_key] = best_score
        return best_score
