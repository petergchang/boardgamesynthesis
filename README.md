# Boardgame Synthesis

This project implements policy extraction for board games.

## Installation

```bash
uv pip install -e '.[dev]'
```

## Experiments

To synthesize a policy for Tic-Tac-Toe and save the result as a `txt` file, run:

```bash
python boardgamesynthesis/src/run_tictactoe.py >> tictactoe_results.txt
```

To synthesize a policy for Nim and save the result as a `txt` file, run:

```bash
python boardgamesynthesis/src/run_nim.py >> tictactoe_results.txt
```
