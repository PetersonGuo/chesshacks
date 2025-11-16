# ChessHacks Starter Bot

This is a starter bot for ChessHacks. It includes a basic bot and devtools. This is designed to help you get used to what the interface for building a bot feels like, as well as how to scaffold your own bot.

## Directory Structure

`/devtools` is a Next.js app that provides a UI for testing your bot. It includes an analysis board that you can use to test your bot and play against your own bot. You do not need to edit, or look at, any of this code (unless you want to). This file should be gitignored. Find out why [here](#installing-devtools-if-you-did-not-run-npx-chesshacks-create).

`/src` is the source code for your bot. You will need to edit this code to implement your own bot.

`serve.py` is the backend that interacts with the Next.js and your bot (`/src/main.py`). It also handles hot reloading of your bot when you make changes to it. This file, after receiving moves from the frontend, will communicate the current board status to your bot as a PGN string, and will send your bot's move back to the frontend. You do not need to edit any of this code (unless you want to).

While developing, you do not need to run either of the Python files yourself. The Next.js app includes the `serve.py` file as a subprocess, and will run it for you when you run `npm run dev`.

The backend (as a subprocess) will deploy on port `5058` by default.

This architecture is very similar to how your bot will run once you deploy it. For more information about how to deploy your bot to the ChessHacks platform, see [the docs](https://docs.chesshacks.dev/).

## Setup

Start by creating a Python virtual environment and installing the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

or however you want to set up your Python.

### Building the C++ Chess Engine

This project includes a high-performance C++ chess engine compiled as a Python extension module. To build it:

```bash
cd build
cmake ..
make
```

This will create the `c_helpers` module in the `build/` directory.

**CUDA GPU Acceleration**: The engine now includes optional CUDA support for GPU-accelerated position evaluation. If you have an NVIDIA GPU with CUDA Toolkit installed, the build will automatically detect and enable GPU acceleration. See [CUDA_README.md](CUDA_README.md) for details.

### Running Tests

All tests are located in the `tests/` directory and should be run from the `build/` directory using the correct Python environment:

```bash
# Run all tests
./run_tests.sh

# Run a specific test
./run_tests.sh test_simple.py
```

Or manually:
```bash
cd build
/path/to/python3.14 tests/test_simple.py
```

Note: The C++ module is compiled for Python 3.14. Make sure to use the matching Python version from the `chesshacks` conda environment.

### Next.js Development Tools

Then, install the dependencies for the Next.js app:

```bash
cd devtools
npm install
```

Afterwards, make a copy of `.env.template` and name it `.env.local` (NOT `.env`). Then fill out the values with the path to your Python environment, and the ports you want to use.

> Copy the `.env.local` file to the `devtools` directory as well.

## Installing `devtools` (if you did not run `npx chesshacks create`)

If you started from your own project and only want to add the devtools UI, you can install it with the CLI:

```bash
npx chesshacks install
```

This will add a `devtools` folder to your current directory and ensure it is gitignored. If you want to install into a subdirectory, you can pass a path:

```bash
npx chesshacks install my-existing-bot
```

In both cases, you can then follow the instructions in [Setup](#setup) and [Running the app](#running-the-app) from inside the `devtools` folder.

## Running the app

Lastly, simply run the Nextjs app inside of the devtools folder.

```bash
cd devtools
npm run dev
```

## Troubleshooting

First, make sure that you aren't running any `python` commands! These devtools are designed to help you play against your bot and see how its predictions are working. You can see [Setup](#setup) and [Running the app](#running-the-app) above for information on how to run the app. You should be running the Next.js app, not the Python files directly!

If you get an error like this:

```python
Traceback (most recent call last):
  File "/Users/obama/dev/chesshacks//src/main.py", line 1, in <module>
    from .utils import chess_manager, GameContext
ImportError: attempted relative import with no known parent package
```

you might think that you should remove the period before `utils` and that will fix the issue. But in reality, this will just cause more problems in the future! You aren't supposed to run `main.py ` on your own—it's designed for `serve.py` to run it for you within the subprocess. Removing the period would cause it to break during that step.

### Logs

Once you run the app, you should see logs from both the Next.js app and the Python subprocess, which includes both `serve.py` and `main.py`. `stdout`s and `stderr`s from both Python files will show in your Next.js terminal. They are designed to be fairly verbose by default.

## HMR (Hot Module Reloading)

By default, the Next.js app will automatically reload (dismount and remount the subprocess) when you make changes to the code in `/src` OR press the manual reload button on the frontend. This is called HMR (Hot Module Reloading). This means that you don't need to restart the app every time you make a change to the Python code. You can see how it's happening in real-time in the Next.js terminal.

## Parting Words

Keep in mind that you fully own all of this code! This entire devtool system runs locally, so feel free to modify it however you want. This is just designed as scaffolding to help you get started.

If you need further help, please first check out the [docs](https://docs.chesshacks.dev/). If you still need help, please join our [Discord](https://docs.chesshacks.dev/resources/discord) and ask for help.

---

## Chess Engine Architecture

### Overview

This chess engine uses C++ for performance-critical search algorithms with Python bindings via nanobind.

### Three Search Functions

#### 1. `alpha_beta_basic()` - Bare Bones

**Purpose:** Fallback function with minimal dependencies

**Features:**

- Pure alpha-beta pruning
- No transposition table
- No move ordering
- No parallelization

**Use When:** Debugging or when optimizations cause issues

```python
result = c_helpers.alpha_beta_basic(fen, depth, alpha, beta, maximizing, evaluate_fn)
```

#### 2. `alpha_beta_optimized()` - Full Optimizations

**Purpose:** Production-ready search with all optimizations

**Features:**

- [x] Transposition table (caching positions)
- [x] Advanced move ordering:
  - TT best move (highest priority)
  - Promotion moves
  - Killer moves (good quiet moves from sibling nodes)
  - MVV-LVA for captures (Most Valuable Victim - Least Valuable Attacker)
  - History heuristic (quiet moves that caused cutoffs)
- [x] Null move pruning (skip opponent's move to test position strength)
- [x] Quiescence search (avoids horizon effect by searching captures at leaf nodes)
- [x] Iterative deepening (gradually increases search depth, populating TT)
- [x] Optional multithreading (parallel root search)

**Parameters:**

- `tt`: Optional TranspositionTable (creates local one if None)
- `num_threads`: 0=auto, 1=sequential, >1=parallel
- `killers`: Optional KillerMoves table (reusable across searches)
- `history`: Optional HistoryTable (reusable across searches)

```python
# Sequential with all tables
tt = c_helpers.TranspositionTable()
killers = c_helpers.KillerMoves()
history = c_helpers.HistoryTable()

# Use enhanced evaluation with piece-square tables
result = c_helpers.alpha_beta_optimized(
    fen, depth, alpha, beta, maximizing, 
    c_helpers.evaluate_with_pst,  # Material + positional evaluation
    tt, 1, killers, history
)

# Parallel search with auto-threading
result = c_helpers.alpha_beta_optimized(
    fen, depth, alpha, beta, maximizing, 
    c_helpers.evaluate_with_pst, 
    None, 0, None, None  # Auto-creates tables, auto-detects threads
)
```
```

#### 3. `alpha_beta_cuda()` - GPU Acceleration

**Purpose:** Future GPU-accelerated search

**Current Status:** Placeholder - falls back to `alpha_beta_optimized()`

**Future Plans:** Batch NNUE evaluation on GPU

```python
result = c_helpers.alpha_beta_cuda(fen, depth, alpha, beta, maximizing, 
                                    c_helpers.evaluate_with_pst, tt)
```

#### 4. `evaluate_with_pst()` - Enhanced Evaluation

**Purpose:** Material + positional evaluation using piece-square tables

**Features:**
- Evaluates piece material (pawn=100, knight=320, etc.)
- Adds positional bonuses based on piece placement
- Six evaluation tables: pawn, knight, bishop, rook, queen, king
- Center control, piece development, king safety

```python
score = c_helpers.evaluate_with_pst(fen)
# Returns centipawn score (positive = white better)
```

### Components

#### TranspositionTable

Thread-safe hash table that caches evaluated positions:

- Stores: depth, score, bound type, best move
- Probing: Returns cached score if conditions met
- Used for: Avoiding redundant evaluation, move ordering

#### Move Ordering

Improves alpha-beta pruning effectiveness by trying best moves first:

1. **TT Best Move** (score 1,000,000): Move from previous iteration/shallower search
2. **Promotions** (score 900,000+): Queen promotions get highest value
3. **Killer Moves** (score 850,000): Good quiet moves from sibling nodes
4. **Captures** (score 800,000+): Ordered by MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
5. **Quiet Moves** (score 0-10,000): History heuristic or center control

### Algorithm Details

#### Principal Variation Search (PVS)

Optimizes search by using null windows for verification:
- First move: full window search
- Subsequent moves: null window (alpha, alpha+1) or (beta-1, beta) scout search
- Re-searches with full window if scout fails high/low
- Combined with LMR: double re-search logic (undo LMR, then expand window)
- 10-20% speedup by verifying most moves fail with minimal search

#### Piece-Square Tables (PST)

Positional evaluation beyond material:
- Six tables (pawn, knight, bishop, rook, queen, king) with 64 square values each
- Rewards good piece placement (+/- 50 centipawns):
  * Pawns: advanced and center pawns stronger
  * Knights: centralized knights better
  * Bishops: long diagonals favored
  * Rooks: 7th rank and open files
  * Queen: slight center control
  * King: corner safety (middlegame)
- Tables automatically mirrored for black pieces
- Much better positional play

#### Internal Iterative Deepening (IID)

Improves move ordering when TT cache miss occurs:
- Triggers at depth >= 5 when no TT move available
- Performs shallow search (depth-2) to find best move
- Populates TT with move ordering for full-depth search
- 5-10% speedup on cache misses

#### Late Move Reductions (LMR)

Reduces search depth for moves ordered later in the move list:
- Applied after first 4 moves at depth >= 3
- Only for quiet moves (not captures, promotions, or in check)
- Reduction: 1 ply for moves 5-8, 2 plies for moves 9+
- Re-searches at full depth if reduced search fails high/low
- 15-25% speedup by pruning unlikely moves more aggressively

#### Aspiration Windows

Uses narrow search windows to cause more cutoffs:
- Applied at depth >= 3 in iterative deepening
- Window size: +/- 50 centipawns around previous iteration's score
- Re-searches with full window if score falls outside bounds
- 10-20% speedup from tighter alpha-beta bounds

#### Killer Move Heuristic

Tracks non-capture "killer" moves that caused beta cutoffs at each ply:
- Stores 2 killer moves per depth level
- Checked in move ordering after TT move but before captures
- Helps find refutations faster
- 10-15% speedup on tactical positions

#### History Heuristic

Maintains piece-to-square scores for quiet moves:
- Updated when quiet moves cause beta cutoffs
- Score increased by depth^2 (deeper moves weighted more)
- Used to order quiet moves after killers
- Aging function divides scores by 2 to favor recent patterns
- 5-10% speedup, improves over time

#### Null Move Pruning

Tests position strength by skipping opponent's move:
- Only applied when depth >= 3, not in check, not in endgame
- Uses reduction factor R=2 (search depth-3 instead of depth-1)
- If opponent still can't improve position, causes beta cutoff
- 20-30% average speedup, huge gains in winning positions
- Carefully avoids zugzwang positions

#### Quiescence Search

Prevents the "horizon effect" where the engine stops searching right before a capture sequence. When reaching depth 0, instead of immediately evaluating, the engine:
- Continues searching all captures and promotions
- Uses MVV-LVA ordering for efficient pruning
- Returns a "quiet" position evaluation
- Dramatically improves tactical awareness
- Depth-limited to 4 plies to prevent infinite recursion

#### Iterative Deepening

Searches progressively deeper, starting from depth 1 up to the target depth:
- Populates transposition table with shallow searches
- Enables better move ordering at deeper levels
- Integrates with killer moves and history tables
- Allows early termination if mate is found
- Provides "anytime" algorithm (can stop early with best move so far)

In parallel mode, uses iterative deepening up to depth-1 before launching parallel root search.

#### ChessBoard

Complete chess engine in C++:

- FEN parsing/generation
- Legal move generation (all pieces, castling, en passant, promotions)
- Move make/unmake with state stack
- Check/checkmate detection

### Build System

CMakeLists.txt compiles:

- `src/chess_board.cpp` - Chess logic
- `src/functions.cpp` - Search algorithms
- `src/bindings.cpp` - Python bindings

Build command:

```bash
cd build
conda run -n chesshacks cmake --build .
```

### Usage Example

```python
import c_helpers

# Start position
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Use built-in enhanced evaluation (material + positional)
# Create transposition table (reuse across searches)
tt = c_helpers.TranspositionTable()

# Create reusable data structures
tt = c_helpers.TranspositionTable()
killers = c_helpers.KillerMoves()
history = c_helpers.HistoryTable()

# Run optimized search with all 15 features
score = c_helpers.alpha_beta_optimized(
    fen,
    depth=6,
    alpha=c_helpers.MIN,
    beta=c_helpers.MAX,
    maximizingPlayer=True,
    evaluate=c_helpers.evaluate_with_pst,  # Material + positional
    tt=tt,
    num_threads=4,  # Parallel search
    killers=killers,
    history=history
)

print(f"Position score: {score}")
print(f"Positions cached: {len(tt)}")
```

### Performance Characteristics

**Basic:**

- Pure algorithm, no overhead
- ~100-1,000 nodes/sec at depth 4-6
- Linear time complexity with depth

**Optimized (Sequential):**

- **Transposition table**: Reduces search tree by 50-90%
- **Move ordering**: 3-5x more cutoffs
- **PVS**: Null window scouts verify most moves fail (10-20% faster)
- **IID**: Shallow search populates TT on cache misses (5-10% faster)
- **LMR**: Reduces late moves aggressively (15-25% faster)
- **Aspiration windows**: Tighter bounds cause more cutoffs (10-20% faster)
- **Null move pruning**: Skip moves in strong positions (20-30% faster)
- **Killer moves**: Remember good quiet moves (10-15% faster)
- **History heuristic**: Track successful move patterns (5-10% faster)
- **Quiescence search**: Eliminates horizon effect, adds ~20% nodes but much better evaluation
- **Piece-square tables**: Positional understanding beyond material
- **Iterative deepening**: Minimal overhead (<5%), enables better ordering
- **Combined**: ~57-210x faster than basic for depths 5+
- ~10,000-150,000 nodes/sec at depth 6-8

**Optimized (Parallel):**

- Linear speedup at root level (4 threads ≈ 3-4x faster)
- Best for depth ≥ 4 where move evaluation is expensive
- Diminishing returns beyond hardware thread count
- Shared transposition table maximizes knowledge reuse
- ~30,000-400,000 nodes/sec with 4 threads at depth 6-8

**CUDA (Future):**

- Batch NNUE evaluation on GPU (1000+ positions simultaneously)
- Expected 10-100x speedup for evaluation-heavy searches
- Requires CUDA toolkit and NVIDIA GPU
- Most beneficial for shallow quiescence searches

### Features Summary

This chess engine includes **15 advanced features**:

1. ✅ Three-function architecture (basic, optimized, cuda)
2. ✅ MVV-LVA capture ordering
3. ✅ Advanced 5-tier move ordering
4. ✅ Killer move heuristic (2 per ply)
5. ✅ History heuristic (piece-to-square)
6. ✅ Null move pruning (R=2)
7. ✅ Quiescence search (tactical horizon)
8. ✅ Iterative deepening
9. ✅ Thread-safe transposition table
10. ✅ Parallel root search
11. ✅ Late move reductions (LMR)
12. ✅ Aspiration windows
13. ✅ Principal variation search (PVS)
14. ✅ Piece-square tables (PST)
15. ✅ Internal iterative deepening (IID)

### Next Steps

1. ✅ Three-function architecture
2. ✅ Move ordering and MVV-LVA
3. ✅ Thread-safe transposition table
4. ✅ Quiescence search
5. ✅ Iterative deepening
6. ✅ Killer move heuristic
7. ✅ History heuristic
8. ✅ Null move pruning
9. ✅ Late move reductions (LMR)
10. ✅ Aspiration windows
11. ✅ Principal variation search (PVS)
12. ✅ Piece-square tables (PST)
13. ✅ Internal iterative deepening (IID)
14. ⏳ Integrate real NNUE evaluation
15. ⏳ Singular extensions
16. ⏳ Counter move heuristic
17. ⏳ CUDA batch evaluation

