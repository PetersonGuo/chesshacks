# ChessHacks

High-performance chess engine with NNUE evaluation, aggressively optimized alpha-beta search, CUDA helpers, and a FastAPI/Next.js control plane. This README is the single source of truth for setup, architecture, and day-to-day commands.

---

## 1. Quick Start

```bash
# 1) Python environment (conda or venv)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2) Build native extension (nanobind + libtorch auto-download)
./build.sh
source ./.chesshacks_env    # exports PYTHONPATH/LD_LIBRARY_PATH/etc.

# 3) Dev UI + API
cd devtools && npm install
cp .env.template .env.local
npm run dev                 # spawns FastAPI backend + Next.js front-end
```

Key paths:

-   `src/` – C++ search, evaluation, CUDA helpers, nanobind bindings.
-   `serve.py` – FastAPI entrypoint; UI launches this automatically.
-   `tests/` & `benchmarks/` – Pytest suites and perf utilities.
-   `build/` – Out-of-tree CMake build; contains the `c_helpers` module.
-   `third_party/surge/` – Magic-bitboard move generator submodule.

---

## 2. Engine Snapshot

| Component         | Notes                                                                                   |
| ----------------- | --------------------------------------------------------------------------------------- |
| `BitboardState`   | 64‑square board with make/unmake, hashing, serialization, and surge interop.            |
| `alpha_beta`      | Production search: TT, killer/history/counter tables, MVV-LVA+SEE, null move, LMR, PVS. |
| `alpha_beta_cuda` | Uses same logic but can dispatch batched evaluations to CUDA if available.              |
| Evaluator         | `c_helpers.evaluate` auto-selects NNUE if loaded, otherwise material-only fallback.     |
| Batch helpers     | `batch_evaluate_mt`, `cuda_batch_*` cover CPU/GPU evaluation and analytics.             |

Usage example:

```python
state = c_helpers.BitboardState(start_fen)
tt = c_helpers.TranspositionTable()
score = c_helpers.alpha_beta(
    state,
    depth=6,
    alpha=c_helpers.MIN,
    beta=c_helpers.MAX,
    maximizingPlayer=True,
    evaluate=c_helpers.evaluate,
    tt=tt,
    num_threads=0,   # auto threads
)
```

---

## 3. CUDA & Performance

-   Enable with `-DCHESSHACKS_ENABLE_CUDA=ON` (CMake auto-detects architectures or respect `CMAKE_CUDA_ARCHITECTURES`).
-   `c_helpers.is_cuda_available()` / `get_cuda_info()` expose runtime status.
-   Kernels live in `src/cuda/` (material eval, piece counts, hashing, MVV-LVA, bitonic move ordering).
-   Heuristic guidance:

| Scenario                    | Recommendation                                             |
| --------------------------- | ---------------------------------------------------------- |
| Depth ≤ 5                   | Stay single-threaded for cache friendliness.               |
| Depth ≥ 6 / tactical search | `num_threads=0` (auto). Shared heuristics are thread-safe. |
| Batch < 500 positions       | CPU sequential evaluation.                                 |
| Batch 500–5k                | `batch_evaluate_mt(..., num_threads)`                      |
| Batch > 5k                  | `cuda_batch_evaluate` amortizes PCIe copies.               |

Benchmarks: `benchmark_multithreading.py`, `benchmark_profile_multithreading.py`, `benchmark_cuda.py`, `benchmark_cpp_functions.py`.

---

## 4. Testing & Tooling

```bash
# Configure + build (if you skipped ./build.sh)
cmake -S . -B build
cmake --build build

# C++ tests (GoogleTest)
cd build
ctest --output-on-failure   # runs tests/cpp_tests
./tests/cpp_tests           # alternatively run the binary directly

# Python suites (pytest)
python -m pytest ../tests -v
python -m pytest tests/bot/test_parallel_optimizations.py -v

# Dev UI (auto rebuild backend, watch logs)
npm run dev
```

CI helpers:

-   `run_tests.sh` – wrapper used by pipelines and local smoke tests.
-   `scripts/` – misc automation (formatting, env introspection).
-   `.chesshacks_env` – always source before running `serve.py` directly.

---

## 5. NNUE Pipeline (train/nnue_model)

-   Architecture: Virgo-style features (768 inputs) → 256 → 32 → 32 → linear output with ClippedReLU activations.
-   Binary export: `python train/nnue_model/export_model.py checkpoints/best_model.pt`.
-   Runtime:

```python
if c_helpers.init_nnue("train/nnue_model/checkpoints/best_model.bin"):
    print("NNUE ready")
state = c_helpers.BitboardState(fen)
score = c_helpers.evaluate(state)   # uses NNUE automatically
```

### Training cheat sheet

```bash
# default config
python train/nnue_model/train.py

# fast smoke test
python train/nnue_model/train.py --config fast

# resume automatically
python train/nnue_model/train.py --resume auto --no-prompt

# clean slate
python train/nnue_model/train.py --fresh
```

Configs (`train/nnue_model/configs/*.yaml`) cover CPU-only, RTX5070, and quality profiles. Data pipeline auto-downloads Lichess games, evaluates with Stockfish, and writes JSONL corpora.

---

## 6. Frequently Used Paths & Commands

| Task                          | Command / File                                                 |
| ----------------------------- | -------------------------------------------------------------- | -------------------------------------------------- |
| Rebuild native module + tests | `./build.sh` (uses existing venv)                              |
| Clean build artifacts         | `rm -rf build` or `./build.sh clean`                           |
| Launch FastAPI without UI     | `source .chesshacks_env && python serve.py`                    |
| Inspect libtorch variant      | `cmake -LAH                                                    | grep CHESSHACKS_LIBTORCH` (after configuring once) |
| Add tests                     | Python: `tests/` (pytest); C++: `tests/cpp/*.cpp` (GoogleTest) |
| Add benchmarks                | drop scripts in `benchmarks/` (CI runs the perf suite nightly) |

---

## 7. Third-Party Notes

-   **Surge** – bundled as a submodule. Provides magic-bitboard move generation, perft tooling, and Zobrist hashing. We wrap it inside `src/bitboard`.
-   **libtorch** – downloaded automatically via FetchContent. Override with:
    -   `CHESSHACKS_LIBTORCH_VERSION` (e.g. `2.3.0`)
    -   `CHESSHACKS_LIBTORCH_VARIANT` (`cpu`, `cu121`, `macos-arm64`, `rocm6.0`, …)
    -   `CHESSHACKS_ENABLE_CUDA=ON` / `CHESSHACKS_ENABLE_ROCM=ON`

---

## 8. Need Help?

-   Review this README (kept up to date as the single reference).
-   Search `tests/` and `benchmarks/` for real-world examples.
