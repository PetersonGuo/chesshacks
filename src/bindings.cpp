#include "functions.h"
#ifdef CUDA_ENABLED
#include "cuda/cuda_eval.h"
#include "evaluation.h"
#endif
#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

using bitboard::BitboardState;

namespace {

std::array<int8_t, 64> PiecesListToArray(const nb::list &pieces) {
  if (pieces.size() != 64) {
    throw nb::value_error("Expected 64 entries for board representation");
  }
  std::array<int8_t, 64> encoded{};
  for (size_t i = 0; i < 64; ++i) {
    encoded[i] = nb::cast<int8_t>(pieces[i]);
  }
  return encoded;
}

void SetStateFromComponents(BitboardState &state, const nb::list &pieces,
                            bool white_to_move, const std::string &castling,
                            int en_passant_square, int halfmove_clock,
                            int fullmove_number) {
  auto encoded = PiecesListToArray(pieces);
  state.set_from_components(encoded, white_to_move, castling, en_passant_square,
                            halfmove_clock, fullmove_number);
}

BitboardState
BuildStateFromComponents(const nb::list &pieces, bool white_to_move,
                         const std::string &castling, int en_passant_square,
                         int halfmove_clock, int fullmove_number) {
  auto encoded = PiecesListToArray(pieces);
  return BitboardState::from_components(encoded, white_to_move, castling,
                                        en_passant_square, halfmove_clock,
                                        fullmove_number);
}

} // namespace

NB_MODULE(c_helpers, m) {
  m.doc() = "ChessHacks C++ extension module";

#ifdef CUDA_ENABLED
  // Initialize CUDA piece-square tables
  cuda_init_tables(
      PieceSquareTables::pawn_table, PieceSquareTables::knight_table,
      PieceSquareTables::bishop_table, PieceSquareTables::rook_table,
      PieceSquareTables::queen_table, PieceSquareTables::king_middlegame_table);
#endif

  // Expose constants as attributes
  m.attr("MIN") = MIN;
  m.attr("MAX") = MAX;

  // Expose TranspositionTable class
  nb::class_<TranspositionTable>(m, "TranspositionTable")
      .def(nb::init<>(), "Create a new transposition table")
      .def("clear", &TranspositionTable::clear, "Clear all cached positions")
      .def("size", &TranspositionTable::size, "Get number of cached positions")
      .def("__len__", &TranspositionTable::size);

  nb::class_<BitboardState>(m, "BitboardState")
      .def(nb::init<>())
      .def(nb::init<const std::string &>())
      .def("to_fen", &BitboardState::to_fen)
      .def("zobrist", &BitboardState::zobrist)
      .def("set_from_fen", &BitboardState::set_from_fen)
      .def(
          "set_from_components",
          [](BitboardState &state, const nb::list &pieces, bool white_to_move,
             const std::string &castling, int en_passant_square,
             int halfmove_clock, int fullmove_number) {
            SetStateFromComponents(state, pieces, white_to_move, castling,
                                   en_passant_square, halfmove_clock,
                                   fullmove_number);
          },
          nb::arg("pieces"), nb::arg("white_to_move"), nb::arg("castling"),
          nb::arg("en_passant_square"), nb::arg("halfmove_clock") = 0,
          nb::arg("fullmove_number") = 1)
      .def_static(
          "from_components",
          [](const nb::list &pieces, bool white_to_move,
             const std::string &castling, int en_passant_square,
             int halfmove_clock, int fullmove_number) {
            return BuildStateFromComponents(pieces, white_to_move, castling,
                                            en_passant_square, halfmove_clock,
                                            fullmove_number);
          },
          nb::arg("pieces"), nb::arg("white_to_move"), nb::arg("castling"),
          nb::arg("en_passant_square"), nb::arg("halfmove_clock") = 0,
          nb::arg("fullmove_number") = 1);

  m.def(
      "bitboard_from_components",
      [](const nb::list &pieces, bool white_to_move,
         const std::string &castling, int en_passant_square, int halfmove_clock,
         int fullmove_number) {
        return BuildStateFromComponents(pieces, white_to_move, castling,
                                        en_passant_square, halfmove_clock,
                                        fullmove_number);
      },
      nb::arg("pieces"), nb::arg("white_to_move"), nb::arg("castling"),
      nb::arg("en_passant_square"), nb::arg("halfmove_clock") = 0,
      nb::arg("fullmove_number") = 1,
      "Construct a BitboardState from raw piece information.");

  // Expose KillerMoves class
  nb::class_<KillerMoves>(m, "KillerMoves")
      .def(nb::init<>(), "Create a new killer moves table")
      .def("clear", &KillerMoves::clear, "Clear all killer moves");

  // Expose HistoryTable class
  nb::class_<HistoryTable>(m, "HistoryTable")
      .def(nb::init<>(), "Create a new history heuristic table")
      .def("clear", &HistoryTable::clear, "Clear all history scores")
      .def("age", &HistoryTable::age, "Age history scores (divide by 2)");

  // Expose CounterMoveTable class
  nb::class_<CounterMoveTable>(m, "CounterMoveTable")
      .def(nb::init<>(), "Create a new counter move table")
      .def("clear", &CounterMoveTable::clear, "Clear all counter moves");

  // Evaluation functions
  m.def("evaluate_with_pst",
        static_cast<int (*)(const std::string &)>(&evaluate_with_pst),
        nb::arg("fen"),
        "Enhanced evaluation function using material + piece-square tables.\n"
        "Provides positional bonuses based on piece placement.");

  m.def(
      "evaluate_material",
      static_cast<int (*)(const std::string &)>(&evaluate_material),
      nb::arg("fen"),
      "Material-only evaluation implemented in C++ (formerly Python lambda).");

  const auto pst_eval_ptr =
      static_cast<int (*)(const std::string &)>(evaluate_with_pst);
  const auto material_eval_ptr =
      static_cast<int (*)(const std::string &)>(evaluate_material);
  nb::object builtin_eval_with_pst = m.attr("evaluate_with_pst");
  nb::object builtin_eval_material = m.attr("evaluate_material");

  auto resolve_eval =
      [pst_eval_ptr, material_eval_ptr, builtin_eval_with_pst,
       builtin_eval_material](
          nb::handle eval_obj) -> std::function<int(const std::string &)> {
    if (!eval_obj || eval_obj.is_none() || eval_obj.is(builtin_eval_with_pst)) {
      return pst_eval_ptr;
    }
    if (eval_obj.is(builtin_eval_material)) {
      return material_eval_ptr;
    }
    return nb::cast<std::function<int(const std::string &)>>(
        nb::object(eval_obj, nb::detail::borrow_t{}));
  };

  // 1. BASIC: Bare-bones alpha-beta (no optimizations) - BACKUP
  m.def(
      "alpha_beta_basic",
      [resolve_eval](const std::string &fen, int depth, int alpha, int beta,
                     bool maximizingPlayer, nb::object evaluate) {
        auto eval_fn = resolve_eval(evaluate);
        return alpha_beta_basic(fen, depth, alpha, beta, maximizingPlayer,
                                eval_fn);
      },
      nb::arg("fen"), nb::arg("depth"), nb::arg("alpha"), nb::arg("beta"),
      nb::arg("maximizingPlayer"), nb::arg("evaluate") = nb::none(),
      "Basic alpha-beta pruning search (no optimizations).\n"
      "This is the fallback version with minimal dependencies.\n"
      "evaluate: Python callback function that takes a FEN string and "
      "returns an int score");
  m.def("alpha_beta_basic_builtin",
        static_cast<int (*)(const std::string &, int, int, int, bool)>(
            &alpha_beta_basic_builtin),
        nb::arg("fen"), nb::arg("depth"), nb::arg("alpha"), nb::arg("beta"),
        nb::arg("maximizingPlayer"),
        "Basic alpha-beta search using the built-in C++ material evaluator.");

  m.def(
      "alpha_beta_basic_state",
      [resolve_eval](const BitboardState &state, int depth, int alpha, int beta,
                     bool maximizingPlayer, nb::object evaluate) {
        BitboardState copy = state;
        auto eval_fn = resolve_eval(evaluate);
        return alpha_beta_basic(copy, depth, alpha, beta, maximizingPlayer,
                                eval_fn);
      },
      nb::arg("state"), nb::arg("depth"), nb::arg("alpha"), nb::arg("beta"),
      nb::arg("maximizingPlayer"), nb::arg("evaluate") = nb::none(),
      "Basic alpha-beta search operating directly on a BitboardState.");

  m.def(
      "alpha_beta_basic_builtin_state",
      [](const BitboardState &state, int depth, int alpha, int beta,
         bool maximizingPlayer) {
        BitboardState copy = state;
        return alpha_beta_basic_builtin(copy, depth, alpha, beta,
                                        maximizingPlayer);
      },
      nb::arg("state"), nb::arg("depth"), nb::arg("alpha"), nb::arg("beta"),
      nb::arg("maximizingPlayer"),
      "Built-in evaluator alpha-beta search using BitboardState input.");

  // 2. OPTIMIZED: Full optimizations (TT + move ordering + parallel + killer
  // moves + history + null move pruning)
  m.def(
      "alpha_beta_optimized",
      [resolve_eval](const std::string &fen, int depth, int alpha, int beta,
                     bool maximizingPlayer, nb::object evaluate,
                     TranspositionTable *tt, int num_threads,
                     KillerMoves *killers, HistoryTable *history,
                     CounterMoveTable *counters) {
        auto eval_fn = resolve_eval(evaluate);
        return alpha_beta_optimized(fen, depth, alpha, beta, maximizingPlayer,
                                    eval_fn, tt, num_threads, killers, history,
                                    counters);
      },
      nb::arg("fen"), nb::arg("depth"), nb::arg("alpha"), nb::arg("beta"),
      nb::arg("maximizingPlayer"), nb::arg("evaluate") = nb::none(),
      nb::arg("tt") = nullptr, nb::arg("num_threads") = 0,
      nb::arg("killers") = nullptr, nb::arg("history") = nullptr,
      nb::arg("counters") = nullptr,
      "Optimized alpha-beta with all enhancements:\n"
      "- Transposition table caching\n"
      "- Advanced move ordering (TT + killer moves + MVV-LVA + history + "
      "counter moves)\n"
      "- Quiescence search\n"
      "- Iterative deepening\n"
      "- Null move pruning\n"
      "- Singular extensions\n"
      "- Optional multithreading\n\n"
      "tt: Optional TranspositionTable instance (creates local one if null)\n"
      "num_threads: Number of threads for parallel search (0 = auto, 1 = "
      "sequential)\n"
      "killers: Optional KillerMoves instance (creates local one if null)\n"
      "history: Optional HistoryTable instance (creates local one if null)\n"
      "counters: Optional CounterMoveTable instance (creates local one if "
      "null)");

  m.def("alpha_beta_optimized_builtin",
        static_cast<int (*)(const std::string &, int, int, int, bool,
                            TranspositionTable *, int, KillerMoves *,
                            HistoryTable *, CounterMoveTable *)>(
            &alpha_beta_optimized_builtin),
        nb::arg("fen"), nb::arg("depth"), nb::arg("alpha"), nb::arg("beta"),
        nb::arg("maximizingPlayer"), nb::arg("tt") = nullptr,
        nb::arg("num_threads") = 0, nb::arg("killers") = nullptr,
        nb::arg("history") = nullptr, nb::arg("counters") = nullptr,
        "Optimized alpha-beta that uses the built-in material evaluator in "
        "C++. No Python callbacks, ideal for multithreading.");

  m.def(
      "alpha_beta_optimized_state",
      [resolve_eval](const BitboardState &state, int depth, int alpha, int beta,
                     bool maximizingPlayer, nb::object evaluate,
                     TranspositionTable *tt, int num_threads,
                     KillerMoves *killers, HistoryTable *history,
                     CounterMoveTable *counters) {
        BitboardState copy = state;
        auto eval_fn = resolve_eval(evaluate);
        return alpha_beta_optimized(copy, depth, alpha, beta, maximizingPlayer,
                                    eval_fn, tt, num_threads, killers, history,
                                    counters);
      },
      nb::arg("state"), nb::arg("depth"), nb::arg("alpha"), nb::arg("beta"),
      nb::arg("maximizingPlayer"), nb::arg("evaluate") = nb::none(),
      nb::arg("tt") = nullptr, nb::arg("num_threads") = 0,
      nb::arg("killers") = nullptr, nb::arg("history") = nullptr,
      nb::arg("counters") = nullptr,
      "Optimized alpha-beta that accepts a BitboardState directly.");

  m.def(
      "alpha_beta_optimized_builtin_state",
      [](const BitboardState &state, int depth, int alpha, int beta,
         bool maximizingPlayer, TranspositionTable *tt, int num_threads,
         KillerMoves *killers, HistoryTable *history,
         CounterMoveTable *counters) {
        BitboardState copy = state;
        return alpha_beta_optimized_builtin(copy, depth, alpha, beta,
                                            maximizingPlayer, tt, num_threads,
                                            killers, history, counters);
      },
      nb::arg("state"), nb::arg("depth"), nb::arg("alpha"), nb::arg("beta"),
      nb::arg("maximizingPlayer"), nb::arg("tt") = nullptr,
      nb::arg("num_threads") = 0, nb::arg("killers") = nullptr,
      nb::arg("history") = nullptr, nb::arg("counters") = nullptr,
      "BitboardState variant of optimized alpha-beta with built-in evaluator.");

  // 3. CUDA: GPU-accelerated search (falls back to optimized)
  m.def(
      "alpha_beta_cuda",
      [resolve_eval](const std::string &fen, int depth, int alpha, int beta,
                     bool maximizingPlayer, nb::object evaluate,
                     TranspositionTable *tt, KillerMoves *killers,
                     HistoryTable *history, CounterMoveTable *counters) {
        auto eval_fn = resolve_eval(evaluate);
        return alpha_beta_cuda(fen, depth, alpha, beta, maximizingPlayer,
                               eval_fn, tt, killers, history, counters);
      },
      nb::arg("fen"), nb::arg("depth"), nb::arg("alpha"), nb::arg("beta"),
      nb::arg("maximizingPlayer"), nb::arg("evaluate") = nb::none(),
      nb::arg("tt") = nullptr, nb::arg("killers") = nullptr,
      nb::arg("history") = nullptr, nb::arg("counters") = nullptr,
      "CUDA-accelerated alpha-beta search (currently falls back to optimized "
      "CPU version).\n"
      "tt: Optional TranspositionTable instance\n"
      "killers: Optional KillerMoves instance\n"
      "history: Optional HistoryTable instance\n"
      "counters: Optional CounterMoveTable instance");

  m.def(
      "alpha_beta_cuda_state",
      [resolve_eval](const BitboardState &state, int depth, int alpha, int beta,
                     bool maximizingPlayer, nb::object evaluate,
                     TranspositionTable *tt, KillerMoves *killers,
                     HistoryTable *history, CounterMoveTable *counters) {
        BitboardState copy = state;
        auto eval_fn = resolve_eval(evaluate);
        return alpha_beta_cuda(copy, depth, alpha, beta, maximizingPlayer,
                               eval_fn, tt, killers, history, counters);
      },
      nb::arg("state"), nb::arg("depth"), nb::arg("alpha"), nb::arg("beta"),
      nb::arg("maximizingPlayer"), nb::arg("evaluate") = nb::none(),
      nb::arg("tt") = nullptr, nb::arg("killers") = nullptr,
      nb::arg("history") = nullptr, nb::arg("counters") = nullptr,
      "BitboardState variant of the CUDA-accelerated alpha-beta search.");

  // Batch evaluation (multithreaded CPU)
  m.def("batch_evaluate_mt", &batch_evaluate_mt, nb::arg("fens"),
        nb::arg("num_threads") = 0,
        "Batch evaluate multiple positions using multithreading.\n"
        "Much more efficient than evaluating positions one at a time.\n\n"
        "fens: List of FEN strings to evaluate\n"
        "num_threads: Number of threads (0 = auto-detect all available cores)\n"
        "Returns: List of evaluation scores");

  // NNUE evaluation functions
  m.def("init_nnue", &init_nnue, nb::arg("model_path"),
        "Initialize NNUE evaluator with a trained model.\n"
        "Load NNUE model from binary file (.bin).\n\n"
        "model_path: Path to binary NNUE model file\n"
        "Returns: True if loaded successfully, False otherwise");

  m.def("is_nnue_loaded", &is_nnue_loaded,
        "Check if NNUE model is loaded and ready for evaluation.\n"
        "Returns: True if NNUE is loaded, False otherwise");

  m.def("evaluate_nnue", nb::overload_cast<const std::string &>(&evaluate_nnue),
        nb::arg("fen"),
        "Evaluate position using NNUE model.\n"
        "Falls back to PST evaluation if NNUE not loaded.\n\n"
        "fen: Position in FEN notation\n"
        "Returns: Evaluation score in centipawns");

  m.def("evaluate", nb::overload_cast<const std::string &>(&evaluate),
        nb::arg("fen"),
        "Evaluate position using best available method.\n"
        "Uses NNUE if loaded, otherwise PST evaluation.\n\n"
        "fen: Position in FEN notation\n"
        "Returns: Evaluation score in centipawns");

  // CUDA availability check
  m.def("is_cuda_available", &is_cuda_available,
        "Check if CUDA is available for GPU acceleration.\n"
        "Returns True if CUDA devices are detected and accessible.");

  m.def("get_cuda_info", &get_cuda_info,
        "Get information about available CUDA devices.\n"
        "Returns a string describing the GPU and CUDA version.");

#ifdef CUDA_ENABLED
  // CUDA batch operations - using Python-friendly wrappers
  m.def("cuda_batch_evaluate", &cuda_batch_evaluate_py, nb::arg("fens"),
        "Batch evaluate multiple chess positions on GPU.\n"
        "Much more efficient than evaluating positions one at a time.\n\n"
        "fens: List of FEN strings to evaluate\n"
        "Returns: List of evaluation scores");

  m.def("cuda_batch_count_pieces", &cuda_batch_count_pieces_py, nb::arg("fens"),
        "Count pieces in multiple positions on GPU.\n"
        "Returns counts for each piece type (6 white, 6 black).\n\n"
        "fens: List of FEN strings\n"
        "Returns: List of lists with piece counts");

  m.def("cuda_batch_hash_positions", &cuda_batch_hash_positions_py,
        nb::arg("fens"),
        "Generate hash values for multiple positions on GPU.\n"
        "Used for transposition table lookups.\n\n"
        "fens: List of FEN strings\n"
        "Returns: List of hash values");
#endif

  // 4. PGN to FEN: Convert PGN string to FEN string
  m.def("pgn_to_fen", &pgn_to_fen, nb::arg("pgn"),
        "Convert PGN (Portable Game Notation) string to FEN (Forsyth-Edwards "
        "Notation) string.\n"
        "Parses PGN moves and returns the final position as FEN.\n"
        "pgn: PGN string containing game moves");

  // Best move finders
  m.def(
      "find_best_move",
      [resolve_eval](const std::string &fen, int depth, nb::object evaluate,
                     TranspositionTable *tt, int num_threads,
                     KillerMoves *killers, HistoryTable *history,
                     CounterMoveTable *counters) {
        auto eval_fn = resolve_eval(evaluate);
        return find_best_move(fen, depth, eval_fn, tt, num_threads, killers,
                              history, counters);
      },
      nb::arg("fen"), nb::arg("depth"), nb::arg("evaluate") = nb::none(),
      nb::arg("tt") = nullptr, nb::arg("num_threads") = 0,
      nb::arg("killers") = nullptr, nb::arg("history") = nullptr,
      nb::arg("counters") = nullptr,
      "Find the best move for a position.\n"
      "Returns the FEN string of the position after the best move.\n\n"
      "fen: Position in FEN notation\n"
      "depth: Search depth\n"
      "evaluate: Evaluation function\n"
      "tt: Optional TranspositionTable\n"
      "num_threads: Number of threads (0=auto)\n"
      "killers: Optional KillerMoves\n"
      "history: Optional HistoryTable\n"
      "counters: Optional CounterMoveTable");

  m.def("find_best_move_builtin",
        static_cast<std::string (*)(
            const std::string &, int, TranspositionTable *, int, KillerMoves *,
            HistoryTable *, CounterMoveTable *)>(&find_best_move_builtin),
        nb::arg("fen"), nb::arg("depth"), nb::arg("tt") = nullptr,
        nb::arg("num_threads") = 0, nb::arg("killers") = nullptr,
        nb::arg("history") = nullptr, nb::arg("counters") = nullptr,
        "Find the best move using the built-in C++ material evaluator (no "
        "Python callback).");

  m.def(
      "find_best_move_state",
      [resolve_eval](const BitboardState &state, int depth, nb::object evaluate,
                     TranspositionTable *tt, int num_threads,
                     KillerMoves *killers, HistoryTable *history,
                     CounterMoveTable *counters) {
        BitboardState copy = state;
        auto eval_fn = resolve_eval(evaluate);
        return find_best_move(copy, depth, eval_fn, tt, num_threads, killers,
                              history, counters);
      },
      nb::arg("state"), nb::arg("depth"), nb::arg("evaluate") = nb::none(),
      nb::arg("tt") = nullptr, nb::arg("num_threads") = 0,
      nb::arg("killers") = nullptr, nb::arg("history") = nullptr,
      nb::arg("counters") = nullptr,
      "Find the best move using a BitboardState input.");

  m.def(
      "find_best_move_builtin_state",
      [](const BitboardState &state, int depth, TranspositionTable *tt,
         int num_threads, KillerMoves *killers, HistoryTable *history,
         CounterMoveTable *counters) {
        BitboardState copy = state;
        return find_best_move_builtin(copy, depth, tt, num_threads, killers,
                                      history, counters);
      },
      nb::arg("state"), nb::arg("depth"), nb::arg("tt") = nullptr,
      nb::arg("num_threads") = 0, nb::arg("killers") = nullptr,
      nb::arg("history") = nullptr, nb::arg("counters") = nullptr,
      "Built-in evaluator best-move search using BitboardState.");

  m.def(
      "get_best_move_uci",
      [resolve_eval](const std::string &fen, int depth, nb::object evaluate,
                     TranspositionTable *tt, int num_threads,
                     KillerMoves *killers, HistoryTable *history,
                     CounterMoveTable *counters) {
        auto eval_fn = resolve_eval(evaluate);
        return get_best_move_uci(fen, depth, eval_fn, tt, num_threads, killers,
                                 history, counters);
      },
      nb::arg("fen"), nb::arg("depth"), nb::arg("evaluate") = nb::none(),
      nb::arg("tt") = nullptr, nb::arg("num_threads") = 0,
      nb::arg("killers") = nullptr, nb::arg("history") = nullptr,
      nb::arg("counters") = nullptr,
      "Find the best move and return it in UCI format (e.g., 'e2e4').\n"
      "Returns the move in UCI notation.\n\n"
      "fen: Position in FEN notation\n"
      "depth: Search depth\n"
      "evaluate: Evaluation function\n"
      "tt: Optional TranspositionTable\n"
      "num_threads: Number of threads (0=auto)\n"
      "killers: Optional KillerMoves\n"
      "history: Optional HistoryTable\n"
      "counters: Optional CounterMoveTable");

  m.def("get_best_move_uci_builtin",
        static_cast<std::string (*)(
            const std::string &, int, TranspositionTable *, int, KillerMoves *,
            HistoryTable *, CounterMoveTable *)>(&get_best_move_uci_builtin),
        nb::arg("fen"), nb::arg("depth"), nb::arg("tt") = nullptr,
        nb::arg("num_threads") = 0, nb::arg("killers") = nullptr,
        nb::arg("history") = nullptr, nb::arg("counters") = nullptr,
        "Find the best move (UCI) using the built-in C++ material evaluator.");

  m.def(
      "get_best_move_uci_state",
      [resolve_eval](const BitboardState &state, int depth, nb::object evaluate,
                     TranspositionTable *tt, int num_threads,
                     KillerMoves *killers, HistoryTable *history,
                     CounterMoveTable *counters) {
        BitboardState copy = state;
        auto eval_fn = resolve_eval(evaluate);
        return get_best_move_uci(copy, depth, eval_fn, tt, num_threads, killers,
                                 history, counters);
      },
      nb::arg("state"), nb::arg("depth"), nb::arg("evaluate") = nb::none(),
      nb::arg("tt") = nullptr, nb::arg("num_threads") = 0,
      nb::arg("killers") = nullptr, nb::arg("history") = nullptr,
      nb::arg("counters") = nullptr,
      "Get the best move in UCI format using a BitboardState.");

  m.def(
      "get_best_move_uci_builtin_state",
      [](const BitboardState &state, int depth, TranspositionTable *tt,
         int num_threads, KillerMoves *killers, HistoryTable *history,
         CounterMoveTable *counters) {
        BitboardState copy = state;
        return get_best_move_uci_builtin(copy, depth, tt, num_threads, killers,
                                         history, counters);
      },
      nb::arg("state"), nb::arg("depth"), nb::arg("tt") = nullptr,
      nb::arg("num_threads") = 0, nb::arg("killers") = nullptr,
      nb::arg("history") = nullptr, nb::arg("counters") = nullptr,
      "Built-in evaluator best move using BitboardState.");

  // Multi-PV Search
  nb::class_<PVLine>(m, "PVLine")
      .def_rw("uci_move", &PVLine::uci_move, "Best move in UCI format")
      .def_rw("score", &PVLine::score, "Evaluation score")
      .def_rw("depth", &PVLine::depth, "Search depth")
      .def_rw("pv", &PVLine::pv, "Principal variation");

  m.def(
      "multi_pv_search",
      [resolve_eval](const std::string &fen, int depth, int num_lines,
                     nb::object evaluate, TranspositionTable *tt,
                     int num_threads, KillerMoves *killers,
                     HistoryTable *history, CounterMoveTable *counters) {
        auto eval_fn = resolve_eval(evaluate);
        return multi_pv_search(fen, depth, num_lines, eval_fn, tt, num_threads,
                               killers, history, counters);
      },
      nb::arg("fen"), nb::arg("depth"), nb::arg("num_lines"),
      nb::arg("evaluate") = nb::none(), nb::arg("tt") = nullptr,
      nb::arg("num_threads") = 0, nb::arg("killers") = nullptr,
      nb::arg("history") = nullptr, nb::arg("counters") = nullptr,
      "Search for multiple principal variations.\n"
      "Returns list of top N moves with scores.\n\n"
      "fen: Position in FEN notation\n"
      "depth: Search depth\n"
      "num_lines: Number of lines to return\n"
      "evaluate: Evaluation function\n"
      "tt: Optional TranspositionTable\n"
      "num_threads: Number of threads (0=auto)\n"
      "killers: Optional KillerMoves\n"
      "history: Optional HistoryTable\n"
      "counters: Optional CounterMoveTable");

  m.def(
      "multi_pv_search_state",
      [resolve_eval](const BitboardState &state, int depth, int num_lines,
                     nb::object evaluate, TranspositionTable *tt,
                     int num_threads, KillerMoves *killers,
                     HistoryTable *history, CounterMoveTable *counters) {
        BitboardState copy = state;
        auto eval_fn = resolve_eval(evaluate);
        return multi_pv_search(copy, depth, num_lines, eval_fn, tt, num_threads,
                               killers, history, counters);
      },
      nb::arg("state"), nb::arg("depth"), nb::arg("num_lines"),
      nb::arg("evaluate") = nb::none(), nb::arg("tt") = nullptr,
      nb::arg("num_threads") = 0, nb::arg("killers") = nullptr,
      nb::arg("history") = nullptr, nb::arg("counters") = nullptr,
      "Compute multiple principal variations from a BitboardState.");

  m.def(
      "set_max_search_depth", &set_max_search_depth, nb::arg("depth") = 5,
      "Set the global maximum search depth (default = 5). "
      "Pass values > 5 to allow deeper searches; values <=0 reset to default.");
}
