#include "bitboard/bitboard_state.h"
#include "core/evaluation.h"
#include "core/move_ordering.h"
#include "core/search.h"
#include "core/transposition_table.h"
#include "core/utils.h"
#include "cuda/cuda_utils.h"
#ifdef CUDA_ENABLED
#include "core/evaluation.h"
#include "cuda/cuda_eval.h"
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

  m.def(
      "evaluate_material",
      [](const BitboardState &state) { return evaluate_material(state); },
      nb::arg("state"),
      "Material-only evaluation implemented in C++ (formerly Python lambda).");

  // NNUE evaluation functions (registered early so resolve_eval can reference)
  m.def("init_nnue", &init_nnue, nb::arg("model_path"),
        "Initialize NNUE evaluator with a trained model.\n"
        "Load NNUE model from binary file (.bin).\n\n"
        "model_path: Path to binary NNUE model file\n"
        "Returns: True if loaded successfully, False otherwise");

  m.def("is_nnue_loaded", &is_nnue_loaded,
        "Check if NNUE model is loaded and ready for evaluation.\n"
        "Returns: True if NNUE is loaded, False otherwise");

  m.def(
      "evaluate_nnue",
      [](const BitboardState &state) { return evaluate_nnue(state); },
      nb::arg("state"),
      "Evaluate position using NNUE model.\n"
      "Falls back to material evaluation if NNUE not loaded.\n\n"
      "state: BitboardState position\n"
      "Returns: Evaluation score in centipawns");

  m.def(
      "evaluate", [](const BitboardState &state) { return evaluate(state); },
      nb::arg("state"),
      "Evaluate position using best available method.\n"
      "Uses NNUE if loaded, otherwise material evaluation.\n\n"
      "state: BitboardState position\n"
      "Returns: Evaluation score in centipawns");

  nb::object builtin_eval_material = m.attr("evaluate_material");
  nb::object builtin_eval_nnue = m.attr("evaluate_nnue");
  nb::object builtin_eval_auto = m.attr("evaluate");

  auto resolve_eval =
      [builtin_eval_material, builtin_eval_nnue, builtin_eval_auto](
          nb::handle eval_obj) -> std::function<int(const BitboardState &)> {
    using EvalPtr = int (*)(const BitboardState &);
    if (!eval_obj || eval_obj.is_none() || eval_obj.is(builtin_eval_auto)) {
      return static_cast<EvalPtr>(&evaluate);
    }
    if (eval_obj.is(builtin_eval_material)) {
      return static_cast<EvalPtr>(&evaluate_material);
    }
    if (eval_obj.is(builtin_eval_nnue)) {
      return static_cast<EvalPtr>(&evaluate_nnue);
    }
    if (eval_obj.is(builtin_eval_auto)) {
      return static_cast<EvalPtr>(&evaluate);
    }
    nb::object callable(eval_obj, nb::detail::borrow_t{});
    return [callable](const BitboardState &state) -> int {
      nb::gil_scoped_acquire gil;
      try {
        return nb::cast<int>(callable(state));
      } catch (nb::python_error &err) {
        if (err.matches(PyExc_TypeError)) {
          err.restore();
          PyErr_Clear();
          return nb::cast<int>(callable(state.to_fen()));
        }
        throw;
      }
    };
  };

  auto alpha_beta_doc =
      "Production alpha-beta with all enhancements:\n"
      "- Transposition table caching\n"
      "- Advanced move ordering (TT + killer moves + MVV-LVA + history + "
      "counter moves)\n"
      "- Quiescence search\n"
      "- Iterative deepening\n"
      "- Null move pruning and singular extensions\n"
      "- Optional multithreading\n\n"
      "tt: Optional TranspositionTable instance (creates local one if null)\n"
      "num_threads: Number of threads for parallel search (0 = auto, 1 = "
      "sequential)\n"
      "killers/history/counters: Optional heuristic tables (locals created if "
      "null)";

  m.def("alpha_beta",
        static_cast<int (*)(BitboardState, int, int, int, bool,
                            TranspositionTable *, int, KillerMoves *,
                            HistoryTable *, CounterMoveTable *)>(
            &alpha_beta_builtin),
        nb::arg("state"), nb::arg("depth"), nb::arg("alpha"), nb::arg("beta"),
        nb::arg("maximizingPlayer"), nb::arg("tt") = nullptr,
        nb::arg("num_threads") = 0, nb::arg("killers") = nullptr,
        nb::arg("history") = nullptr, nb::arg("counters") = nullptr,
        alpha_beta_doc);

  auto alpha_beta_builtin_doc =
      "Alpha-beta using the built-in evaluation pipeline (material/NNUE).";
  m.def("alpha_beta_builtin",
        static_cast<int (*)(BitboardState, int, int, int, bool,
                            TranspositionTable *, int, KillerMoves *,
                            HistoryTable *, CounterMoveTable *)>(
            &alpha_beta_builtin),
        nb::arg("state"), nb::arg("depth"), nb::arg("alpha"), nb::arg("beta"),
        nb::arg("maximizingPlayer"), nb::arg("tt") = nullptr,
        nb::arg("num_threads") = 0, nb::arg("killers") = nullptr,
        nb::arg("history") = nullptr, nb::arg("counters") = nullptr,
        alpha_beta_builtin_doc);

  // 3. CUDA: GPU-accelerated search (falls back to optimized)
#ifdef CUDA_ENABLED
  auto alpha_beta_cuda_doc =
      "CUDA-accelerated alpha-beta search (currently falls back to optimized "
      "CPU version if CUDA is unavailable).";

  m.def(
      "alpha_beta_cuda",
      [](const BitboardState &state, int depth, int alpha, int beta,
         bool maximizingPlayer, TranspositionTable *tt, KillerMoves *killers,
         HistoryTable *history, CounterMoveTable *counters) {
        BitboardState copy = state;
        auto eval_fn = static_cast<int (*)(const BitboardState &)>(&evaluate);
        return alpha_beta_cuda(copy, depth, alpha, beta, maximizingPlayer,
                               eval_fn, tt, killers, history, counters);
      },
      nb::arg("state"), nb::arg("depth"), nb::arg("alpha"), nb::arg("beta"),
      nb::arg("maximizingPlayer"), nb::arg("tt") = nullptr,
      nb::arg("killers") = nullptr, nb::arg("history") = nullptr,
      nb::arg("counters") = nullptr, alpha_beta_cuda_doc);
#endif

  // Batch evaluation (multithreaded CPU)
  m.def(
      "batch_evaluate_mt", &batch_evaluate_mt, nb::arg("states"),
      nb::arg("num_threads") = 0,
      "Batch evaluate multiple BitboardState positions using multithreading.\n"
      "Much more efficient than evaluating positions one at a time.\n\n"
      "states: List of BitboardState objects to evaluate\n"
      "num_threads: Number of threads (0 = auto-detect all available cores)\n"
      "Returns: List of evaluation scores");

  // CUDA availability check
  m.def("is_cuda_available", &is_cuda_available,
        "Check if CUDA is available for GPU acceleration.\n"
        "Returns True if CUDA devices are detected and accessible.");

  m.def("get_cuda_info", &get_cuda_info,
        "Get information about available CUDA devices.\n"
        "Returns a string describing the GPU and CUDA version.");

#ifdef CUDA_ENABLED
  // CUDA batch operations - using Python-friendly wrappers
  m.def("cuda_batch_evaluate", &cuda_batch_evaluate_py, nb::arg("states"),
        "Batch evaluate multiple BitboardState positions on GPU.\n"
        "Much more efficient than evaluating positions one at a time.\n\n"
        "states: List of BitboardState objects to evaluate\n"
        "Returns: List of evaluation scores");

  m.def("cuda_batch_count_pieces", &cuda_batch_count_pieces_py,
        nb::arg("states"),
        "Count pieces in multiple positions on GPU.\n"
        "Returns counts for each piece type (6 white, 6 black).\n\n"
        "states: List of BitboardState objects\n"
        "Returns: List of lists with piece counts");

  m.def("cuda_batch_hash_positions", &cuda_batch_hash_positions_py,
        nb::arg("states"),
        "Generate hash values for multiple positions on GPU.\n"
        "Used for transposition table lookups.\n\n"
        "states: List of BitboardState objects\n"
        "Returns: List of hash values");
#endif

  // 4. PGN to FEN: Convert PGN string to FEN string
  m.def("pgn_to_fen", &pgn_to_fen, nb::arg("pgn"),
        "Convert PGN (Portable Game Notation) string to FEN (Forsyth-Edwards "
        "Notation) string.\n"
        "Parses PGN moves and returns the final position as FEN.\n"
        "pgn: PGN string containing game moves");
  m.def(
      "pgn_to_bitboard",
      [](const std::string &pgn) { return pgn_to_bitboard(pgn); },
      nb::arg("pgn"),
      "Convert PGN string directly into a BitboardState representing the "
      "final position.");

  // Best move finders

  auto find_best_move_binding =
      [resolve_eval](const BitboardState &state, int depth, nb::object evaluate,
                     TranspositionTable *tt, int num_threads,
                     KillerMoves *killers, HistoryTable *history,
                     CounterMoveTable *counters) {
        BitboardState copy = state;
        auto eval_fn = resolve_eval(evaluate);
        return find_best_move(copy, depth, eval_fn, tt, num_threads, killers,
                              history, counters);
      };
  m.def("find_best_move", find_best_move_binding, nb::arg("state"),
        nb::arg("depth"), nb::arg("evaluate") = nb::none(),
        nb::arg("tt") = nullptr, nb::arg("num_threads") = 0,
        nb::arg("killers") = nullptr, nb::arg("history") = nullptr,
        nb::arg("counters") = nullptr,
        "Find the best move for a position. Returns the resulting "
        "BitboardState.");
  m.def("find_best_move_state", find_best_move_binding, nb::arg("state"),
        nb::arg("depth"), nb::arg("evaluate") = nb::none(),
        nb::arg("tt") = nullptr, nb::arg("num_threads") = 0,
        nb::arg("killers") = nullptr, nb::arg("history") = nullptr,
        nb::arg("counters") = nullptr,
        "Alias of find_best_move for backwards compatibility.");

  auto find_best_move_builtin_binding =
      [](const BitboardState &state, int depth, TranspositionTable *tt,
         int num_threads, KillerMoves *killers, HistoryTable *history,
         CounterMoveTable *counters) {
        BitboardState copy = state;
        return find_best_move_builtin(copy, depth, tt, num_threads, killers,
                                      history, counters);
      };
  m.def("find_best_move_builtin", find_best_move_builtin_binding,
        nb::arg("state"), nb::arg("depth"), nb::arg("tt") = nullptr,
        nb::arg("num_threads") = 0, nb::arg("killers") = nullptr,
        nb::arg("history") = nullptr, nb::arg("counters") = nullptr,
        "Built-in evaluator best-move search using BitboardState.");
  m.def("find_best_move_builtin_state", find_best_move_builtin_binding,
        nb::arg("state"), nb::arg("depth"), nb::arg("tt") = nullptr,
        nb::arg("num_threads") = 0, nb::arg("killers") = nullptr,
        nb::arg("history") = nullptr, nb::arg("counters") = nullptr,
        "Alias of find_best_move_builtin for backwards compatibility.");

  auto get_best_move_uci_binding =
      [resolve_eval](const BitboardState &state, int depth, nb::object evaluate,
                     TranspositionTable *tt, int num_threads,
                     KillerMoves *killers, HistoryTable *history,
                     CounterMoveTable *counters) {
        BitboardState copy = state;
        auto eval_fn = resolve_eval(evaluate);
        return get_best_move_uci(copy, depth, eval_fn, tt, num_threads, killers,
                                 history, counters);
      };
  m.def("get_best_move_uci", get_best_move_uci_binding, nb::arg("state"),
        nb::arg("depth"), nb::arg("evaluate") = nb::none(),
        nb::arg("tt") = nullptr, nb::arg("num_threads") = 0,
        nb::arg("killers") = nullptr, nb::arg("history") = nullptr,
        nb::arg("counters") = nullptr,
        "Get the best move in UCI format for a BitboardState.");
  m.def("get_best_move_uci_state", get_best_move_uci_binding, nb::arg("state"),
        nb::arg("depth"), nb::arg("evaluate") = nb::none(),
        nb::arg("tt") = nullptr, nb::arg("num_threads") = 0,
        nb::arg("killers") = nullptr, nb::arg("history") = nullptr,
        nb::arg("counters") = nullptr,
        "Alias of get_best_move_uci for backwards compatibility.");

  auto get_best_move_uci_builtin_binding =
      [](const BitboardState &state, int depth, TranspositionTable *tt,
         int num_threads, KillerMoves *killers, HistoryTable *history,
         CounterMoveTable *counters) {
        BitboardState copy = state;
        return get_best_move_uci_builtin(copy, depth, tt, num_threads, killers,
                                         history, counters);
      };
  m.def("get_best_move_uci_builtin", get_best_move_uci_builtin_binding,
        nb::arg("state"), nb::arg("depth"), nb::arg("tt") = nullptr,
        nb::arg("num_threads") = 0, nb::arg("killers") = nullptr,
        nb::arg("history") = nullptr, nb::arg("counters") = nullptr,
        "Built-in evaluator version of get_best_move_uci.");
  m.def("get_best_move_uci_builtin_state", get_best_move_uci_builtin_binding,
        nb::arg("state"), nb::arg("depth"), nb::arg("tt") = nullptr,
        nb::arg("num_threads") = 0, nb::arg("killers") = nullptr,
        nb::arg("history") = nullptr, nb::arg("counters") = nullptr,
        "Alias of get_best_move_uci_builtin for backwards compatibility.");

  // Multi-PV Search
  nb::class_<PVLine>(m, "PVLine")
      .def_rw("uci_move", &PVLine::uci_move, "Best move in UCI format")
      .def_rw("score", &PVLine::score, "Evaluation score")
      .def_rw("depth", &PVLine::depth, "Search depth")
      .def_rw("pv", &PVLine::pv, "Principal variation");

  auto multi_pv_binding =
      [resolve_eval](const BitboardState &state, int depth, int num_lines,
                     nb::object evaluate, TranspositionTable *tt,
                     int num_threads, KillerMoves *killers,
                     HistoryTable *history, CounterMoveTable *counters) {
        BitboardState copy = state;
        auto eval_fn = resolve_eval(evaluate);
        return multi_pv_search(copy, depth, num_lines, eval_fn, tt, num_threads,
                               killers, history, counters);
      };
  m.def("multi_pv_search", multi_pv_binding, nb::arg("state"), nb::arg("depth"),
        nb::arg("num_lines"), nb::arg("evaluate") = nb::none(),
        nb::arg("tt") = nullptr, nb::arg("num_threads") = 0,
        nb::arg("killers") = nullptr, nb::arg("history") = nullptr,
        nb::arg("counters") = nullptr,
        "Search for multiple principal variations from a BitboardState.");
  m.def("multi_pv_search_state", multi_pv_binding, nb::arg("state"),
        nb::arg("depth"), nb::arg("num_lines"),
        nb::arg("evaluate") = nb::none(), nb::arg("tt") = nullptr,
        nb::arg("num_threads") = 0, nb::arg("killers") = nullptr,
        nb::arg("history") = nullptr, nb::arg("counters") = nullptr,
        "Alias of multi_pv_search for backwards compatibility.");

  m.def(
      "set_max_search_depth", &set_max_search_depth, nb::arg("depth") = 5,
      "Set the global maximum search depth (default = 5). "
      "Pass values > 5 to allow deeper searches; values <=0 reset to default.");
}
