#include "bitboard/bitboard_state.h"
#include "core/evaluation.h"
#include "core/nnue_evaluator.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

using bitboard::BitboardState;

namespace {

namespace fs = std::filesystem;

class ScopedNNUEGuard {
public:
  ScopedNNUEGuard() = default;
  ~ScopedNNUEGuard() { g_nnue_evaluator.reset(); }
};

fs::path GenerateModel() {
#ifndef CHESSHACKS_SOURCE_DIR
#error "CHESSHACKS_SOURCE_DIR must be defined to locate helper scripts."
#endif
  const fs::path script = fs::path(CHESSHACKS_SOURCE_DIR) / "tests" /
                          "scripts" / "generate_dummy_nnue.py";
  if (!fs::exists(script)) {
    throw std::runtime_error("Missing NNUE generator script: " +
                             script.string());
  }

  fs::path output_dir = fs::temp_directory_path() / "chesshacks_nnue_tests";
  fs::create_directories(output_dir);
  fs::path model_path = output_dir / "dummy_nnue.pt";

  std::string command =
      "python3 \"" + script.string() + "\" \"" + model_path.string() + "\"";
  int rc = std::system(command.c_str());
  if (rc != 0) {
    throw std::runtime_error(
        "Failed to run Python NNUE generator. Ensure torch is installed.");
  }

  fs::path stats_path = output_dir / "nnue_stats.json";
  if (!fs::exists(model_path) || !fs::exists(stats_path)) {
    throw std::runtime_error("NNUE generator did not produce expected files.");
  }

  return model_path;
}

const fs::path &ModelPath() {
  static const fs::path cached = GenerateModel();
  return cached;
}

} // namespace

TEST(NNUEEvaluator, LoadsAndEvaluatesBitboard) {
  fs::path model;
  try {
    model = ModelPath();
  } catch (const std::exception &ex) {
    GTEST_SKIP() << "NNUE tests unavailable: " << ex.what();
  }

  NNUEEvaluator evaluator;
  ASSERT_TRUE(evaluator.load_model(model.string()));
  EXPECT_TRUE(evaluator.is_loaded());

  BitboardState state(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  int score = evaluator.evaluate(state);
  EXPECT_TRUE(std::isfinite(static_cast<double>(score)));
}

TEST(NNUEEvaluator, EvaluateFromFenMatchesBitboard) {
  fs::path model;
  try {
    model = ModelPath();
  } catch (const std::exception &ex) {
    GTEST_SKIP() << "NNUE tests unavailable: " << ex.what();
  }

  NNUEEvaluator evaluator;
  ASSERT_TRUE(evaluator.load_model(model.string()));

  const std::string fen =
      "r1bq1rk1/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4";
  BitboardState state(fen);
  int via_state = evaluator.evaluate(state);
  int via_fen = evaluator.evaluate(fen);
  EXPECT_EQ(via_state, via_fen);
}

TEST(NNUEPipeline, InitFunctionRoutesEvaluate) {
  fs::path model;
  try {
    model = ModelPath();
  } catch (const std::exception &ex) {
    GTEST_SKIP() << "NNUE tests unavailable: " << ex.what();
  }

  {
    ScopedNNUEGuard guard;
    ASSERT_TRUE(init_nnue(model.string()));
    ASSERT_TRUE(is_nnue_loaded());

    BitboardState state("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/2NP1N2/PPP2PPP/"
                        "R1BQK2R w KQkq - 4 5");

    int nnue_score = evaluate_nnue(state);
    int routed_score = evaluate(state);
    EXPECT_EQ(nnue_score, routed_score);
  }

  BitboardState state2("rnbq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP2BPPP/"
                       "R1BQ1RK1 w - - 0 9");
  int material_only = evaluate(state2);
  EXPECT_EQ(material_only, evaluate_material(state2));
}
