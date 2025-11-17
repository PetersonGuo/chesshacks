#include "bitboard/bitboard_state.h"
#include "core/evaluation.h"
#include "core/search.h"

#include <gtest/gtest.h>

using bitboard::BitboardState;

namespace {

BitboardState MakeState(const std::string &fen) { return BitboardState(fen); }

} // namespace

TEST(BatchEvaluation, MatchesSequential) {
  const std::vector<std::string> fens = {
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
      "rnbq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 b - - 0 9"};

  std::vector<BitboardState> states;
  states.reserve(fens.size());
  for (const auto &fen : fens) {
    states.push_back(MakeState(fen));
  }

  std::vector<int> sequential;
  sequential.reserve(states.size());
  for (const auto &state : states) {
    sequential.push_back(evaluate(state));
  }

  std::vector<int> parallel = batch_evaluate_mt(states, /*num_threads=*/4);
  ASSERT_EQ(sequential.size(), parallel.size());
  EXPECT_EQ(sequential, parallel);
}

TEST(AlphaBeta, ParallelMatchesSequentialWithinTolerance) {
  const std::vector<std::string> fens = {
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "r2qkb1r/ppp2ppp/2n5/3np1B1/2BPP1b1/2P2N2/PP3PPP/RN1Q1RK1 w kq - 0 9",
      "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"};

  for (const auto &fen : fens) {
    BitboardState state = MakeState(fen);

    TranspositionTable tt_seq;
    KillerMoves killers_seq;
    HistoryTable history_seq;
    CounterMoveTable counters_seq;

    int seq_score = alpha_beta_builtin(state, /*depth=*/4, MIN, MAX,
                                       /*maximizingPlayer=*/true, &tt_seq,
                                       /*num_threads=*/1, &killers_seq,
                                       &history_seq, &counters_seq);

    TranspositionTable tt_par;
    KillerMoves killers_par;
    HistoryTable history_par;
    CounterMoveTable counters_par;

    int par_score = alpha_beta_builtin(state, /*depth=*/4, MIN, MAX,
                                       /*maximizingPlayer=*/true, &tt_par,
                                       /*num_threads=*/4, &killers_par,
                                       &history_par, &counters_par);

    EXPECT_NEAR(seq_score, par_score, 150) << "FEN: " << fen;
  }
}

TEST(BatchEvaluation, LargeBatchMatchesSequential) {
  const std::vector<std::string> seeds = {
      "rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 2",
      "r3k2r/pppb1ppp/2npbn2/4p3/2B1P3/2NP1N2/PPPQ1PPP/R3K2R w KQkq - 0 9",
      "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 b - - 3 9"};

  std::vector<BitboardState> states;
  for (int i = 0; i < 128; ++i) {
    states.push_back(MakeState(seeds[i % seeds.size()]));
  }

  std::vector<int> sequential = batch_evaluate_mt(states, /*num_threads=*/1);
  std::vector<int> threaded = batch_evaluate_mt(states, /*num_threads=*/0);
  EXPECT_EQ(sequential, threaded);
}

TEST(BatchEvaluation, HandlesDuplicateStates) {
  BitboardState repeated = MakeState("8/8/8/8/8/8/8/4K3 w - - 0 1");
  std::vector<BitboardState> states(32, repeated);
  std::vector<int> scores = batch_evaluate_mt(states, /*num_threads=*/8);
  ASSERT_EQ(scores.size(), states.size());
  for (int score : scores) {
    EXPECT_EQ(score, evaluate_material(repeated));
  }
}

TEST(BatchEvaluation, ReturnsEmptyForZeroStates) {
  std::vector<BitboardState> empty;
  std::vector<int> scores = batch_evaluate_mt(empty, /*num_threads=*/8);
  EXPECT_TRUE(scores.empty());
}

TEST(Evaluation, MaterialReflectsSimpleAdvantages) {
  BitboardState white_adv = MakeState("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1");
  BitboardState black_adv = MakeState("4k3/4p3/8/8/8/8/8/4K3 w - - 0 1");

  EXPECT_EQ(evaluate_material(white_adv), 100);
  EXPECT_EQ(evaluate_material(black_adv), -100);
}

TEST(Evaluation, SymmetricBoardsNegateScores) {
  BitboardState white_control = MakeState("4k3/8/8/3P4/8/8/8/4K3 w - - 0 1");
  BitboardState black_control = MakeState("4k3/8/8/8/3p4/8/8/4K3 w - - 0 1");
  EXPECT_EQ(evaluate_material(white_control),
            -evaluate_material(black_control));
}

TEST(AlphaBeta, DetectsStalemateAsDraw) {
  BitboardState stalemate = MakeState("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
  TranspositionTable tt;
  KillerMoves killers;
  HistoryTable history;
  CounterMoveTable counters;
  int score =
      alpha_beta_builtin(stalemate, /*depth=*/3, MIN, MAX,
                         /*maximizingPlayer=*/false, &tt,
                         /*num_threads=*/1, &killers, &history, &counters);
  EXPECT_EQ(score, 0);
}
