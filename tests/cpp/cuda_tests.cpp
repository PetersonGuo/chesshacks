#include "bitboard/bitboard_state.h"
#include "core/evaluation.h"
#include "cuda/cuda_eval.h"
#include "cuda/cuda_utils.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using bitboard::BitboardState;

namespace {

BitboardState MakeState(const std::string &fen) { return BitboardState(fen); }

constexpr const char *STARTING_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
constexpr const char *ADVANTAGE_FEN =
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2";
constexpr const char *TACTIC_FEN =
    "rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 2";

} // namespace

#ifndef CUDA_ENABLED
bool cuda_batch_evaluate(const std::vector<bitboard::BitboardState> &boards,
                         std::vector<int> &scores) {
  (void)boards;
  (void)scores;
  return false;
}

bool cuda_batch_mvv_lva(const std::vector<bitboard::BitboardState> &boards,
                        const std::vector<int> &from_squares,
                        const std::vector<int> &to_squares,
                        std::vector<int> &scores) {
  (void)boards;
  (void)from_squares;
  (void)to_squares;
  (void)scores;
  return false;
}

bool cuda_batch_count_pieces(const std::vector<bitboard::BitboardState> &boards,
                             std::vector<std::vector<int>> &piece_counts) {
  (void)boards;
  (void)piece_counts;
  return false;
}

bool cuda_batch_hash_positions(
    const std::vector<bitboard::BitboardState> &boards,
    std::vector<unsigned long long> &hashes) {
  (void)boards;
  (void)hashes;
  return false;
}
#endif

TEST(CUDABatchOps, EvaluateMatchesCPU) {
  if (!is_cuda_available()) {
    GTEST_SKIP() << "CUDA tests require a CUDA-enabled build/runtime";
    return;
  }

  std::vector<BitboardState> boards = {MakeState(STARTING_FEN),
                                       MakeState(ADVANTAGE_FEN)};
  std::vector<int> gpu_scores;
  ASSERT_TRUE(cuda_batch_evaluate(boards, gpu_scores));
  ASSERT_EQ(gpu_scores.size(), boards.size());

  for (size_t i = 0; i < boards.size(); ++i) {
    EXPECT_EQ(gpu_scores[i], evaluate(boards[i]));
  }
}

TEST(CUDABatchOps, PieceCountsAndHashes) {
  if (!is_cuda_available()) {
    GTEST_SKIP() << "CUDA tests require a CUDA-enabled build/runtime";
    return;
  }

  std::vector<BitboardState> boards = {MakeState(STARTING_FEN)};
  std::vector<std::vector<int>> counts;
  ASSERT_TRUE(cuda_batch_count_pieces(boards, counts));
  ASSERT_EQ(counts.size(), boards.size());
  ASSERT_EQ(counts[0].size(), 12u);

  std::vector<unsigned long long> hashes;
  boards.push_back(MakeState(ADVANTAGE_FEN));
  ASSERT_TRUE(cuda_batch_hash_positions(boards, hashes));
  ASSERT_EQ(hashes.size(), boards.size());
  EXPECT_NE(hashes[0], hashes[1]);

  std::vector<unsigned long long> hashes_second;
  ASSERT_TRUE(cuda_batch_hash_positions(boards, hashes_second));
  EXPECT_EQ(hashes, hashes_second);
  for (size_t i = 0; i < boards.size(); ++i) {
    EXPECT_EQ(hashes[i], boards[i].zobrist());
  }
}

TEST(CUDABatchOps, MVVLVAConsistentWithCPU) {
  if (!is_cuda_available()) {
    GTEST_SKIP() << "CUDA tests require a CUDA-enabled build/runtime";
    return;
  }

  BitboardState board = MakeState(TACTIC_FEN);
  std::vector<BoardMove> moves = board.generate_legal_moves();
  std::vector<int> from;
  std::vector<int> to;
  std::vector<int> cpu_scores;

  for (const auto &move : moves) {
    if (!board.is_capture(move)) {
      continue;
    }
    from.push_back(move.from);
    to.push_back(move.to);
    cpu_scores.push_back(mvv_lva_score(board, move));
  }

  if (from.empty()) {
    GTEST_SKIP() << "No captures available in the test position.";
  }

  std::vector<BitboardState> boards(from.size(), board);
  std::vector<int> gpu_scores;
  ASSERT_TRUE(cuda_batch_mvv_lva(boards, from, to, gpu_scores));
  ASSERT_EQ(gpu_scores.size(), cpu_scores.size());
  EXPECT_EQ(gpu_scores, cpu_scores);
}
