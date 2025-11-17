#include "bitboard/bitboard_state.h"
#include "core/utils.h"

#include <gtest/gtest.h>

#include <array>
#include <sstream>
#include <string>
#include <vector>

using bitboard::BitboardState;

TEST(PGNConversion, ToFenMatchesExpected) {
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"1. e4 e5 2. Nf3 Nc6 3. Bb5 a6",
       "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R"},
      {"1. d4 Nf6 2. c4 g6 3. Nc3 Bg7",
       "rnbqk2r/ppppppbp/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR"}};

  for (const auto &[pgn, expected] : cases) {
    std::string fen = pgn_to_fen(pgn);
    std::istringstream iss(fen);
    std::string piece_placement;
    iss >> piece_placement;
    EXPECT_EQ(piece_placement, expected) << "PGN: " << pgn;
  }
}

TEST(PGNConversion, BitboardMatchesFenConversion) {
  const std::string pgn = "1. e4 Nh6 2. d4 Rg8 3. Nf3 Rh8 4. Ne5";
  const BitboardState state = pgn_to_bitboard(pgn);
  const std::string fen = pgn_to_fen(pgn);
  EXPECT_EQ(state.to_fen(), fen);
}

TEST(PGNConversion, BitboardFenRoundTrip) {
  const std::string pgn = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6";
  BitboardState state = pgn_to_bitboard(pgn);
  const std::string fen = state.to_fen();
  BitboardState rebuilt(fen);
  EXPECT_EQ(rebuilt.to_fen(), fen);
}

TEST(PGNConversion, HandlesComplexGamesFullFen) {
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 Nbd7",
       "r1bqk2r/pppnbppp/4pn2/3p2B1/2PP4/2N1P3/PP3PPP/R2QKBNR w KQkq - 1 6"},
      {"1. c4 e5 2. Nc3 Nc6 3. g3 g6 4. Bg2 Bg7 5. d3 d6 6. e4 Nge7",
       "r1bqk2r/ppp1npbp/2np2p1/4p3/2P1P3/2NP2P1/PP3PBP/R1BQK1NR w KQkq - 1 7"},
      {"1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6",
       "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6"},
      {"1. d4 Nf6 2. c4 g6 3. Nc3 d5 4. cxd5 Nxd5 5. e4 Nxc3 6. bxc3 Bg7",
       "rnbqk2r/ppp1ppbp/6p1/8/3PP3/2P5/P4PPP/R1BQKBNR w KQkq - 1 7"},
      {"1. e4 e6 2. d4 d5 3. Nc3 Bb4 4. exd5 exd5 5. Bd3 Nf6",
       "rnbqk2r/ppp2ppp/5n2/3p4/1b1P4/2NB4/PPP2PPP/R1BQK1NR w KQkq - 2 6"},
      {"1. c4 Nf6 2. Nc3 e6 3. g3 d5 4. cxd5 exd5 5. d4 c6 6. Bg2 Bd6",
       "rnbqk2r/pp3ppp/2pb1n2/3p4/3P4/2N3P1/PP2PPBP/R1BQK1NR w KQkq - 2 7"}};

  for (const auto &[pgn, expected_fen] : cases) {
    try {
      const std::string fen = pgn_to_fen(pgn);
      EXPECT_EQ(fen, expected_fen) << "PGN: " << pgn;
      EXPECT_EQ(pgn_to_bitboard(pgn).to_fen(), expected_fen);
    } catch (const std::exception &ex) {
      FAIL() << "PGN parse failed for \"" << pgn << "\": " << ex.what();
    }
  }
}

TEST(PGNConversion, ParsesHeadersAndResultTokens) {
  const std::string pgn = R"(
[Event "Casual"]
[Site "Somewhere"]
[Round "-"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0
)";
  const std::string expected =
      "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4";
  EXPECT_EQ(pgn_to_fen(pgn), expected);
}

TEST(PGNConversion, InvalidMovesAreIgnored) {
  const std::string baseline = "1. e4 e5 2. Nf3 Nc6";
  const std::string noisy = "1. e4 e5 2. Nf3 Nc6 3. Qh9?? Qa8?? 4. Bb5??";
  EXPECT_EQ(pgn_to_fen(noisy), pgn_to_fen(baseline));
  EXPECT_EQ(pgn_to_bitboard(noisy).to_fen(), pgn_to_fen(baseline));
}

TEST(PGNConversion, HandlesAmbiguousSANSequences) {
  const std::string pgn = "1. Nf3 Nf6 2. Nc3 Nc6 3. Nb5 Nb4 4. Nxa7 Nxa2 5. "
                          "Nxc8 Nxc1 6. Nd6+ exd6 7. Rxa8";
  BitboardState state = pgn_to_bitboard(pgn);
  EXPECT_EQ(state.to_fen(), pgn_to_fen(pgn));
}
