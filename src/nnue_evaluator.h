#ifndef NNUE_EVALUATOR_H
#define NNUE_EVALUATOR_H

#include "bitboard/bitboard_state.h"
#include <filesystem>
#include <memory>
#include <string>
#include <torch/script.h>
#include <vector>

/**
 * NNUE (Efficiently Updatable Neural Network) Evaluator
 *
 * Loads the TorchScript NNUE model produced from train/nnue_model/checkpoints/
 * best_model.pt and runs inference via libtorch.
 */
class NNUEEvaluator {
public:
  /**
   * Constructor - creates an uninitialized evaluator
   * Call load_model() to load weights from a file
   */
  NNUEEvaluator();

  /**
   * Load NNUE model from TorchScript file
   *
   * @param model_path Path to TorchScript model file (.pt)
   * @return true if loaded successfully, false otherwise
   */
  bool load_model(const std::string &model_path);

  /**
   * Check if model is loaded and ready for evaluation
   *
   * @return true if model is loaded, false otherwise
   */
  bool is_loaded() const { return model_loaded_; }

  /**
   * Evaluate a chess position using the NNUE model
   *
   * @param board Chess board to evaluate
   * @return Evaluation score in centipawns (positive = white advantage)
   */
  int evaluate(const bitboard::BitboardState &board) const;

  /**
   * Evaluate a chess position from FEN string
   *
   * @param fen FEN string of position to evaluate
   * @return Evaluation score in centipawns (positive = white advantage)
   */
  int evaluate(const std::string &fen) const;

private:
  static constexpr int INPUT_SIZE = 768;
  static constexpr int HALF_INPUT = INPUT_SIZE / 2;

  bool model_loaded_;
  mutable torch::jit::script::Module module_;
  double eval_mean_;
  double eval_std_;

  /**
   * Convert BitboardState to Virgo-style bitboard features
   *
   * @param board Board to convert
   * @param features Output array of 768 features (must be pre-allocated)
   * @param perspective Side to move (true = white, false = black)
   */
  void board_to_features(const bitboard::BitboardState &board, float *features,
                         bool perspective) const;
  bool load_stats_for(const std::string &model_path);
};

#endif // NNUE_EVALUATOR_H
