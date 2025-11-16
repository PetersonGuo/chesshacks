#include "nnue_evaluator.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

// Piece type constants (matching Virgo-style)
enum PieceType {
  PAWN_TYPE = 0,
  KNIGHT_TYPE = 1,
  BISHOP_TYPE = 2,
  ROOK_TYPE = 3,
  QUEEN_TYPE = 4,
  KING_TYPE = 5
};

// Color constants (matching Virgo-style)
enum ColorType { BLACK_COLOR = 0, WHITE_COLOR = 1 };

NNUEEvaluator::NNUEEvaluator()
    : model_loaded_(false), hidden_size_(0), hidden2_size_(0), hidden3_size_(0),
      ft_weight_(nullptr), ft_bias_(nullptr), fc1_weight_(nullptr),
      fc1_bias_(nullptr), fc2_weight_(nullptr), fc2_bias_(nullptr),
      fc3_weight_(nullptr), fc3_bias_(nullptr) {}

NNUEEvaluator::~NNUEEvaluator() { free_memory(); }

void NNUEEvaluator::free_memory() {
  delete[] ft_weight_;
  delete[] ft_bias_;
  delete[] fc1_weight_;
  delete[] fc1_bias_;
  delete[] fc2_weight_;
  delete[] fc2_bias_;
  delete[] fc3_weight_;
  delete[] fc3_bias_;

  ft_weight_ = nullptr;
  ft_bias_ = nullptr;
  fc1_weight_ = nullptr;
  fc1_bias_ = nullptr;
  fc2_weight_ = nullptr;
  fc2_bias_ = nullptr;
  fc3_weight_ = nullptr;
  fc3_bias_ = nullptr;
}

void NNUEEvaluator::allocate_memory() {
  // Free any existing memory
  free_memory();

  // Allocate new memory
  ft_weight_ = new float[hidden_size_ * INPUT_SIZE];
  ft_bias_ = new float[hidden_size_];

  fc1_weight_ = new float[hidden2_size_ * hidden_size_];
  fc1_bias_ = new float[hidden2_size_];

  fc2_weight_ = new float[hidden3_size_ * hidden2_size_];
  fc2_bias_ = new float[hidden3_size_];

  fc3_weight_ = new float[1 * hidden3_size_];
  fc3_bias_ = new float[1];
}

bool NNUEEvaluator::load_model(const std::string &model_path) {
  std::ifstream file(model_path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open NNUE model file: " << model_path
              << std::endl;
    return false;
  }

  // Read and verify magic number
  char magic[4];
  file.read(magic, 4);
  if (std::strncmp(magic, "NNUE", 4) != 0) {
    std::cerr << "Error: Invalid NNUE model file (bad magic number)"
              << std::endl;
    return false;
  }

  // Read version
  uint32_t version;
  file.read(reinterpret_cast<char *>(&version), sizeof(uint32_t));
  if (version != 1) {
    std::cerr << "Error: Unsupported NNUE model version: " << version
              << std::endl;
    return false;
  }

  // Read architecture
  uint32_t hidden_size, hidden2_size, hidden3_size;
  file.read(reinterpret_cast<char *>(&hidden_size), sizeof(uint32_t));
  file.read(reinterpret_cast<char *>(&hidden2_size), sizeof(uint32_t));
  file.read(reinterpret_cast<char *>(&hidden3_size), sizeof(uint32_t));

  hidden_size_ = static_cast<int>(hidden_size);
  hidden2_size_ = static_cast<int>(hidden2_size);
  hidden3_size_ = static_cast<int>(hidden3_size);

  std::cout << "Loading NNUE model from: " << model_path << std::endl;
  std::cout << "  Architecture: 768 → " << hidden_size_ << " → "
            << hidden2_size_ << " → " << hidden3_size_ << " → 1" << std::endl;

  // Allocate memory for weights
  allocate_memory();

  // Helper lambda to read layer weights
  auto read_layer = [&](const char *name, float *weight, int weight_size,
                        float *bias, int bias_size) {
    file.read(reinterpret_cast<char *>(weight), weight_size * sizeof(float));
    file.read(reinterpret_cast<char *>(bias), bias_size * sizeof(float));

    if (!file.good()) {
      std::cerr << "Error reading layer: " << name << std::endl;
      return false;
    }
    return true;
  };

  // Read all layers
  if (!read_layer("ft", ft_weight_, hidden_size_ * INPUT_SIZE, ft_bias_,
                  hidden_size_)) {
    free_memory();
    return false;
  }

  if (!read_layer("fc1", fc1_weight_, hidden2_size_ * hidden_size_, fc1_bias_,
                  hidden2_size_)) {
    free_memory();
    return false;
  }

  if (!read_layer("fc2", fc2_weight_, hidden3_size_ * hidden2_size_, fc2_bias_,
                  hidden3_size_)) {
    free_memory();
    return false;
  }

  if (!read_layer("fc3", fc3_weight_, 1 * hidden3_size_, fc3_bias_, 1)) {
    free_memory();
    return false;
  }

  model_loaded_ = true;
  std::cout << "✓ NNUE model loaded successfully" << std::endl;

  return true;
}

void NNUEEvaluator::board_to_features(const bitboard::BitboardState &board,
                                      float *features, bool perspective) const {
  // Initialize all features to 0
  std::fill(features, features + INPUT_SIZE, 0.0f);

  // Map piece values to piece types
  auto get_piece_type = [](Piece piece) -> int {
    int abs_piece = (piece < 0) ? -piece : piece;
    switch (abs_piece) {
    case 1:
      return PAWN_TYPE;
    case 2:
      return KNIGHT_TYPE;
    case 3:
      return BISHOP_TYPE;
    case 4:
      return ROOK_TYPE;
    case 5:
      return QUEEN_TYPE;
    case 6:
      return KING_TYPE;
    default:
      return -1;
    }
  };

  // Square mirror table for black's perspective (180 degree rotation)
  static const int mirror[64] = {
      56, 57, 58, 59, 60, 61, 62, 63, 48, 49, 50, 51, 52, 53, 54, 55,
      40, 41, 42, 43, 44, 45, 46, 47, 32, 33, 34, 35, 36, 37, 38, 39,
      24, 25, 26, 27, 28, 29, 30, 31, 16, 17, 18, 19, 20, 21, 22, 23,
      8,  9,  10, 11, 12, 13, 14, 15, 0,  1,  2,  3,  4,  5,  6,  7};

  // Iterate through all squares and set features
  for (int square = 0; square < 64; square++) {
    Piece piece = board.get_piece_at(square);
    if (piece == EMPTY) {
      continue;
    }

    bool is_white = (piece > 0);
    int piece_type = get_piece_type(piece);
    if (piece_type < 0) {
      continue;
    }

    // Determine color index and square based on perspective
    int color_idx, feature_square;

    if (perspective) { // White's perspective
      color_idx = is_white ? WHITE_COLOR : BLACK_COLOR;
      feature_square = square;
    } else { // Black's perspective (flip board and colors)
      color_idx = is_white ? BLACK_COLOR : WHITE_COLOR;
      feature_square = mirror[square];
    }

    // Calculate feature index: [color][piece_type][square]
    // Layout: first 384 features for color 0, next 384 for color 1
    // Within each color: 64 features per piece type
    int feature_idx = color_idx * (6 * 64) + piece_type * 64 + feature_square;
    features[feature_idx] = 1.0f;
  }
}

void NNUEEvaluator::dense_layer(const float *input, int input_size,
                                const float *weight, const float *bias,
                                float *output, int output_size) const {
  // Initialize output with bias
  std::copy(bias, bias + output_size, output);

  // Matrix multiplication: output = input * weight^T + bias
  // weight is stored in row-major format: [output_size, input_size]
  for (int i = 0; i < output_size; i++) {
    for (int j = 0; j < input_size; j++) {
      output[i] += input[j] * weight[i * input_size + j];
    }
  }
}

float NNUEEvaluator::forward(const float *input) const {
  // Allocate temporary buffers for intermediate activations
  float *hidden1 = new float[hidden_size_];
  float *hidden2 = new float[hidden2_size_];
  float *hidden3 = new float[hidden3_size_];
  float output;

  // Feature transformer: 768 → hidden_size
  dense_layer(input, INPUT_SIZE, ft_weight_, ft_bias_, hidden1, hidden_size_);
  clipped_relu(hidden1, hidden_size_);

  // Hidden layer 1: hidden_size → hidden2_size
  dense_layer(hidden1, hidden_size_, fc1_weight_, fc1_bias_, hidden2,
              hidden2_size_);
  clipped_relu(hidden2, hidden2_size_);

  // Hidden layer 2: hidden2_size → hidden3_size
  dense_layer(hidden2, hidden2_size_, fc2_weight_, fc2_bias_, hidden3,
              hidden3_size_);
  clipped_relu(hidden3, hidden3_size_);

  // Output layer: hidden3_size → 1 (linear, no activation)
  dense_layer(hidden3, hidden3_size_, fc3_weight_, fc3_bias_, &output, 1);

  // Clean up
  delete[] hidden1;
  delete[] hidden2;
  delete[] hidden3;

  return output;
}

int NNUEEvaluator::evaluate(const bitboard::BitboardState &board) const {
  if (!model_loaded_) {
    std::cerr << "Error: NNUE model not loaded" << std::endl;
    return 0;
  }

  // Extract features from side-to-move perspective
  float features[INPUT_SIZE];
  bool perspective = board.white_to_move();
  board_to_features(board, features, perspective);

  // Run forward pass
  float raw_score = forward(features);

  // Convert from normalized score to centipawns
  // The model outputs normalized scores, we need to convert back
  // Assuming the model was trained with scores normalized to approximately [-3,
  // 3] and the original scores were in centipawns divided by 100 So: centipawns
  // = raw_score * 100
  float score_cp = raw_score * 100.0f;

  // Apply sign based on side to move
  // Model is trained from white's perspective, so if black to move, negate
  if (!perspective) {
    score_cp = -score_cp;
  }

  return static_cast<int>(std::round(score_cp));
}

int NNUEEvaluator::evaluate(const std::string &fen) const {
  bitboard::BitboardState board(fen);
  return evaluate(board);
}
