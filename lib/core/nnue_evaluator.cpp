#include "core/nnue_evaluator.h"
#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/serialize.h>
#include <torch/torch.h>

namespace {

constexpr int kInputSize = 768;
constexpr int kHalfInputSize = kInputSize / 2;

torch::Tensor clipped_relu(const torch::Tensor &x) {
  return torch::clamp(x, 0.0, 1.0);
}

} // namespace

TorchNNUEImpl::TorchNNUEImpl(int hidden_size, int hidden2_size,
                             int hidden3_size)
    : input_size_(kInputSize), half_input_(kHalfInputSize),
      ft_friendly_(register_module("ft_friendly",
                                   torch::nn::Linear(torch::nn::LinearOptions(
                                       half_input_, hidden_size)))),
      ft_enemy_(register_module(
          "ft_enemy", torch::nn::Linear(
                          torch::nn::LinearOptions(half_input_, hidden_size)))),
      fc1_(register_module("fc1", torch::nn::Linear(torch::nn::LinearOptions(
                                      hidden_size, hidden2_size)))),
      res1_(register_module("res1", torch::nn::Linear(torch::nn::LinearOptions(
                                        hidden2_size, hidden2_size)))),
      res2_(register_module("res2", torch::nn::Linear(torch::nn::LinearOptions(
                                        hidden2_size, hidden2_size)))),
      fc2_(register_module("fc2", torch::nn::Linear(torch::nn::LinearOptions(
                                      hidden2_size, hidden3_size)))),
      fc3_(register_module("fc3", torch::nn::Linear(torch::nn::LinearOptions(
                                      hidden3_size, 1)))) {}

torch::Tensor TorchNNUEImpl::clipped_relu(const torch::Tensor &x) const {
  return ::clipped_relu(x);
}

torch::Tensor TorchNNUEImpl::forward(const torch::Tensor &input) {
  auto friendly = input.slice(1, 0, half_input_);
  auto enemy = input.slice(1, half_input_, input_size_);

  friendly = clipped_relu(ft_friendly_(friendly));
  enemy = clipped_relu(ft_enemy_(enemy));

  auto x = friendly + enemy;
  x = clipped_relu(fc1_(x));
  auto residual = x;
  x = clipped_relu(res1_(x));
  x = res2_(x);
  x = clipped_relu(x + residual);
  x = clipped_relu(fc2_(x));
  x = fc3_(x);
  return x;
}

// Piece type constants (matching Virgo-style)
enum NNUEPieceType {
  PAWN_TYPE = 0,
  KNIGHT_TYPE = 1,
  BISHOP_TYPE = 2,
  ROOK_TYPE = 3,
  QUEEN_TYPE = 4,
  KING_TYPE = 5
};

namespace {

bool LoadStateDictFromPickle(const std::string &model_path, TorchNNUE &module) {
  std::vector<char> buffer;
  try {
    caffe2::serialize::PyTorchStreamReader reader(model_path);
    const std::string kPayload = "data.pkl";
    std::string record_name = kPayload;
    for (const auto &entry : reader.getAllRecords()) {
      if (entry.size() >= kPayload.size() &&
          entry.substr(entry.size() - kPayload.size()) == kPayload) {
        record_name = entry;
        break;
      }
    }
    auto record = reader.getRecord(record_name);
    size_t size = std::get<1>(record);
    buffer.resize(size);
    std::memcpy(buffer.data(), std::get<0>(record).get(), size);
  } catch (const c10::Error &e) {
    std::cerr << "Error: Unable to read NNUE checkpoint " << model_path << ": "
              << e.what() << std::endl;
    return false;
  }

  torch::IValue ivalue;
  try {
    ivalue = torch::pickle_load(buffer);
  } catch (const c10::Error &e) {
    std::cerr << "Error: Unable to parse NNUE checkpoint " << model_path << ": "
              << e.what() << std::endl;
    return false;
  }

  if (!ivalue.isGenericDict()) {
    std::cerr << "Error: NNUE checkpoint must be a state_dict serialized via "
                 "torch.save(model.state_dict())."
              << std::endl;
    return false;
  }

  auto dict = ivalue.toGenericDict();
  auto assign = [&](torch::OrderedDict<std::string, torch::Tensor> named) {
    for (const auto &item : named) {
      const std::string &name = item.key();
      torch::Tensor tensor = item.value();
      auto key = c10::IValue(name);
      auto it = dict.find(key);
      if (it == dict.end()) {
        std::cerr << "Error: Missing parameter '" << name
                  << "' in NNUE checkpoint." << std::endl;
        return false;
      }
      tensor.copy_(it->value().toTensor());
    }
    return true;
  };

  if (!assign(module->named_parameters(/*recurse=*/true)))
    return false;
  if (!assign(module->named_buffers(/*recurse=*/true)))
    return false;
  return true;
}

} // namespace

bool NNUEEvaluator::load_model(const std::string &model_path) {
  torch_module_ = TorchNNUE();
  if (!LoadStateDictFromPickle(model_path, torch_module_)) {
    model_loaded_ = false;
    return false;
  }
  torch_module_->to(torch::kCPU);
  torch_module_->eval();

  load_stats_for(model_path);

  model_loaded_ = true;
  std::cout << "  Loaded NNUE model (torch.save format): " << model_path
            << std::endl;
  std::cout << "  Normalization stats: mean=" << eval_mean_
            << " cp, std=" << eval_std_ << " cp" << std::endl;
  return true;
}

void NNUEEvaluator::board_to_features(const bitboard::BitboardState &board,
                                      float *features, bool perspective) const {
  std::fill(features, features + INPUT_SIZE, 0.0f);

  auto get_piece_type = [](BoardPiece piece) -> int {
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

  static const int mirror[64] = {
      56, 57, 58, 59, 60, 61, 62, 63, 48, 49, 50, 51, 52, 53, 54, 55,
      40, 41, 42, 43, 44, 45, 46, 47, 32, 33, 34, 35, 36, 37, 38, 39,
      24, 25, 26, 27, 28, 29, 30, 31, 16, 17, 18, 19, 20, 21, 22, 23,
      8,  9,  10, 11, 12, 13, 14, 15, 0,  1,  2,  3,  4,  5,  6,  7};

  const bool mover_is_white = perspective;

  for (int square = 0; square < 64; square++) {
    BoardPiece piece = board.get_piece_at(square);
    if (piece == EMPTY) {
      continue;
    }

    bool is_white = (piece > 0);
    int piece_type = get_piece_type(piece);
    if (piece_type < 0) {
      continue;
    }

    bool is_friendly = (is_white == mover_is_white);
    int color_idx = is_friendly ? 0 : 1; // First half = friendly
    int feature_square = mover_is_white ? square : mirror[square];

    int feature_idx = color_idx * (6 * 64) + piece_type * 64 + feature_square;
    features[feature_idx] = 1.0f;
  }
}

int NNUEEvaluator::evaluate(const bitboard::BitboardState &board) const {
  if (!model_loaded_) {
    std::cerr << "Warning: NNUE model not loaded; returning 0 evaluation."
              << std::endl;
    return 0;
  }

  std::array<float, INPUT_SIZE> features{};
  board_to_features(board, features.data(), board.white_to_move());

  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor input =
      torch::from_blob(features.data(), {1, INPUT_SIZE}, options).clone();

  torch::InferenceMode guard;
  torch::Tensor output = torch_module_->forward(input).reshape({});
  double normalized = output.item<double>();
  double score_cp = normalized * eval_std_ + eval_mean_;

  if (!board.white_to_move()) {
    score_cp = -score_cp;
  }

  return static_cast<int>(std::round(score_cp));
}

int NNUEEvaluator::evaluate(const std::string &fen) const {
  bitboard::BitboardState board(fen);
  return evaluate(board);
}

bool NNUEEvaluator::load_stats_for(const std::string &model_path) {
  namespace fs = std::filesystem;
  fs::path stats_path = fs::path(model_path).parent_path() / "nnue_stats.json";
  if (!fs::exists(stats_path)) {
    std::cerr << "Warning: NNUE stats file not found at " << stats_path
              << ". Using defaults (mean=0, std=1)." << std::endl;
    eval_mean_ = 0.0;
    eval_std_ = 1.0;
    return false;
  }

  std::ifstream stats(stats_path);
  if (!stats.is_open()) {
    std::cerr << "Warning: Could not open NNUE stats file " << stats_path
              << ". Using defaults (mean=0, std=1)." << std::endl;
    eval_mean_ = 0.0;
    eval_std_ = 1.0;
    return false;
  }

  std::string content((std::istreambuf_iterator<char>(stats)),
                      std::istreambuf_iterator<char>());
  auto extract = [&](const std::string &key, double &out) -> bool {
    auto pos = content.find(key);
    if (pos == std::string::npos) {
      return false;
    }
    pos = content.find(':', pos);
    if (pos == std::string::npos) {
      return false;
    }
    ++pos;
    while (pos < content.size() &&
           std::isspace(static_cast<unsigned char>(content[pos]))) {
      ++pos;
    }
    size_t end = pos;
    while (end < content.size() &&
           (std::isdigit(static_cast<unsigned char>(content[end])) ||
            content[end] == '.' || content[end] == '-' || content[end] == '+' ||
            content[end] == 'e' || content[end] == 'E')) {
      ++end;
    }
    try {
      out = std::stod(content.substr(pos, end - pos));
      return true;
    } catch (...) {
      return false;
    }
  };

  bool mean_ok = extract("eval_mean", eval_mean_);
  bool std_ok = extract("eval_std", eval_std_);
  if (!std_ok || eval_std_ <= 0.0) {
    eval_std_ = 1.0;
    std_ok = false;
  }
  if (!mean_ok) {
    eval_mean_ = 0.0;
  }

  if (!mean_ok || !std_ok) {
    std::cerr << "Warning: Invalid values in " << stats_path
              << ". Using defaults (mean=0, std=" << eval_std_ << ")."
              << std::endl;
  }

  return mean_ok && std_ok;
}
