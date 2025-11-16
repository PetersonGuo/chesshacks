#ifndef NNUE_EVALUATOR_H
#define NNUE_EVALUATOR_H

#include "chess_board.h"
#include <string>
#include <vector>
#include <memory>

/**
 * NNUE (Efficiently Updatable Neural Network) Evaluator
 *
 * This class loads and runs inference on a trained NNUE model for chess position evaluation.
 * The model uses Virgo-style bitboard representation (768 features) and a simple feedforward
 * architecture optimized for fast inference.
 *
 * Architecture: 768 → 256 → 32 → 32 → 1
 * - Input: Virgo-style bitboards (2 colors × 6 pieces × 64 squares)
 * - Hidden layers use ClippedReLU activation
 * - Output is linear (no activation) for full eval range
 */
class NNUEEvaluator {
public:
    /**
     * Constructor - creates an uninitialized evaluator
     * Call load_model() to load weights from a file
     */
    NNUEEvaluator();

    /**
     * Destructor - frees allocated memory
     */
    ~NNUEEvaluator();

    /**
     * Load NNUE model from binary file
     *
     * @param model_path Path to binary model file (.bin)
     * @return true if loaded successfully, false otherwise
     */
    bool load_model(const std::string& model_path);

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
    int evaluate(const ChessBoard& board) const;

    /**
     * Evaluate a chess position from FEN string
     *
     * @param fen FEN string of position to evaluate
     * @return Evaluation score in centipawns (positive = white advantage)
     */
    int evaluate(const std::string& fen) const;

    /**
     * Get model architecture info
     */
    int get_hidden_size() const { return hidden_size_; }
    int get_hidden2_size() const { return hidden2_size_; }
    int get_hidden3_size() const { return hidden3_size_; }

private:
    // Model loaded flag
    bool model_loaded_;

    // Model architecture
    static constexpr int INPUT_SIZE = 768;  // 2 colors × 6 pieces × 64 squares
    int hidden_size_;
    int hidden2_size_;
    int hidden3_size_;

    // Layer weights and biases (stored as 1D arrays for efficiency)
    // Feature transformer (ft): 768 → hidden_size
    float* ft_weight_;      // [hidden_size, 768]
    float* ft_bias_;        // [hidden_size]

    // Hidden layer 1 (fc1): hidden_size → hidden2_size
    float* fc1_weight_;     // [hidden2_size, hidden_size]
    float* fc1_bias_;       // [hidden2_size]

    // Hidden layer 2 (fc2): hidden2_size → hidden3_size
    float* fc2_weight_;     // [hidden3_size, hidden2_size]
    float* fc2_bias_;       // [hidden3_size]

    // Output layer (fc3): hidden3_size → 1
    float* fc3_weight_;     // [1, hidden3_size]
    float* fc3_bias_;       // [1]

    /**
     * Free all allocated memory
     */
    void free_memory();

    /**
     * Allocate memory for model weights based on architecture
     */
    void allocate_memory();

    /**
     * Convert ChessBoard to Virgo-style bitboard features
     *
     * @param board Chess board to convert
     * @param features Output array of 768 features (must be pre-allocated)
     * @param perspective Side to move (true = white, false = black)
     */
    void board_to_features(const ChessBoard& board, float* features, bool perspective) const;

    /**
     * Forward pass through the network
     *
     * @param input Input features (768 elements)
     * @return Raw network output (normalized score)
     */
    float forward(const float* input) const;

    /**
     * ClippedReLU activation: min(max(x, 0), 1)
     * Applied in-place to array
     *
     * @param x Array to apply activation to
     * @param size Size of array
     */
    inline void clipped_relu(float* x, int size) const {
        for (int i = 0; i < size; i++) {
            x[i] = std::min(std::max(x[i], 0.0f), 1.0f);
        }
    }

    /**
     * Dense layer: output = input * weight^T + bias
     *
     * @param input Input vector
     * @param input_size Size of input
     * @param weight Weight matrix (row-major, [output_size, input_size])
     * @param bias Bias vector
     * @param output Output vector (must be pre-allocated)
     * @param output_size Size of output
     */
    void dense_layer(const float* input, int input_size,
                    const float* weight, const float* bias,
                    float* output, int output_size) const;
};

#endif // NNUE_EVALUATOR_H
