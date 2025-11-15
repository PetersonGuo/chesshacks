#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <climits>
#include <functional>
#include <string>
#include <vector>

const int MIN = INT_MIN;
const int MAX = INT_MAX;

int alpha_beta(const std::string &fen, int depth, int alpha, int beta,
               bool maximizingPlayer,
               const std::function<int(const std::string &)> &evaluate);

#endif // FUNCTIONS_H
