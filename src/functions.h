#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <string>
#include <vector>
#include <climits>

const int MIN = INT_MIN;
const int MAX = INT_MAX;

int alpha_beta(int node, int depth, int alpha, int beta, bool maximizingPlayer);

#endif // FUNCTIONS_H
