#define VIRGO_IMPLEMENTATION
#include "../third_party/virgo/virgo.h"
#undef VIRGO_IMPLEMENTATION

#include <vector>

template void virgo::get_legal_moves<virgo::WHITE>(
    virgo::Chessboard &, std::vector<uint16_t> &);
template void virgo::get_legal_moves<virgo::BLACK>(
    virgo::Chessboard &, std::vector<uint16_t> &);

template void virgo::make_move<virgo::WHITE>(uint16_t, virgo::Chessboard &);
template void virgo::make_move<virgo::BLACK>(uint16_t, virgo::Chessboard &);

template void virgo::take_move<virgo::WHITE>(virgo::Chessboard &);
template void virgo::take_move<virgo::BLACK>(virgo::Chessboard &);

