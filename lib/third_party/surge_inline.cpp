#include <cstdint>

using Bitboard = uint64_t;

enum Square : int;

extern const Bitboard k1;
extern const Bitboard k2;
extern const Bitboard k4;
extern const Bitboard kf;
extern const int DEBRUIJN64[64];
extern const Bitboard MAGIC;

Square bsf(Bitboard b);

int pop_count(Bitboard x) {
  x = x - ((x >> 1) & k1);
  x = (x & k2) + ((x >> 2) & k2);
  x = (x + (x >> 4)) & k4;
  x = (x * kf) >> 56;
  return static_cast<int>(x);
}

int sparse_pop_count(Bitboard x) {
  int count = 0;
  while (x) {
    ++count;
    x &= x - 1;
  }
  return count;
}

Square pop_lsb(Bitboard *b) {
  const int lsb = bsf(*b);
  *b &= *b - 1;
  return Square(lsb);
}

Square bsf(Bitboard b) {
  return Square(DEBRUIJN64[(MAGIC * (b ^ (b - 1))) >> 58]);
}
