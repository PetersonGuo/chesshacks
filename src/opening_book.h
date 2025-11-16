#ifndef OPENING_BOOK_H
#define OPENING_BOOK_H

#include <cstdint>
#include <string>
#include <vector>

struct BookEntry {
  uint64_t key;    // Polyglot hash of position
  uint16_t move;   // Move in Polyglot format
  uint16_t weight; // Weight/frequency
  uint32_t learn;  // Learning data (unused)
};

struct BookMove {
  std::string uci_move;
  int weight;
};

class OpeningBook {
public:
  OpeningBook() : loaded(false) {}

  bool load(const std::string &book_path);
  bool is_loaded() const { return loaded; }
  std::vector<BookMove> probe(const std::string &fen);
  std::string probe_best(const std::string &fen);
  std::string
  probe_weighted(const std::string &fen); // Random weighted selection
  void clear();

private:
  std::vector<BookEntry> entries;
  bool loaded;

  uint64_t polyglot_hash(const std::string &fen);
  std::string decode_move(uint16_t move);
};

#endif // OPENING_BOOK_H
