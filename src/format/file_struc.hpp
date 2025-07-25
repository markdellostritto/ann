#ifndef FILE_STRUC_HPP
#define FILE_STRUC_HPP

// ann - structure
#include "struc/structure_fwd.hpp"
// ann - file i/o
#include "format/format.hpp"

Structure& read_struc(const char* file, FILE_FORMAT::type format, const AtomType& atomT, Structure& struc);
const Structure& write_struc(const char* file, FILE_FORMAT::type format, const AtomType& atomT, const Structure& struc);

#endif