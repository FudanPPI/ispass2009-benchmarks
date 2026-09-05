// Minimal boost::filesystem fstream shim — AES includes this header but
// only uses std::fstream types. Pull in the operations shim for path etc.
#ifndef BOOST_FILESYSTEM_FSTREAM_SHIM_HPP
#define BOOST_FILESYSTEM_FSTREAM_SHIM_HPP
#include "boost/filesystem/operations.hpp"
#endif /* BOOST_FILESYSTEM_FSTREAM_SHIM_HPP */
