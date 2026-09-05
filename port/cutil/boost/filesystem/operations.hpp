// Minimal boost::filesystem shim for AES (ispass2009), which only uses
// boost::filesystem::path, exists(), file_size() and boost::intmax_t.
// Lives under port/cutil/boost/filesystem/ so that `#include
// "boost/filesystem/operations.hpp"` resolves via the cutil include path.
#ifndef BOOST_FILESYSTEM_OPERATIONS_SHIM_HPP
#define BOOST_FILESYSTEM_OPERATIONS_SHIM_HPP

#include <sys/stat.h>
#include <string>

namespace boost {

typedef long long intmax_t;

namespace filesystem {

class path {
public:
    path() {}
    path(const char* c) : s_(c) {}
    path(const std::string& s) : s_(s) {}
    const std::string& string() const { return s_; }
private:
    std::string s_;
};

inline bool exists(const path& p) {
    struct stat st;
    return ::stat(p.string().c_str(), &st) == 0;
}

inline boost::intmax_t file_size(const path& p) {
    struct stat st;
    if (::stat(p.string().c_str(), &st) != 0) return 0;
    return static_cast<boost::intmax_t>(st.st_size);
}

} // namespace filesystem
} // namespace boost

#endif /* BOOST_FILESYSTEM_OPERATIONS_SHIM_HPP */
