// metal/src/engine/cvec.cpp - the control vector on Metal (--control-vector-scaled; the experimental speed projection).
//
// The vector is kept here on the host, as cvec_upload takes it (every layer's direction, every layer's scale).  The Window
// copies the directions to the device when it first runs with a vector (Window::cv_alloc), and the steering itself is the
// kernels gr_norm_cv (gr.metal) and gr_norm_h_cv (prompt.metal): each folds the vector of the layer that made the pending
// write into the next hyper-connection read.  One Metal device, so replicating the tables is nothing to do.
//
// The request switch (cvec_set_enabled) is host state here and a device word in the Window (cv_flag, before each run), so
// a request switches the vector without a new graph.  A loaded vector starts on, as on the other engines.
#include "strata/kernels/cvec.hpp"

#include <algorithm>
#include <vector>

namespace strata::kernels {
namespace {
std::vector<float> g_dir, g_s;   // the loaded vector: n_layers x n_embd directions, n_layers scales
Cvec g_cvec;                     // the view the engine reads: dir and s point at the vectors above
bool g_on = true;
}  // namespace

const Cvec& cvec() { return g_cvec; }

bool cvec_upload(const std::vector<float>& dir, const std::vector<float>& s, int mode, int first, int last,
                 int64_t n_embd, int64_t hc, std::string& err) {
    const int64_t L = (int64_t) s.size();
    if (L < 1 || n_embd < 1 || (int64_t) dir.size() != L * n_embd) {
        err = "control vector: the directions do not match the layers and n_embd";
        return false;
    }
    if (mode != 0 && mode != 1) {
        err = "control vector: the mode is neither project (0) nor add (1)";
        return false;
    }
    g_dir = dir;
    g_s = s;
    g_cvec = Cvec{};
    g_cvec.dir = g_dir.data();
    g_cvec.s = g_s.data();
    g_cvec.on = nullptr;   // the request switch: the Window's device word (cv_flag)
    g_cvec.mode = mode;
    g_cvec.first = first;
    g_cvec.last = last;
    g_cvec.n_embd = n_embd;
    g_cvec.hc = hc;
    g_cvec.steered.assign((size_t) L, false);
    for (int64_t l = std::max<int64_t>(first, 0); l <= last && l < L; ++l) g_cvec.steered[(size_t) l] = g_s[(size_t) l] != 0.0f;
    g_on = true;
    return true;
}

bool cvec_replicate(std::string& err) {
    (void) err;
    return true;   // a single device: the Window's copy of the directions is its device table
}

void cvec_set_enabled(bool on) { g_on = on; }
bool cvec_enabled() { return g_on; }

}  // namespace strata::kernels
