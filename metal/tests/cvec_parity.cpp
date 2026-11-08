// metal/tests/cvec_parity.cpp - the control vector folded into the hyper-connection read (gr.metal's gr_norm_cv and
// prompt.metal's gr_norm_h_cv) against a double-precision CPU reference, and its off switch against gr_norm (gr_norm_h).
//
//   cvec_parity    T = 1, 3, 8 tokens; project (scale 1 and 0.5, and without the pending write) and add.
//                  On: the residual and the normed rows match the reference.  Off: bit for bit the plain kernel.
#include "mtl_rt.h"
#include "ref_quant.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
constexpr int N = 2560, HC = 4, LR = 320;
constexpr float EPS = 1e-6f;
struct GrArgs { uint32_t n_embd, hc, hc_lr, n_tok, apply; float eps; };
struct CvArgs { float s; uint32_t mode; };   // mode 0 = project, 1 = add

// Relative L2 error: |g - r| over |r|.
struct Rel {
    double se = 0, sr = 0;
    void add(double g, double r) { se += (g - r) * (g - r); sr += r * r; }
    double v() const { return sr > 0 ? std::sqrt(se / sr) : std::sqrt(se); }
};

// The reference in double: the pending write (when `apply`), the layer's vector on every stream (the same v for all of
// them), then the norm.  Rout: the steered residual; xo: the normed rows.
void reference(int T, const float* R, const float* bo, const float* inj, const float* wn, const float* v, float s, int mode,
               bool apply, std::vector<double>& Rout, std::vector<double>& xo) {
    Rout.assign((size_t) T * HC * N, 0.0);
    xo.assign((size_t) T * HC * N, 0.0);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < HC; ++c) {
            const size_t base = (size_t) (t * HC + c) * N;
            const double g = apply ? 2.0 / (1.0 + std::exp(-(double) inj[t * HC + c] / HC)) : 0.0;
            std::vector<double> r(N);
            for (int d = 0; d < N; ++d) r[d] = (double) R[base + d] + (apply ? (double) bo[(size_t) t * N + d] * g : 0.0);
            double dot = 0.0;
            for (int d = 0; d < N; ++d) dot += r[d] * (double) v[d];
            for (int d = 0; d < N; ++d) {
                r[d] = mode == 0 ? r[d] - (double) s * dot * (double) v[d] : r[d] + (double) s * (double) v[d];
                Rout[base + d] = r[d];
            }
            double ss = 0.0;
            for (int d = 0; d < N; ++d) ss += r[d] * r[d];
            const double rs = 1.0 / std::sqrt(ss / N + EPS);
            for (int d = 0; d < N; ++d) xo[base + d] = r[d] * (double) wn[(size_t) c * N + d] * rs;
        }
}
}  // namespace

int main() {
    mtl::Context cx;
    const int lib = cx.compile({"quant.metal", "units.metal", "mma.metal", "dense_mm.metal", "moe_mm.metal", "gr.metal",
                                "gemv.metal", "prompt.metal"});
    std::printf("device %s\n", cx.device_name().c_str());
    mtl::Pipeline pn = cx.pipeline(lib, "gr_norm"), pcv = cx.pipeline(lib, "gr_norm_cv");
    mtl::Pipeline phn = cx.pipeline(lib, "gr_norm_h"), phcv = cx.pipeline(lib, "gr_norm_h_cv");
    ref::Rng rng(7);
    int fails = 0;
    struct Case { const char* name; int mode; float s; bool apply; bool unit; };
    const Case cases[] = {{"project s=1", 0, 1.0f, true, true},
                          {"project s=0.5", 0, 0.5f, true, true},
                          {"project s=1, no pending write", 0, 1.0f, false, true},
                          {"add s=1", 1, 1.0f, true, false}};
    for (int T : {1, 3, 8}) {
        const size_t nR = (size_t) T * HC * N;
        mtl::Buffer R0 = cx.alloc(4 * nR), R = cx.alloc(4 * nR), Rn = cx.alloc(4 * nR), bo = cx.alloc(4 * (size_t) T * N),
                    inj = cx.alloc(4 * (size_t) T * HC), wn = cx.alloc(4 * (size_t) HC * N), xn = cx.alloc(4 * nR),
                    xnn = cx.alloc(4 * nR), xnh = cx.alloc(2 * nR), xnhn = cx.alloc(2 * nR), dir = cx.alloc(4 * N),
                    on = cx.alloc(4);
        for (size_t i = 0; i < nR; ++i) R0.as<float>()[i] = rng.sym();
        for (size_t i = 0; i < (size_t) T * N; ++i) bo.as<float>()[i] = 0.2f * rng.sym();
        for (int i = 0; i < T * HC; ++i) inj.as<float>()[i] = 2.0f * rng.sym();
        for (int i = 0; i < HC * N; ++i) wn.as<float>()[i] = 1.0f + 0.1f * rng.sym();
        for (const Case& cs : cases) {
            // the layer's direction: unit for project, a plain vector of norm 1 for add
            std::vector<float> v(N);
            double nrm = 0.0;
            for (int d = 0; d < N; ++d) { v[d] = rng.sym(); nrm += (double) v[d] * v[d]; }
            nrm = std::sqrt(nrm);
            for (int d = 0; d < N; ++d) { if (cs.unit) v[d] = (float) (v[d] / nrm); else v[d] = (float) (v[d] / nrm); }
            std::memcpy(dir.as<float>(), v.data(), 4 * (size_t) N);
            const GrArgs a{N, HC, LR, (uint32_t) T, cs.apply ? 1u : 0u, EPS};
            const CvArgs cva{cs.s, (uint32_t) cs.mode};
            // ---- on: the steered residual and the normed rows against the reference
            std::memcpy(R.as<float>(), R0.as<float>(), 4 * nR);
            *on.as<uint32_t>() = 1u;
            mtl::Dispatch d;
            d.p = &pcv;
            d.args = {mtl::Arg::B(R), mtl::Arg::B(bo), mtl::Arg::B(inj), mtl::Arg::B(wn), mtl::Arg::B(xn), mtl::Arg::V(a),
                      mtl::Arg::V(cva), mtl::Arg::B(on), mtl::Arg::B(dir)};
            d.grid[0] = HC;
            d.grid[1] = T;
            d.tg[0] = 256;
            cx.run({d});
            std::vector<double> Rref, xref;
            reference(T, R0.as<float>(), bo.as<float>(), inj.as<float>(), wn.as<float>(), v.data(), cs.s, cs.mode, cs.apply, Rref, xref);
            Rel er, ex;
            for (size_t i = 0; i < nR; ++i) {
                er.add(R.as<float>()[i], Rref[i]);
                ex.add(xn.as<float>()[i], xref[i]);
            }
            // project with scale 1 removes the vector's component: the steered residual has no dot with v left
            double dmax = 0.0;
            if (cs.mode == 0 && cs.s == 1.0f)
                for (int i = 0; i < T * HC; ++i) {
                    double dd = 0.0;
                    for (int d2 = 0; d2 < N; ++d2) dd += (double) R.as<float>()[(size_t) i * N + d2] * (double) v[d2];
                    dmax = std::max(dmax, std::fabs(dd));
                }
            const bool ok_on = er.v() < 1e-5 && ex.v() < 1e-5 && (cs.mode != 0 || cs.s != 1.0f || dmax < 1e-4);
            // ---- off: bit for bit the plain kernel
            std::memcpy(R.as<float>(), R0.as<float>(), 4 * nR);
            *on.as<uint32_t>() = 0u;
            cx.run({d});
            mtl::Dispatch dn;
            dn.p = &pn;
            dn.args = {mtl::Arg::B(Rn), mtl::Arg::B(bo), mtl::Arg::B(inj), mtl::Arg::B(wn), mtl::Arg::B(xnn), mtl::Arg::V(a)};
            dn.grid[0] = HC;
            dn.grid[1] = T;
            dn.tg[0] = 256;
            std::memcpy(Rn.as<float>(), R0.as<float>(), 4 * nR);
            cx.run({dn});
            const bool off_bits = std::memcmp(R.as<float>(), Rn.as<float>(), 4 * nR) == 0 &&
                                  std::memcmp(xn.as<float>(), xnn.as<float>(), 4 * nR) == 0;
            // ---- the fp16 rows of the prompt path (gr_norm_h_cv), on and off
            std::memcpy(R.as<float>(), R0.as<float>(), 4 * nR);
            *on.as<uint32_t>() = 1u;
            mtl::Dispatch dh;
            dh.p = &phcv;
            dh.args = {mtl::Arg::B(R), mtl::Arg::B(bo), mtl::Arg::B(inj), mtl::Arg::B(wn), mtl::Arg::B(xnh), mtl::Arg::V(a),
                       mtl::Arg::V(cva), mtl::Arg::B(on), mtl::Arg::B(dir)};
            dh.grid[0] = HC;
            dh.grid[1] = T;
            dh.tg[0] = 256;
            cx.run({dh});
            Rel eh;
            for (size_t i = 0; i < nR; ++i) eh.add((double) ref::f16_to_f32(xnh.as<uint16_t>()[i]), xref[i]);
            std::memcpy(R.as<float>(), R0.as<float>(), 4 * nR);
            *on.as<uint32_t>() = 0u;
            cx.run({dh});
            mtl::Dispatch dhn;
            dhn.p = &phn;
            dhn.args = {mtl::Arg::B(Rn), mtl::Arg::B(bo), mtl::Arg::B(inj), mtl::Arg::B(wn), mtl::Arg::B(xnhn), mtl::Arg::V(a)};
            dhn.grid[0] = HC;
            dhn.grid[1] = T;
            dhn.tg[0] = 256;
            std::memcpy(Rn.as<float>(), R0.as<float>(), 4 * nR);
            cx.run({dhn});
            const bool off_h = std::memcmp(xnhn.as<uint16_t>(), xnh.as<uint16_t>(), 2 * nR) == 0;
            const bool half_ok = eh.v() < 2e-3;
            const bool pass = ok_on && off_bits && off_h && half_ok;
            fails += !pass;
            char dot_note[64] = "";
            if (cs.mode == 0 && cs.s == 1.0f) std::snprintf(dot_note, sizeof dot_note, ", max |v.h| after %.1e", dmax);
            std::printf("T %d %-30s: residual rel %.2e, xn rel %.2e%s, fp16 rows rel %.2e, off == plain: %s  %s\n", T, cs.name,
                        er.v(), ex.v(), dot_note, eh.v(), off_bits && off_h ? "bitwise" : "NO", pass ? "PASS" : "FAIL");
        }
    }
    std::printf("%s\n", fails ? "SOME FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
