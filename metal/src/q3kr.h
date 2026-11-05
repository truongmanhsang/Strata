// metal/src/q3kr.h - Q3_K blocks re-laid for the Metal decode GEMV (the T_Q3_KR = 111 format of quant.metal / units.metal).
//
// A Q3_K block (ggml block_q3_K, 110 bytes) keeps value i = 128 n + 32 j + l (n 0..1, j 0..3, l 0..31) as bits 2j..2j+1
// of qs[32 n + l] and bit 4n + j of hmask[l], and its 16 6-bit scales packed in 12 bytes.  A 32-value unit (n, j) thus
// reads all 64 bytes of qs and hmask and unpacks two scales.  Q3_KR (114 bytes) keeps each unit's bits together:
//
//   [0, 2)            d (half, Q3_K's)
//   [2 + 14 u, +14)   unit u = 4 n + j: L[4] ushort, H[2] ushort, S[2] int8
//                     word w (0..15) covers the unit's values 2w and 2w + 1 (low byte, high byte):
//                       low 2 bits:  bits 2a, 8 + 2a of L[b], w = 4a + b
//                       high bit:    bits a, 8 + a of H[b],    w = 2a + b
//                     S = the unit's two 16-value scales - 32 (Q3_K's sc 0..63 - 32)
//
// The decoder builds the same 3-bit codes, scales and order as Q3_K's, so a GEMV over Q3_KR is bitwise the Q3_K one.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace strata_metal {

constexpr size_t kQ3KBlock = 110, kQ3KRBlock = 114;

// Q3_K's 6-bit scale i (0..15) of a block's 12 scale bytes (ggml's unpacking)
inline int q3k_scale(const uint8_t* s, int i) {
    const int lo = i < 8 ? (s[i] & 0xF) : (s[i - 8] >> 4);
    const int hi = (s[8 + (i & 3)] >> (2 * (i >> 2))) & 3;
    return lo | (hi << 4);
}

inline void q3kr_block(const uint8_t* src, uint8_t* dst) {
    const uint8_t* hmask = src;
    const uint8_t* qs = src + 32;
    const uint8_t* sc = src + 96;
    std::memcpy(dst, src + 108, 2);   // d
    for (int u = 0; u < 8; ++u) {
        const int n = u >> 2, j = u & 3;
        uint16_t L[4] = {0, 0, 0, 0}, H[2] = {0, 0};
        for (int l = 0; l < 32; ++l) {
            const int lo = (qs[32 * n + l] >> (2 * j)) & 3;
            const int hb = (hmask[l] >> (4 * n + j)) & 1;
            const int w = l >> 1, hi8 = (l & 1) ? 8 : 0;
            L[w & 3] |= (uint16_t) (lo << (hi8 + 2 * (w >> 2)));
            H[w & 1] |= (uint16_t) (hb << (hi8 + (w >> 1)));
        }
        uint8_t* p = dst + 2 + 14 * u;
        std::memcpy(p, L, 8);
        std::memcpy(p + 8, H, 4);
        p[12] = (uint8_t) (int8_t) (q3k_scale(sc, 8 * n + 2 * j) - 32);
        p[13] = (uint8_t) (int8_t) (q3k_scale(sc, 8 * n + 2 * j + 1) - 32);
    }
}

// rows x n_in Q3_K values (n_in a multiple of 256) -> the same rows in Q3_KR; dst holds rows * n_in / 256 * 114 bytes
inline void q3kr_repack(const uint8_t* src, uint8_t* dst, size_t rows, size_t n_in) {
    const size_t nb = rows * (n_in / 256);
    for (size_t b = 0; b < nb; ++b) q3kr_block(src + b * kQ3KBlock, dst + b * kQ3KRBlock);
}

}  // namespace strata_metal
