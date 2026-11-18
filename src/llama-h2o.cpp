#include "llama-h2o.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <numeric>

static void h2o_grow(llama_h2o::layer & l, size_t n) {
    for (auto & d : l.dropped) {
        if (d.size() < n) {
            d.resize(n, 0);
        }
    }
    for (auto & a : l.acc) {
        if (a.size() < n) {
            a.resize(n, 0.0f);
        }
    }
}

void llama_h2o::touch(int il, uint32_t n_head_kv_cur) {
    GGML_ASSERT(n_head_kv == 0 || n_head_kv == n_head_kv_cur);
    n_head_kv = n_head_kv_cur;

    auto & l = layers[il];
    if (l.dropped.empty()) {
        l.dropped.resize(n_head_kv);
        l.acc.resize(n_head_kv);
    }
}

void llama_h2o::fill_mask(int il, float * data, int64_t n_kv, int64_t n_head) {
    auto & l = layers.at(il);
    h2o_grow(l, n_kv);

    GGML_ASSERT(n_head % n_head_kv == 0);
    const int64_t r = n_head / n_head_kv;

    for (int64_t h = 0; h < n_head; ++h) {
        const auto & d = l.dropped[h / r];
        float * row = data + h*n_kv;
        for (int64_t j = 0; j < n_kv; ++j) {
            row[j] = d[j] ? -INFINITY : 0.0f;
        }
    }
}

void llama_h2o::add_acc(int il, const float * data, int64_t n_kv, int64_t n_head) {
    auto & l = layers.at(il);
    h2o_grow(l, n_kv);

    GGML_ASSERT(n_head % n_head_kv == 0);
    const int64_t r = n_head / n_head_kv;

    for (int64_t h = 0; h < n_head; ++h) {
        auto & a = l.acc[h / r];
        const float * row = data + h*n_kv;
        for (int64_t j = 0; j < n_kv; ++j) {
            a[j] += row[j];
        }
    }
}

uint32_t llama_h2o::evict() {
    uint32_t n_evicted = 0;

    const uint32_t B = budget;
    if (B <= n_sink) {
        return 0;
    }

    std::vector<int32_t> kept;
    std::vector<int32_t> mid;

    for (auto & [il, l] : layers) {
        h2o_grow(l, n_pos);

        for (uint32_t h = 0; h < n_head_kv; ++h) {
            auto & d = l.dropped[h];
            auto & a = l.acc[h];

            kept.clear();
            for (uint32_t j = 0; j < n_pos; ++j) {
                if (!d[j]) {
                    kept.push_back(j);
                }
            }

            const uint32_t n = kept.size();
            if (n <= B) {
                continue;
            }

            const uint32_t nr = (B - n_sink)/2;
            const uint32_t nh = B - n_sink - nr;

            // 中间段:保留表里第 [SINK, n - nr) 个
            mid.assign(kept.begin() + n_sink, kept.begin() + (n - nr));

            // acc 最高的 nh 个留下(并列时位置靠前的优先),其余淘汰
            std::stable_sort(mid.begin(), mid.end(), [&](int32_t x, int32_t y) {
                return a[x] > a[y];
            });

            for (size_t i = nh; i < mid.size(); ++i) {
                d[mid[i]] = 1;
                a[mid[i]] = 0.0f;
                n_evicted++;
            }
        }
    }

    return n_evicted;
}

std::vector<int32_t> llama_h2o::keep(int il, uint32_t head_kv) const {
    std::vector<int32_t> res;

    const auto it = layers.find(il);
    if (it == layers.end() || head_kv >= it->second.dropped.size()) {
        return res;
    }

    const auto & d = it->second.dropped[head_kv];
    for (uint32_t j = 0; j < n_pos; ++j) {
        if (j >= d.size() || !d[j]) {
            res.push_back(j);
        }
    }

    return res;
}

void llama_h2o::reset() {
    n_pos = 0;
    for (auto & [il, l] : layers) {
        for (auto & d : l.dropped) {
            d.clear();
        }
        for (auto & a : l.acc) {
            a.clear();
        }
    }
}
