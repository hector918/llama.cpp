#pragma once

// H2O 式 KV 保留(M1:掩码模式)
//
// 每个全注意力层、每个键值头各自一张保留表;被淘汰的缓存位置在注意力里加 -inf,
// 缓存本身不压缩。规则照参考实现 s478.py(evict 的 H 支、probe):
//   - 累计量 acc[层][键值头][位置] = 历史轮读入时,该键值头名下各查询头、各查询位置
//     softmax 后注意力权重之和(查询头 h 归键值头 h / (n_head / n_head_kv));
//   - 每读完一轮调一次 evict():某 (层, 头) 仍保留的位置数 n > B 时,留
//     最前 SINK(4)个 + 最近 nr = (B - SINK) / 2 个 + 中间段里 acc 最高的 nh = B - SINK - nr 个;
//   - 被淘汰位置的 acc 一并丢掉(之后它们注意力恒为 0,不会再被选回)。
//
// 限制(M1):单序列、只追加;cell 下标 == 第几个读入的位置。

#include <cstdint>
#include <map>
#include <vector>

struct llama_h2o {
    struct layer {
        // [键值头][cell]
        std::vector<std::vector<uint8_t>> dropped;
        std::vector<std::vector<float>>   acc;
    };

    uint32_t budget     = 0;     // B
    uint32_t n_sink     = 4;     // SINK
    bool     accumulate = true;  // 读入时是否累计(探题时关)

    uint32_t n_head_kv = 0;
    uint32_t n_pos     = 0;      // 已读入的位置数

    std::map<int, layer> layers; // 键 = 模型层号(只有全注意力层)

    // 建图时登记该层(已有则不动)
    void touch(int il, uint32_t n_head_kv);

    // 填 [n_kv, 1, n_head, 1] 的加性掩码:淘汰位 -inf,其余 0
    void fill_mask(int il, float * data, int64_t n_kv, int64_t n_head);

    // data: [1, n_kv, n_head] —— 本 ubatch 各查询位置的 softmax 权重沿查询维求和
    void add_acc(int il, const float * data, int64_t n_kv, int64_t n_head);

    // 按规则淘汰;返回本次新淘汰的 (层, 头, 位置) 个数
    uint32_t evict();

    // 某层某键值头当前保留的位置(升序)
    std::vector<int32_t> keep(int il, uint32_t head_kv) const;

    void reset();
};
