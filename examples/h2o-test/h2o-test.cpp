// H2O M1 验证用驱动:按轮喂 token,导出 logits 与各 (层, 键值头) 保留表。
// 同一份源码也能对未改的 llama.cpp 编译(没有 LLAMA_H2O_API 时只跑关功能的路径)。
//
//   llama-h2o-test <model.gguf> <turns.txt> <out 前缀> [--budget N] [--fa auto|off] [--kv f16|f32]
//                  [--threads N] [--batch N] [--ctx N]
//
// turns.txt 每行一段:
//   H id id ...   历史轮:累计开,读完后淘汰一次;导出最后一个 token 的 logits
//   P id id ...   探题:累计关,不淘汰;导出每个 token 的 logits
// 输出:<out>.logits.bin(float32,行 = 导出的位置,列 = 词表)、<out>.keep.jsonl(每个 H 轮淘汰后的保留表)、
//       <out>.time.txt(每段耗时)

#include "llama.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model turns out [--budget N] [--fa auto|off] [--kv f16|f32] [--threads N] [--batch N] [--ctx N]\n", argv[0]);
        return 1;
    }
    const std::string path_model = argv[1];
    const std::string path_turns = argv[2];
    const std::string out        = argv[3];

    int budget = 0, n_threads = 6, n_batch = 512, n_ctx = 8192;
    std::string fa = "auto", kv = "f16";
    for (int i = 4; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if      (a == "--budget")  budget    = atoi(argv[i + 1]);
        else if (a == "--fa")      fa        = argv[i + 1];
        else if (a == "--kv")      kv        = argv[i + 1];
        else if (a == "--threads") n_threads = atoi(argv[i + 1]);
        else if (a == "--batch")   n_batch   = atoi(argv[i + 1]);
        else if (a == "--ctx")     n_ctx     = atoi(argv[i + 1]);
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }

    struct turn { char kind; std::vector<llama_token> ids; };
    std::vector<turn> turns;
    {
        std::ifstream f(path_turns);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            turn t; ss >> t.kind;
            int id;
            while (ss >> id) t.ids.push_back(id);
            turns.push_back(std::move(t));
        }
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path_model.c_str(), mparams);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = n_ctx;
    cparams.n_batch         = n_batch;
    cparams.n_ubatch        = n_batch;
    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.flash_attn_type = fa == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_AUTO;
    cparams.type_k = cparams.type_v = kv == "f32" ? GGML_TYPE_F32 : GGML_TYPE_F16;
#ifdef LLAMA_H2O_API
    cparams.h2o_budget = budget;
#else
    if (budget > 0) { fprintf(stderr, "this build has no H2O\n"); return 1; }
#endif

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    FILE * f_logits = fopen((out + ".logits.bin").c_str(), "wb");
    FILE * f_keep   = fopen((out + ".keep.jsonl").c_str(), "w");
    FILE * f_time   = fopen((out + ".time.txt").c_str(), "w");

    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    llama_pos n_past = 0;

    for (size_t it = 0; it < turns.size(); ++it) {
        const auto & t = turns[it];
        const bool probe = t.kind == 'P';
#ifdef LLAMA_H2O_API
        llama_h2o_set_accumulate(ctx, !probe);
#endif
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t c0 = 0; c0 < t.ids.size(); c0 += n_batch) {
            const size_t n = std::min((size_t) n_batch, t.ids.size() - c0);
            batch.n_tokens = n;
            for (size_t i = 0; i < n; ++i) {
                batch.token[i]     = t.ids[c0 + i];
                batch.pos[i]       = n_past + i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = probe || (c0 + i + 1 == t.ids.size());
            }
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
            n_past += n;
            for (size_t i = 0; i < n; ++i) {
                if (batch.logits[i]) {
                    fwrite(llama_get_logits_ith(ctx, i), sizeof(float), n_vocab, f_logits);
                }
            }
        }
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        int n_ev = -1;
#ifdef LLAMA_H2O_API
        if (!probe && budget > 0) {
            n_ev = llama_h2o_evict(ctx);
            std::vector<int32_t> buf(n_past);
            const int nh = llama_h2o_n_head_kv(ctx);
            for (int li = 0; li < llama_h2o_n_layers(ctx); ++li) {
                const int il = llama_h2o_layer(ctx, li);
                for (int h = 0; h < nh; ++h) {
                    const int n = llama_h2o_get_keep(ctx, il, h, buf.data(), buf.size());
                    fprintf(f_keep, "{\"turn\": %zu, \"layer\": %d, \"head\": %d, \"keep\": [", it, il, h);
                    for (int i = 0; i < n; ++i) fprintf(f_keep, "%s%d", i ? "," : "", buf[i]);
                    fprintf(f_keep, "]}\n");
                }
            }
        }
#endif
        fprintf(f_time, "%zu %c %zu %.3f\n", it, t.kind, t.ids.size(), secs);
        fprintf(stderr, "turn %zu %c n=%zu n_past=%d evicted=%d %.2fs\n", it, t.kind, t.ids.size(), n_past, n_ev, secs);
    }

    fclose(f_logits); fclose(f_keep); fclose(f_time);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
