// Standalone validation harness for Qwen3Tokenizer (not part of the build).
#include "../include/sd/qwen3_tokenizer.h"
#include <iostream>
#include <vector>
#include <string>

int main() {
    using namespace sd_npu;
    const std::string dir =
        "C:\\Users\\mickraus\\Work\\sd-sandbox\\models\\amd_FLUX.2-klein-4B-amdnpu\\tokenizer\\";
    Qwen3Tokenizer tok;
    if (!tok.load(dir + "vocab.json", dir + "merges.txt")) {
        std::cerr << "load failed\n";
        return 1;
    }

    struct Case { std::string prompt; std::vector<int64_t> expect; };
    std::vector<Case> cases = {
        {"A cat", {151644,872,198,32,8251,151645,198,151644,77091,198,151667,271,151668,271}},
        {"A cat sitting on a wooden bench in a sunny park",
            {151644,872,198,32,8251,11699,389,264,22360,13425,304,264,39698,6118,151645,198,151644,77091,198,151667,271,151668,271}},
        {"photo-realistic: a Red Fox (v2), 4k!!!",
            {151644,872,198,11556,74795,4532,25,264,3731,13282,320,85,17,701,220,19,74,12069,151645,198,151644,77091,198,151667,271,151668,271}},
        {"123 dogs & 45 cats",
            {151644,872,198,16,17,18,12590,609,220,19,20,19423,151645,198,151644,77091,198,151667,271,151668,271}},
    };

    int failures = 0;
    for (auto& c : cases) {
        auto enc = tok.encode_prompt(c.prompt, 256);
        std::vector<int64_t> got(enc.input_ids.begin(),
                                 enc.input_ids.begin() + c.expect.size());
        bool ok = (got == c.expect);
        // also confirm padding starts right after expected
        bool pad_ok = enc.input_ids[c.expect.size()] == tok.pad_token_id();
        std::cout << (ok && pad_ok ? "PASS" : "FAIL") << "  \"" << c.prompt << "\"\n";
        if (!ok) {
            failures++;
            std::cout << "  expect:";
            for (auto v : c.expect) std::cout << " " << v;
            std::cout << "\n  got   :";
            for (auto v : got) std::cout << " " << v;
            std::cout << "\n";
        }
    }
    std::cout << (failures ? "SOME FAILED\n" : "ALL PASSED\n");
    return failures;
}
