// qwen3_tokenizer.cpp - Qwen3 byte-level BPE tokenizer for FLUX.2-klein
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#include "qwen3_tokenizer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <climits>

namespace sd_npu {

// ── UTF-8 helpers ───────────────────────────────────────────────────────────

static std::string char32_to_utf8(char32_t c) {
    std::string r;
    if (c < 0x80) {
        r += static_cast<char>(c);
    } else if (c < 0x800) {
        r += static_cast<char>(0xC0 | (c >> 6));
        r += static_cast<char>(0x80 | (c & 0x3F));
    } else if (c < 0x10000) {
        r += static_cast<char>(0xE0 | (c >> 12));
        r += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
        r += static_cast<char>(0x80 | (c & 0x3F));
    } else {
        r += static_cast<char>(0xF0 | (c >> 18));
        r += static_cast<char>(0x80 | ((c >> 12) & 0x3F));
        r += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
        r += static_cast<char>(0x80 | (c & 0x3F));
    }
    return r;
}

// Decode a UTF-8 string into codepoints, tracking each codepoint's byte offset.
struct Utf8Cp { char32_t cp; size_t byte_start; size_t byte_len; };

static std::vector<Utf8Cp> decode_utf8(const std::string& s) {
    std::vector<Utf8Cp> out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t len = 1;
        char32_t cp = c;
        if (c >= 0xF0)      { len = 4; cp = c & 0x07; }
        else if (c >= 0xE0) { len = 3; cp = c & 0x0F; }
        else if (c >= 0xC0) { len = 2; cp = c & 0x1F; }
        if (i + len > s.size()) len = s.size() - i;
        for (size_t k = 1; k < len; k++)
            cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
        out.push_back({cp, i, len});
        i += len;
    }
    return out;
}

// ── Character-class predicates (approximation of Rust regex \p{L}, \p{N}, \s) ─

static bool cp_is_digit(char32_t c) { return c >= '0' && c <= '9'; }
static bool cp_is_space(char32_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
           c == '\v' || c == 0x00A0 || c == 0x0085;
}
static bool cp_is_crlf(char32_t c) { return c == '\r' || c == '\n'; }
// \p{L}: ASCII letters plus (as an approximation) any non-ASCII, non-space codepoint.
static bool cp_is_letter(char32_t c) {
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) return true;
    if (c < 0x80) return false;
    return !cp_is_space(c);
}

// ── Byte encoder (GPT-2 / Qwen bytes_to_unicode) ────────────────────────────

void Qwen3Tokenizer::init_byte_encoder() {
    std::vector<int> bs;
    for (int i = 33; i <= 126; i++) bs.push_back(i);
    for (int i = 161; i <= 172; i++) bs.push_back(i);
    for (int i = 174; i <= 255; i++) bs.push_back(i);

    std::vector<char32_t> cs;
    for (int b : bs) cs.push_back(static_cast<char32_t>(b));

    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(static_cast<char32_t>(256 + n));
            n++;
        }
    }
    for (size_t i = 0; i < bs.size(); i++)
        byte_encoder_[static_cast<uint8_t>(bs[i])] = cs[i];
}

std::string Qwen3Tokenizer::byte_encode(const std::string& utf8_piece) const {
    std::string result;
    for (unsigned char c : utf8_piece) {
        auto it = byte_encoder_.find(c);
        if (it != byte_encoder_.end()) result += char32_to_utf8(it->second);
    }
    return result;
}

// ── JSON string parser (handles \uXXXX and raw UTF-8) ───────────────────────

static bool parse_json_string(const std::string& json, size_t& pos, std::string& out) {
    if (pos >= json.size() || json[pos] != '"') return false;
    pos++;
    out.clear();
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\') {
            pos++;
            if (pos >= json.size()) return false;
            switch (json[pos]) {
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case 'n':  out += '\n'; break;
                case 'r':  out += '\r'; break;
                case 't':  out += '\t'; break;
                case '/':  out += '/';  break;
                case 'b':  out += '\b'; break;
                case 'f':  out += '\f'; break;
                case 'u': {
                    if (pos + 4 >= json.size()) return false;
                    std::string hex = json.substr(pos + 1, 4);
                    char32_t cp = static_cast<char32_t>(std::stoul(hex, nullptr, 16));
                    // Handle UTF-16 surrogate pairs.
                    if (cp >= 0xD800 && cp <= 0xDBFF && pos + 10 < json.size() &&
                        json[pos + 5] == '\\' && json[pos + 6] == 'u') {
                        std::string hex2 = json.substr(pos + 7, 4);
                        char32_t lo = static_cast<char32_t>(std::stoul(hex2, nullptr, 16));
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        pos += 6;
                    }
                    out += char32_to_utf8(cp);
                    pos += 4;
                    break;
                }
                default: out += json[pos]; break;
            }
        } else {
            out += json[pos];
        }
        pos++;
    }
    if (pos < json.size()) pos++;
    return true;
}

// ── Load vocab + merges ─────────────────────────────────────────────────────

bool Qwen3Tokenizer::load(const std::string& vocab_path, const std::string& merges_path) {
    init_byte_encoder();

    {
        std::ifstream f(vocab_path, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "Qwen3Tokenizer: failed to open vocab: " << vocab_path << std::endl;
            return false;
        }
        std::string content((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
        size_t pos = content.find('{');
        if (pos == std::string::npos) return false;
        pos++;
        while (pos < content.size()) {
            while (pos < content.size() &&
                   (content[pos] == ' ' || content[pos] == '\n' ||
                    content[pos] == '\r' || content[pos] == '\t' || content[pos] == ','))
                pos++;
            if (pos >= content.size() || content[pos] == '}') break;
            std::string key;
            if (!parse_json_string(content, pos, key)) break;
            while (pos < content.size() && content[pos] != ':') pos++;
            if (pos < content.size()) pos++;
            while (pos < content.size() &&
                   (content[pos] == ' ' || content[pos] == '\n' ||
                    content[pos] == '\r' || content[pos] == '\t'))
                pos++;
            size_t end = pos;
            if (end < content.size() && content[end] == '-') end++;
            while (end < content.size() && content[end] >= '0' && content[end] <= '9') end++;
            int value = std::stoi(content.substr(pos, end - pos));
            pos = end;
            vocab_[key] = value;
        }
        std::cout << "  Qwen3 vocab: " << vocab_.size() << " tokens" << std::endl;
    }

    {
        std::ifstream f(merges_path, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "Qwen3Tokenizer: failed to open merges: " << merges_path << std::endl;
            return false;
        }
        std::string line;
        bool first_line = true;
        int rank = 0;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (first_line && !line.empty() && line[0] == '#') { first_line = false; continue; }
            first_line = false;
            if (line.empty()) continue;
            auto sp = line.find(' ');
            if (sp == std::string::npos) continue;
            std::string a = line.substr(0, sp);
            std::string b = line.substr(sp + 1);
            merge_ranks_[a + " " + b] = rank++;
        }
        std::cout << "  Qwen3 merges: " << rank << " rules" << std::endl;
    }
    return true;
}

// ── Pre-tokenizer (hand-rolled equivalent of the Qwen Split regex) ──────────
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   | [^\r\n\p{L}\p{N}]?\p{L}+
//   | \p{N}
//   |  ?[^\s\p{L}\p{N}]+[\r\n]*
//   | \s*[\r\n]+
//   | \s+(?!\S)
//   | \s+

std::vector<std::string> Qwen3Tokenizer::pre_tokenize(const std::string& text) const {
    std::vector<std::string> pieces;
    auto cps = decode_utf8(text);
    const size_t n = cps.size();
    size_t i = 0;

    auto lower = [](char32_t c) -> char32_t {
        return (c >= 'A' && c <= 'Z') ? c + 32 : c;
    };
    auto emit = [&](size_t start_cp, size_t end_cp) {
        if (end_cp <= start_cp) return;
        size_t bs = cps[start_cp].byte_start;
        size_t be = cps[end_cp - 1].byte_start + cps[end_cp - 1].byte_len;
        pieces.push_back(text.substr(bs, be - bs));
    };

    while (i < n) {
        char32_t c = cps[i].cp;

        // 1. Contractions ('s 't 're 've 'm 'll 'd), case-insensitive.
        if (c == '\'' && i + 1 < n) {
            char32_t a = lower(cps[i + 1].cp);
            if (a == 's' || a == 't' || a == 'm' || a == 'd') {
                emit(i, i + 2); i += 2; continue;
            }
            if (i + 2 < n) {
                char32_t b = lower(cps[i + 2].cp);
                if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') ||
                    (a == 'l' && b == 'l')) {
                    emit(i, i + 3); i += 3; continue;
                }
            }
        }

        // 2. [^\r\n\p{L}\p{N}]? \p{L}+
        {
            size_t j = i;
            bool have_prefix = false;
            if (!cp_is_crlf(c) && !cp_is_letter(c) && !cp_is_digit(c) &&
                i + 1 < n && cp_is_letter(cps[i + 1].cp)) {
                have_prefix = true;
                j = i + 1;
            }
            if (cp_is_letter(cps[j].cp)) {
                size_t start = have_prefix ? i : j;
                while (j < n && cp_is_letter(cps[j].cp)) j++;
                emit(start, j); i = j; continue;
            }
        }

        // 3. \p{N} (single digit)
        if (cp_is_digit(c)) { emit(i, i + 1); i += 1; continue; }

        // 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
        {
            size_t j = i;
            if (c == ' ' && i + 1 < n) {
                char32_t d = cps[i + 1].cp;
                if (!cp_is_space(d) && !cp_is_letter(d) && !cp_is_digit(d)) j = i + 1;
            }
            if (j < n && !cp_is_space(cps[j].cp) &&
                !cp_is_letter(cps[j].cp) && !cp_is_digit(cps[j].cp)) {
                while (j < n && !cp_is_space(cps[j].cp) &&
                       !cp_is_letter(cps[j].cp) && !cp_is_digit(cps[j].cp)) j++;
                while (j < n && cp_is_crlf(cps[j].cp)) j++;
                emit(i, j); i = j; continue;
            }
        }

        // 5. \s*[\r\n]+
        if (cp_is_space(c)) {
            size_t j = i;
            while (j < n && cp_is_space(cps[j].cp) && !cp_is_crlf(cps[j].cp)) j++;
            if (j < n && cp_is_crlf(cps[j].cp)) {
                while (j < n && cp_is_crlf(cps[j].cp)) j++;
                emit(i, j); i = j; continue;
            }
        }

        // 6/7. \s+(?!\S) and \s+
        if (cp_is_space(c)) {
            size_t j = i;
            while (j < n && cp_is_space(cps[j].cp)) j++;
            // \s+(?!\S): keep a trailing whitespace run whole. If more non-space
            // follows, the ByteLevel splitter would leave the last space to the
            // next word; approximate by splitting off all but the last space.
            if (j < n) {
                // leave last space for the following token
                if (j - i > 1) { emit(i, j - 1); i = j - 1; continue; }
                emit(i, j); i = j; continue;
            }
            emit(i, j); i = j; continue;
        }

        // Fallback: emit single codepoint.
        emit(i, i + 1); i += 1;
    }
    return pieces;
}

// ── BPE (byte-level, no </w> suffix) ────────────────────────────────────────

static std::vector<std::string> utf8_chars(const std::string& s) {
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t len = 1;
        if (c >= 0xF0) len = 4;
        else if (c >= 0xE0) len = 3;
        else if (c >= 0xC0) len = 2;
        if (i + len > s.size()) len = s.size() - i;
        chars.push_back(s.substr(i, len));
        i += len;
    }
    return chars;
}

std::string Qwen3Tokenizer::bpe(const std::string& token) const {
    auto cache_it = bpe_cache_.find(token);
    if (cache_it != bpe_cache_.end()) return cache_it->second;

    std::vector<std::string> word = utf8_chars(token);
    if (word.size() <= 1) {
        bpe_cache_[token] = token;
        return token;
    }

    while (true) {
        int best_rank = INT_MAX;
        std::pair<std::string, std::string> best_pair;
        for (size_t k = 0; k + 1 < word.size(); k++) {
            auto it = merge_ranks_.find(word[k] + " " + word[k + 1]);
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pair = {word[k], word[k + 1]};
            }
        }
        if (best_rank == INT_MAX) break;

        std::vector<std::string> new_word;
        size_t i = 0;
        while (i < word.size()) {
            if (i + 1 < word.size() &&
                word[i] == best_pair.first && word[i + 1] == best_pair.second) {
                new_word.push_back(best_pair.first + best_pair.second);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i += 1;
            }
        }
        word.swap(new_word);
        if (word.size() == 1) break;
    }

    std::string result;
    for (size_t i = 0; i < word.size(); i++) {
        if (i > 0) result += " ";
        result += word[i];
    }
    bpe_cache_[token] = result;
    return result;
}

// ── Encode ──────────────────────────────────────────────────────────────────

std::vector<int64_t> Qwen3Tokenizer::encode_segment(const std::string& text) const {
    std::vector<int64_t> ids;
    for (const auto& piece : pre_tokenize(text)) {
        std::string encoded = byte_encode(piece);
        std::string bpe_result = bpe(encoded);
        std::istringstream ss(bpe_result);
        std::string tok;
        while (ss >> tok) {
            auto it = vocab_.find(tok);
            if (it != vocab_.end()) ids.push_back(it->second);
        }
    }
    return ids;
}

Qwen3Encoding Qwen3Tokenizer::encode_prompt(const std::string& prompt, int max_length) const {
    // Qwen chat template (enable_thinking=False -> empty think block):
    //   <|im_start|>user\n{PROMPT}<|im_end|>\n
    //   <|im_start|>assistant\n<think>\n\n</think>\n\n
    std::vector<int64_t> ids;
    ids.push_back(im_start_id_);
    { auto s = encode_segment("user\n");      ids.insert(ids.end(), s.begin(), s.end()); }
    { auto s = encode_segment(prompt);        ids.insert(ids.end(), s.begin(), s.end()); }
    ids.push_back(im_end_id_);
    { auto s = encode_segment("\n");          ids.insert(ids.end(), s.begin(), s.end()); }
    ids.push_back(im_start_id_);
    { auto s = encode_segment("assistant\n"); ids.insert(ids.end(), s.begin(), s.end()); }
    ids.push_back(think_open_id_);
    { auto s = encode_segment("\n\n");         ids.insert(ids.end(), s.begin(), s.end()); }
    ids.push_back(think_close_id_);
    { auto s = encode_segment("\n\n");         ids.insert(ids.end(), s.begin(), s.end()); }

    // Truncate to max_length if needed.
    if (static_cast<int>(ids.size()) > max_length)
        ids.resize(max_length);

    int real = static_cast<int>(ids.size());

    Qwen3Encoding enc;
    enc.seq_len = max_length;
    enc.input_ids.resize(max_length, pad_token_id_);
    enc.attention_mask.assign(max_length, 0);
    enc.position_ids.resize(max_length);
    for (int i = 0; i < real; i++) {
        enc.input_ids[i] = ids[i];
        enc.attention_mask[i] = 1;
    }
    for (int i = 0; i < max_length; i++) enc.position_ids[i] = i;
    return enc;
}

} // namespace sd_npu
