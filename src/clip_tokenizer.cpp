// clip_tokenizer.cpp - CLIP BPE Tokenizer Implementation
// Copyright (C) 2025 Advanced Micro Devices, Inc.
//
// Implements OpenAI CLIP's byte-level BPE tokenizer:
//   1. Lowercase + whitespace normalize
//   2. Split into word tokens (letters, digits, punctuation runs)
//   3. Byte-encode each word via bytes_to_unicode mapping
//   4. Apply BPE with </w> end-of-word suffix
//   5. Map to vocab IDs, add BOS/EOS, pad to max_length

#include "clip_tokenizer.h"
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

static std::vector<std::string> utf8_chars(const std::string& s) {
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < s.size()) {
        size_t len = 1;
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c >= 0xF0) len = 4;
        else if (c >= 0xE0) len = 3;
        else if (c >= 0xC0) len = 2;
        if (i + len > s.size()) len = s.size() - i;
        chars.push_back(s.substr(i, len));
        i += len;
    }
    return chars;
}

// ── Byte encoder (OpenAI bytes_to_unicode) ──────────────────────────────────

void CLIPTokenizer::init_byte_encoder() {
    // Bytes that map to themselves (printable ASCII + Latin-1 supplement)
    std::vector<int> bs;
    for (int i = 33; i <= 126; i++) bs.push_back(i);   // ! to ~
    for (int i = 161; i <= 172; i++) bs.push_back(i);   // inverted-! to not-sign
    for (int i = 174; i <= 255; i++) bs.push_back(i);   // registered to y-diaeresis

    std::vector<char32_t> cs;
    for (int b : bs) cs.push_back(static_cast<char32_t>(b));

    // Remaining bytes (0-32, 127-160, 173) map to 256+
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(static_cast<char32_t>(256 + n));
            n++;
        }
    }

    for (size_t i = 0; i < bs.size(); i++) {
        byte_encoder_[static_cast<uint8_t>(bs[i])] = cs[i];
    }
}

std::string CLIPTokenizer::byte_encode_token(const std::string& token) const {
    std::string result;
    for (unsigned char c : token) {
        auto it = byte_encoder_.find(c);
        if (it != byte_encoder_.end()) {
            result += char32_to_utf8(it->second);
        }
    }
    return result;
}

// ── JSON string parser ─────────────────────────────────────────────────────

static bool parse_json_string(const std::string& json, size_t& pos, std::string& out) {
    if (pos >= json.size() || json[pos] != '"') return false;
    pos++; // skip opening quote
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
                case 'u': {
                    if (pos + 4 >= json.size()) return false;
                    std::string hex = json.substr(pos + 1, 4);
                    char32_t cp = static_cast<char32_t>(std::stoul(hex, nullptr, 16));
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
    if (pos < json.size()) pos++; // skip closing quote
    return true;
}

// ── Load vocab + merges ─────────────────────────────────────────────────────

bool CLIPTokenizer::load(const std::string& vocab_path, const std::string& merges_path) {
    init_byte_encoder();

    // ── vocab.json ──
    {
        std::ifstream f(vocab_path);
        if (!f.is_open()) {
            std::cerr << "Failed to open vocab: " << vocab_path << std::endl;
            return false;
        }
        std::string content((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());

        size_t pos = content.find('{');
        if (pos == std::string::npos) return false;
        pos++;

        while (pos < content.size()) {
            // skip whitespace / commas
            while (pos < content.size() &&
                   (content[pos] == ' ' || content[pos] == '\n' ||
                    content[pos] == '\r' || content[pos] == '\t' || content[pos] == ','))
                pos++;

            if (pos >= content.size() || content[pos] == '}') break;

            std::string key;
            if (!parse_json_string(content, pos, key)) break;

            // skip colon + whitespace
            while (pos < content.size() && content[pos] != ':') pos++;
            if (pos < content.size()) pos++;
            while (pos < content.size() &&
                   (content[pos] == ' ' || content[pos] == '\n' ||
                    content[pos] == '\r' || content[pos] == '\t'))
                pos++;

            // parse integer value
            size_t end = pos;
            if (end < content.size() && content[end] == '-') end++;
            while (end < content.size() && content[end] >= '0' && content[end] <= '9') end++;
            int value = std::stoi(content.substr(pos, end - pos));
            pos = end;

            vocab_[key] = value;
        }
        std::cout << "  Loaded vocab: " << vocab_.size() << " tokens" << std::endl;
    }

    // ── merges.txt ──
    {
        std::ifstream f(merges_path);
        if (!f.is_open()) {
            std::cerr << "Failed to open merges: " << merges_path << std::endl;
            return false;
        }

        std::string line;
        bool first_line = true;
        int rank = 0;
        while (std::getline(f, line)) {
            // strip trailing \r
            if (!line.empty() && line.back() == '\r') line.pop_back();
            // skip header (#version: ...)
            if (first_line && !line.empty() && line[0] == '#') { first_line = false; continue; }
            first_line = false;
            if (line.empty()) continue;

            auto sp = line.find(' ');
            if (sp == std::string::npos) continue;

            std::string a = line.substr(0, sp);
            std::string b = line.substr(sp + 1);
            merge_ranks_[a + " " + b] = rank++;
        }
        std::cout << "  Loaded merges: " << rank << " rules" << std::endl;
    }

    return true;
}

// ── Text preprocessing ──────────────────────────────────────────────────────

std::string CLIPTokenizer::whitespace_clean(const std::string& text) {
    std::string r;
    bool last_space = false;
    for (char c : text) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (!last_space) { r += ' '; last_space = true; }
        } else {
            r += c; last_space = false;
        }
    }
    size_t s = r.find_first_not_of(' ');
    if (s == std::string::npos) return "";
    size_t e = r.find_last_not_of(' ');
    return r.substr(s, e - s + 1);
}

std::string CLIPTokenizer::to_lower(const std::string& text) {
    std::string r = text;
    std::transform(r.begin(), r.end(), r.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return r;
}

// ── Word splitting ──────────────────────────────────────────────────────────
// Approximation of CLIP's regex:
//   <|startoftext|>|<|endoftext|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+

std::vector<std::string> CLIPTokenizer::split_words(const std::string& text) const {
    std::vector<std::string> words;
    size_t i = 0;
    while (i < text.size()) {
        // Skip whitespace
        if (text[i] == ' ' || text[i] == '\t' || text[i] == '\n' || text[i] == '\r') {
            i++; continue;
        }

        // Contractions: 's 't 'm 'd 're 've 'll
        if (text[i] == '\'' && i + 1 < text.size()) {
            char n = text[i + 1];
            if (n == 's' || n == 't' || n == 'm' || n == 'd') {
                words.push_back(std::string("'") + n);
                i += 2; continue;
            }
            if (i + 2 < text.size()) {
                std::string two(text.data() + i + 1, 2);
                if (two == "re" || two == "ve" || two == "ll") {
                    words.push_back("'" + two);
                    i += 3; continue;
                }
            }
        }

        // Run of ASCII letters
        if (std::isalpha(static_cast<unsigned char>(text[i]))) {
            size_t start = i;
            while (i < text.size() && std::isalpha(static_cast<unsigned char>(text[i]))) i++;
            words.push_back(text.substr(start, i - start));
            continue;
        }

        // Single digit
        if (std::isdigit(static_cast<unsigned char>(text[i]))) {
            words.push_back(std::string(1, text[i]));
            i++; continue;
        }

        // Run of non-letter, non-digit, non-whitespace (punctuation etc.)
        {
            size_t start = i;
            while (i < text.size() &&
                   !std::isalpha(static_cast<unsigned char>(text[i])) &&
                   !std::isdigit(static_cast<unsigned char>(text[i])) &&
                   text[i] != ' ' && text[i] != '\t' &&
                   text[i] != '\n' && text[i] != '\r')
                i++;
            if (i > start) words.push_back(text.substr(start, i - start));
        }
    }
    return words;
}

// ── BPE ─────────────────────────────────────────────────────────────────────

static std::vector<std::pair<std::string, std::string>>
get_pairs(const std::vector<std::string>& word) {
    std::vector<std::pair<std::string, std::string>> pairs;
    for (size_t i = 0; i + 1 < word.size(); i++)
        pairs.push_back({word[i], word[i + 1]});
    return pairs;
}

std::string CLIPTokenizer::bpe(const std::string& token) const {
    auto cache_it = bpe_cache_.find(token);
    if (cache_it != bpe_cache_.end()) return cache_it->second;

    auto chars = utf8_chars(token);
    if (chars.empty()) return "";

    // CLIP adds </w> suffix to the last character
    std::vector<std::string> word(chars.begin(), chars.end());
    word.back() += "</w>";

    if (word.size() == 1) {
        bpe_cache_[token] = word[0];
        return word[0];
    }

    while (true) {
        auto pairs = get_pairs(word);
        if (pairs.empty()) break;

        // Find pair with lowest merge rank
        int best_rank = INT_MAX;
        std::pair<std::string, std::string> best_pair;
        for (auto& p : pairs) {
            auto it = merge_ranks_.find(p.first + " " + p.second);
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pair = p;
            }
        }
        if (best_rank == INT_MAX) break;

        // Merge all occurrences of best_pair
        std::vector<std::string> new_word;
        size_t i = 0;
        while (i < word.size()) {
            // Find next occurrence of first element
            size_t j = i;
            bool found = false;
            while (j < word.size()) {
                if (word[j] == best_pair.first) { found = true; break; }
                j++;
            }
            if (!found) {
                for (size_t k = i; k < word.size(); k++) new_word.push_back(word[k]);
                break;
            }
            for (size_t k = i; k < j; k++) new_word.push_back(word[k]);

            if (j < word.size() - 1 &&
                word[j] == best_pair.first && word[j + 1] == best_pair.second) {
                new_word.push_back(best_pair.first + best_pair.second);
                i = j + 2;
            } else {
                new_word.push_back(word[j]);
                i = j + 1;
            }
        }
        word = new_word;
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

std::vector<int64_t> CLIPTokenizer::encode(const std::string& text, int max_length) const {
    std::string processed = to_lower(whitespace_clean(text));

    std::vector<int64_t> ids;
    ids.push_back(bos_token_id_); // <|startoftext|>

    auto words = split_words(processed);
    for (const auto& word : words) {
        std::string encoded = byte_encode_token(word);
        std::string bpe_result = bpe(encoded);

        std::istringstream ss(bpe_result);
        std::string tok;
        while (ss >> tok) {
            auto it = vocab_.find(tok);
            ids.push_back(it != vocab_.end() ? it->second : eos_token_id_);
        }
    }

    ids.push_back(eos_token_id_); // <|endoftext|>

    // Truncate (preserve BOS)
    if (static_cast<int>(ids.size()) > max_length) {
        ids.resize(max_length);
        ids.back() = eos_token_id_;
    }

    // Pad
    while (static_cast<int>(ids.size()) < max_length)
        ids.push_back(pad_token_id_);

    return ids;
}

} // namespace sd_npu
