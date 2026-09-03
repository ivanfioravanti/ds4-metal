#!/usr/bin/env python3
"""Tokenizer fixtures for the Qwen3.8 ("qwen35") pre-tokenizer.

--write: tokenize a fixed multilingual/code corpus with the HF tokenizer of
the given model dir and store the expected ids in tokenizer_cases.json.
--check: run `ds4 --raw --dump-tokens --cpu` on each case and compare.

Usage:
  python tokenizer_fixtures.py --write --model ~/ds4-gguf/qwen4-mini/hf
  python tokenizer_fixtures.py --check --ds4 ./ds4 --gguf ~/ds4-gguf/Qwen4-Mini-F32.gguf
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CASES_PATH = os.path.join(HERE, "tokenizer_cases.json")

CASES = [
    "Hello world",
    "Hello, world! How are you doing today?",
    "I'm sure they're here; we'll see. You've won't can't I'D THEY'RE",
    "It's 3.14159 and 1,000,000 or 42 items in 2024-09-02.",
    "   leading spaces and trailing   ",
    "tabs\tand\tnewlines\n\nand more\r\nwindows\r\n",
    "line1\n  line2\n\n\n    line3\n",
    "def f(x):\n    return x**2 + 3*x - 1  # comment\n",
    "int main(void) { printf(\"%d\\n\", 42); return 0; }",
    "const s = `template ${a + b} literal`;\n",
    "SELECT * FROM t WHERE a >= 10 AND b <> 'x';",
    "email me at john.doe@example.com or visit https://example.com/path?q=1&r=2",
    "日本語のテキストです。東京は大きな都市です。",
    "中文文本测试：北京是中国的首都。数字123和字母abc混合。",
    "한국어 텍스트 테스트입니다. 서울은 대한민국의 수도입니다.",
    "Привет, мир! Это тестовое предложение на русском языке.",
    "مرحبا بالعالم! هذه جملة اختبار باللغة العربية.",
    "שלום עולם! משפט בדיקה בעברית.",
    "नमस्ते दुनिया! यह हिंदी में एक परीक्षण वाक्य है।",
    "Ελληνικά: Καλημέρα κόσμε!",
    "café naïve résumé Zürich Straße",
    "e\u0301 (combining acute) and o\u0308 (combining diaeresis) marks",
    "emoji 😀 test 🚀🔥 and 👨‍👩‍👧 family",
    "Mixed: abc123def456 x1y2 ABC-123_def",
    "Punct!!! ??? ... --- === +++ *** ///",
    "(parens) [brackets] {braces} <angles> \"quotes\" 'single'",
    "spaces  between   words    here",
    "\n\n\n",
    "   ",
    "a\nb\nc",
    "word.\nNext",
    "12345678901234567890",
    "x = -1.5e-10; y = +3E+7;",
    "UPPER lower MiXeD cAsE wOrDs",
    "The quick brown fox jumps over the lazy dog. THE QUICK BROWN FOX.",
    "trailing newline\n",
    "\ttab start",
    "ends with space ",
    "multiple\n\nblank\n\n\nlines",
    "Ünïcödé Nörmälïzätïön",
    "日本語 English 混合 mixed テキスト text",
    "1+1=2, 2*3=6; 10/2=5.",
    "C:\\Users\\name\\file.txt and /usr/local/bin",
    "<|im_start|>user\nhi<|im_end|>\n",
    "<think>\nreasoning\n</think>\n\nanswer",
]


def write_cases(model_dir: str) -> None:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    cases = [{"text": t, "ids": tok.encode(t, add_special_tokens=False).ids} for t in CASES]
    with open(CASES_PATH, "w") as f:
        json.dump(cases, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(cases)} cases to {CASES_PATH}")


def ds4_ids(ds4: str, gguf: str, text: str) -> list[int]:
    out = subprocess.run([ds4, "-m", gguf, "--cpu", "--raw", "--dump-tokens", "-p", text],
                         capture_output=True, text=True)
    ids = []
    for line in out.stdout.splitlines():
        m = re.match(r"^\s*(\d+)\s\s", line)
        if m:
            ids.append(int(m.group(1)))
    if out.returncode != 0 and not ids:
        sys.stderr.write(out.stderr[-2000:])
    return ids


def check_cases(ds4: str, gguf: str) -> int:
    cases = json.load(open(CASES_PATH))
    bad = 0
    for c in cases:
        got = ds4_ids(ds4, gguf, c["text"])
        ok = got == c["ids"]
        bad += not ok
        print(("ok  " if ok else "BAD ") + repr(c["text"])[:60])
        if not ok:
            print("   expected", c["ids"])
            print("   got     ", got)
    print(f"{len(cases) - bad}/{len(cases)} cases match")
    return 1 if bad else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--ds4", default="./ds4")
    ap.add_argument("--gguf")
    args = ap.parse_args()
    if args.write:
        write_cases(args.model)
    if args.check:
        sys.exit(check_cases(args.ds4, args.gguf))


if __name__ == "__main__":
    main()
