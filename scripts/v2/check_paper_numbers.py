"""Assert manuscript numbers come from generated artifacts (HANDOFF §5 rule 5, §7.9).

Checks: (1) the generated `numbers.tex` matches `numbers.json` regenerated from the
results; (2) every `\\res...` macro used in the given .tex files is defined;
(3) no generated value is hand-typed as a literal number in the manuscript body
(reported as violations unless the line carries `% literal-ok`).
Exit status 1 on any violation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=Path, required=True, help="make_tables_v2 output dir")
    parser.add_argument("--extra", type=Path, action="append", default=[],
                        help="additional generated macro JSON (e.g. numbers_extra.json from "
                             "make_assets_v2.py); its .tex twin is sync-checked too")
    parser.add_argument("tex", type=Path, nargs="*", help="manuscript .tex files to check")
    args = parser.parse_args()
    pattern = r"\\newcommand\{\\(res\w+)\}\{([^}]*)\}"
    numbers = json.loads((args.generated / "numbers.json").read_text())
    defined = dict(re.findall(pattern, (args.generated / "numbers.tex").read_text()))
    violations = []
    if defined != numbers:
        violations.append("numbers.tex is out of sync with numbers.json; regenerate")
    for extra in args.extra:
        values = json.loads(extra.read_text())
        if dict(re.findall(pattern, extra.with_suffix(".tex").read_text())) != values:
            violations.append(f"{extra.with_suffix('.tex')} is out of sync with {extra}; regenerate")
        if clash := sorted(set(values) & set(numbers)):
            violations.append(f"{extra}: macros already defined elsewhere: {clash[:5]}")
        numbers = {**numbers, **values}
    values = {v for v in numbers.values() if len(v.replace(".", "")) >= 3}
    for path in args.tex:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            body = line.split("%", 1)[0] if "% literal-ok" not in line else ""
            for name in re.findall(r"\\(res[A-Z]\w*)", body):
                if name not in numbers:
                    violations.append(f"{path}:{number}: undefined macro \\{name}")
            for literal in re.findall(r"(?<![\w.])\d+\.\d+(?![\w.])", body):
                if literal in values:
                    violations.append(f"{path}:{number}: hand-typed generated value {literal}")
    for violation in violations:
        print(violation)
    print(f"{len(numbers)} generated numbers; {len(violations)} violations")
    sys.exit(1 if violations else 0)


if __name__ == "__main__":
    main()
