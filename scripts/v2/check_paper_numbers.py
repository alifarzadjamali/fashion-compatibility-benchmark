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
    parser.add_argument("tex", type=Path, nargs="*", help="manuscript .tex files to check")
    args = parser.parse_args()
    numbers = json.loads((args.generated / "numbers.json").read_text())
    defined = dict(re.findall(r"\\newcommand\{\\(res\w+)\}\{([^}]*)\}",
                              (args.generated / "numbers.tex").read_text()))
    violations = []
    if defined != numbers:
        violations.append("numbers.tex is out of sync with numbers.json; regenerate")
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
