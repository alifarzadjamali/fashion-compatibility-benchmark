"""IVC prose counts. Run with Python 3 from any directory."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def expand(source):
    return re.sub(r"\\input\{([^}]+)\}", lambda m: expand((ROOT / (m[1] + ".tex")).read_text(encoding="utf8")), source)


def count(source, numbers):
    source = re.sub(r"(?<!\\)%[^\n]*", "", source)
    source = re.sub(r"\\begin\{(table\*?|figure\*?|equation\*?|align\*?)\}.*?\\end\{\1\}", "", source, flags=re.S)
    source = re.sub(r"^\\paperfigure\{.*$", "", source, flags=re.M)
    source = re.sub(r"\\(?:begin|end|label|ref|cite|corref)\{[^}]*\}", "", source)
    source = re.sub(r"\\(?:author|affiliation|cortext)\[[^]]*\]", "", source)
    source = re.sub(r"\b(?:organization|addressline|city|postcode|country)=", "", source)
    source = re.sub(r"\\res[A-Za-z]+", lambda m: numbers[m[0][1:]], source)
    source = re.sub(r"\$[^$]+\$", " FORMULA ", source)
    source = re.sub(r"\\(?:url|path|ead)\{[^}]*\}", " ADDRESS ", source)
    source = re.sub(r"\\[A-Za-z]+\*?", " ", source).replace("~", " ")
    return len(re.findall(r"\d+(?:[.,]\d+)+|[^\W_]+(?:[-'’][^\W_]+)*", source))


if __name__ == "__main__":
    main = (ROOT / "main.tex").read_text(encoding="utf8")
    source = expand(main.split(r"\begin{document}", 1)[1].split(r"\bibliographystyle", 1)[0])
    numbers = {}
    for filename in ("numbers.tex", "numbers_extra.tex"):
        numbers.update(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", (ROOT / filename).read_text(encoding="utf8")))
    conclusion_end = source.index(r"\section*{", source.index(r"\section{Conclusion}"))
    abstract = source.split(r"\begin{abstract}")[1].split(r"\end{abstract}")[0]
    print("Abstract:", count(abstract, numbers))
    print("Title through Conclusion:", count(source[:conclusion_end], numbers))
    print("Title through end matter:", count(source, numbers))
