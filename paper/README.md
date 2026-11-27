# IVC manuscript

This folder contains the Image and Vision Computing submission source. Run the following command from this directory to build the manuscript and supplement.

```sh
latexmk -pdf main.tex supplementary.tex
```

`main.tex` includes the manuscript section files. Run `python count_words_ivc.py` here to check the IVC word limit.

The [v2 result files](../research/v2/results/) supply the reported values. [make_tables_v2.py](../scripts/v2/make_tables_v2.py) generates `numbers.tex` and `table_main.tex`. [make_assets_v2.py](../scripts/v2/make_assets_v2.py) generates `numbers_extra.tex`, `table_efficiency.tex` and the qualitative FITB figure. [typeset_evidence.py](typeset_evidence.py) generates `evidence_views.tex` from the completed reports. The `fig_*.pdf` files are generated figures, including the plots from [make_figures_v2.py](../scripts/v2/make_figures_v2.py).
