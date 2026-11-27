# RepBench-Fashion

## Benchmark

RepBench-Fashion evaluates 15 frozen image encoders on outfit compatibility and fill-in-the-blank completion across Polyvore, IQON3000 and POG. A100 provides an external comparison with expert and crowd choices.

**Platform.** The pipeline targets Linux (validated on an NVIDIA DGX Spark, aarch64). Six tests are Linux-only and are skipped on Windows.

## Install

Install [uv](https://docs.astral.sh/uv/), then run these commands from the repository root.

```sh
uv sync --locked --extra dev
uv run pytest
```

## Reproduce

The [frozen protocol](configs/v2_protocol.yaml) specifies the checkpoints, splits and evaluation settings. The [v2 scripts](scripts/v2/) include the Linux GPU pipeline (`run_full_gpu.sh`) for extraction, training and profiling, and the CPU pipeline (`run_full_cpu.sh`) for probes and statistics. The [decision log](docs/v2_decisions.md) records deviations and follow-up runs.

## Results

[research/v2/](research/v2/) contains the frozen split reports, readiness checks and compact results. The [results directory](research/v2/results/) includes the metrics and paired comparisons used in the paper.

## Paper

[paper/](paper/) contains the Image and Vision Computing submission source, [manuscript](paper/main.pdf) and [supplement](paper/supplementary.pdf). To rebuild both documents, run `latexmk -pdf main.tex supplementary.tex` from that directory. The `v2-freeze` tag identifies the frozen protocol. The `v2.0-ivc-submission` tag identifies the manuscript source.

## Data

Dataset images are not redistributed. Obtain them from the original Polyvore Outfits, IQON3000, POG and A100 providers under their respective terms.

## Citation

Use the repository citation metadata in [CITATION.cff](CITATION.cff).

## License

The benchmark code is available under the [Apache License 2.0](LICENSE).
