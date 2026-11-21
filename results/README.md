# Result files

Everything the tables and figures in [`report/`](../report/) are computed from: for every benchmark folder the
instance list, the mapping from instances to formulas, the formulas and the unedited result files of the
verifier runs. [`tests/test_results.py`](../tests/test_results.py) regenerates every table in `report/` from
these folders and checks that it matches byte for byte.

Not included: the networks (`onnx/`) and properties (`vnnlib/`), about 650 MB, which the same seed rebuilds
byte for byte (see [Rebuilding the networks](#rebuilding-the-networks)); the known solutions (`solutions/`),
which the generator recomputes; and the tool logs.

## Files per folder

| file | content |
|---|---|
| `instances.csv` | VNN-COMP instance list: `onnx,vnnlib,timeout` per line |
| `metadata.csv` | per instance: formula, family, encoding, expected answer (`sat` = counterexample exists), sizes, threshold, right-hand side in the VNN-LIB file, number of output constraints (empty = 1), SAT-solver time; in the `long-*` folders also `source`, the instance with 60 s |
| `formulas.csv` | per formula: generator parameters and seed, answer and where it comes from, CaDiCaL's status and time |
| `cnf/` | the formulas in DIMACS format (not in the `satrelu-*` folders, see below) |
| `results/<tool>.csv` | one row per instance and run: the tool's answer, `verdict` (`correct`, `wrong`, `timeout`, ...), run time in seconds, and for every counterexample the independent check (`cex_check`) |
| `results/abcrown-mip-rerun.csv` | diagnostic re-runs of the MIP runs that hung before Gurobi started; the tables count the first run |

The column meanings are documented in [`run.py`](../src/relu_sat_bench/run.py) (`RESULT_COLUMNS`) and
[`export.py`](../src/relu_sat_bench/export.py) (`METADATA_COLUMNS`). Tools: `marabou` (Marabou 2.0.0),
`abcrown-bab` (α,β-CROWN, branch-and-bound) and `abcrown-mip` (α,β-CROWN, MIP mode with Gurobi).

## Folders

| folder | instances | encoding, property | tools | built with |
|---|---|---|---|---|
| `final` | 213 | compact | Marabou, branch-and-bound | `generate_properties.py 1` |
| `final-shifted` | 213 | compact, threshold in the network | MIP | `generate_properties.py 1 --threshold-in-network` |
| `scaling-shifted` | 65 | compact, threshold in the network | MIP | `generate_properties.py 1 --scaling --threshold-in-network` |
| `scaling-two` | 65 | compact, two constraints | MIP | `generate_properties.py 1 --scaling --two-constraints` |
| `scaling-ncb` | 65 | `sat_relu` (neurocodebench) | MIP | `generate_properties.py 1 --scaling --neurocodebench` |
| `final-ncb` | 209 | `sat_relu` | all three | `generate_properties.py 1 --neurocodebench` |
| `large-ncb` | 45 | `sat_relu` | MIP | `generate_properties.py 1 --large --neurocodebench` |
| `sample-ncb` | 57 | `sat_relu` | all three | `generate_properties.py 1 --sample --neurocodebench` |
| `satrelu-compact` | 98 | compact | Marabou, branch-and-bound | `relu_sat_bench.satrelu` |
| `satrelu-compact-shifted` | 98 | compact, threshold in the network | MIP | `relu_sat_bench.satrelu --threshold-in-network` |
| `satrelu-compact-two` | 98 | compact, two constraints | MIP | `relu_sat_bench.satrelu --two-constraints` |
| `satrelu-paper` | 98 | paper | Marabou, branch-and-bound | `relu_sat_bench.satrelu --encoding paper` |
| `satrelu-paper-shifted` | 98 | paper, threshold in the network | MIP | `relu_sat_bench.satrelu --encoding paper --threshold-in-network` |
| `satrelu-ncb` | 98 | `sat_relu` | all three | `relu_sat_bench.satrelu --encoding neurocodebench` |
| `long-marabou` | 20 | compact, 600 s | Marabou | `relu_sat_bench.subset --bench final --tool marabou` |
| `long-bab` | 22 | compact, 600 s | branch-and-bound | `relu_sat_bench.subset --bench final --tool abcrown-bab` |
| `long-mip` | 12 | compact, threshold in the network, 600 s | MIP | `relu_sat_bench.subset --bench final-shifted scaling-shifted --tool abcrown-mip` |
| `long2-bab` | 13 | compact, 600 s | branch-and-bound | as `long-bab`, with `--skip 1 --answer unsat` |
| `long2-mip` | 9 | compact, threshold in the network, 600 s | MIP | as `long-mip`, with `--skip 1 --answer unsat` |

All timeouts are 60 s except in the `long-*` folders (600 s). The `--threshold-in-network` folders hold the
same networks with the threshold moved into the last bias, because α,β-CROWN's MIP mode mishandles the
right-hand side of single-constraint properties (see the main README).

**`satrelu-*`.** The 98 formulas read back from the networks of the VNN-COMP 2025 benchmark `sat_relu`. Its
repository states no license, so these folders contain only our metadata and results: no formulas and no
networks. `python -m relu_sat_bench.satrelu --source <folder> --out <folder> [options]` rebuilds them from a
download of `benchmarks/sat_relu` from `VNN-COMP/vnncomp2025_benchmarks`.

## Regenerating the report

```bash
pip install -e ".[report]"
python -m relu_sat_bench.evaluate --bench results/final results/final-shifted \
    --extension results/scaling-shifted --out report
python -m relu_sat_bench.evaluate --bench results/final-ncb --out report/ncb
python -m relu_sat_bench.evaluate --bench results/large-ncb --out report/large-ncb --no-figures
python -m relu_sat_bench.evaluate --compare results/satrelu-compact results/satrelu-compact-shifted \
    results/satrelu-compact-two results/satrelu-paper results/satrelu-paper-shifted results/satrelu-ncb \
    --name satrelu --out report
python -m relu_sat_bench.evaluate --compare results/final results/final-shifted results/sample-ncb \
    --name sample --out report
python -m relu_sat_bench.evaluate --compare results/scaling-shifted results/scaling-two results/scaling-ncb \
    --name scaling --out report
python -m relu_sat_bench.evaluate --compare results/final results/final-shifted results/final-ncb \
    --name overlap --out report
python -m relu_sat_bench.evaluate --long results/long-marabou results/long-bab results/long-mip \
    results/long2-bab results/long2-mip --out report
```

The tables come out identical; the SVG figures show the same data, but their bytes depend on the matplotlib
version.

## Rebuilding the networks

`python generate_properties.py 1 <folder> [options]` with the options of the table writes the folder
again, with `onnx/`, `vnnlib/` and `solutions/`, under the same file names; the same seed gives
byte-identical files. The `long-*` folders select by the 60 s results: copy `results/<folder>/results/` into
the rebuilt `final`, `final-shifted` and `scaling-shifted` folders, then run `python -m relu_sat_bench.subset`
with the options of the table and `--out <folder>`.

## License

[CC BY 4.0](../LICENSE-DATA), like all benchmark data in this repository.
