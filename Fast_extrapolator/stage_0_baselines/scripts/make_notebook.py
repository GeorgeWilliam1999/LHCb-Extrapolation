"""Write and run the notebook of stage 0: it loads the tables in results/ and
shows them with the figures. The notebook holds no number of its own.

    make_notebook.py      writes stage_0_baselines/stage_0_baselines.ipynb
"""
from __future__ import annotations

import os

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CELLS = [
    ("markdown", "# Stage 0: the compiled program and the LHCb baselines\n\n"
                 "Every table is read from `results/`; every figure is made by `scripts/plots.py`. "
                 "The README says which script writes which file."),
    ("code", "import pandas as pd\nfrom IPython.display import Image, display\n"
             "pd.set_option('display.max_rows', 200)\npd.set_option('display.max_colwidth', 90)\n"
             "pd.set_option('display.float_format', lambda v: '%.4g' % v)\n"
             "methods = pd.read_csv('results/methods.csv')"),
    ("markdown", "## 1. The gate: the compiled methods against Python"),
    ("code", "pd.read_csv('results/gate_compiled_against_python.csv')"),
    ("markdown", "## 2. The copy of the HLT2 integrator against the LHCb software\n\n"
                 "The grid of 1,331 states of the Rec test; the map the test used shows itself."),
    ("code", "display(pd.read_csv('results/rec_copy_against_the_rec_reference.csv'))\n"
             "display(pd.read_csv('../lhcb_software/rec_extrapolator_test/output/summary_stbc-i2.csv'))"),
    ("markdown", "## 3. The LHCb methods on our tracks"),
    ("code", "cols = ['label', 'precision', 'field_reads_mean', 'field_reads_largest', 'ns_lowest', 'ns_median',\n"
             "        'cold_ns_median', 'radial_median_um', 'radial_95_um', 'radial_99_um', 'fraction_above_100_um',\n"
             "        'worst_band_median_um', 'flat_radial_median_um', 'flat_radial_95_um']\n"
             "methods[methods.family == 'LHCb'][cols]"),
    ("code", "bands = [c for c in methods.columns if c.startswith('radial_median_um_') and c.endswith('_gev')]\n"
             "methods[methods.family == 'LHCb'][['label'] + bands]"),
    ("markdown", "The parametrised step of HLT1 against two field maps, and the check of our stored reference:"),
    ("code", "p = pd.read_csv('results/parametrised_step.csv')\n"
             "p[['method', 'reference_map', 'radial_median_um', 'radial_95_um', 'worst_band_median_um', 'x_signed_median_um']]"),
    ("markdown", "## 4. The unit costs: one field read and one network call"),
    ("code", "display(pd.read_csv('results/micro_field.csv'))\n"
             "m = pd.read_csv('results/micro_mlp.csv')\n"
             "m.pivot_table(index=['depth', 'width'], columns=['precision', 'activation'], values='ns_lowest')"),
    ("code", "display(Image('figures/cost_of_a_network_call.png'))"),
    ("markdown", "## 5. Every method: time and error"),
    ("code", "display(Image('figures/time_against_error.png'))"),
    ("code", "methods[methods.family == 'classical'][cols]"),
    ("code", "methods['precision and tanh'] = methods['precision'] + methods['tanh'].fillna('').map(lambda t: ', fast tanh' if t == 'fast' else '')\n"
             "methods[methods.family == 'network'][['label', 'precision and tanh', 'network_calls', 'multiply_adds_a_call'] + cols[2:]]"),
    ("markdown", "## 6. The error by momentum"),
    ("code", "display(Image('figures/error_by_momentum.png'))"),
    ("markdown", "## 7. The transport matrix of the copy against finite differences"),
    ("code", "pd.read_csv('results/transport_matrix_of_the_copy.csv')"),
]


def main():
    notebook = nbformat.v4.new_notebook()
    for kind, source in CELLS:
        notebook.cells.append(nbformat.v4.new_markdown_cell(source) if kind == "markdown"
                              else nbformat.v4.new_code_cell(source))
    ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(
        notebook, {"metadata": {"path": HERE}})
    with open(os.path.join(HERE, "stage_0_baselines.ipynb"), "w") as handle:
        nbformat.write(notebook, handle)


if __name__ == "__main__":
    main()
