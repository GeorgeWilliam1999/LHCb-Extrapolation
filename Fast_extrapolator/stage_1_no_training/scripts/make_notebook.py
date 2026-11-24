"""Write and run the notebook of stage 1: it loads the tables in results/ and
shows them with the figures. The notebook holds no number of its own.

    make_notebook.py      writes stage_1_no_training/stage_1_no_training.ipynb
"""
from __future__ import annotations

import os

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CELLS = [
    ("markdown", "# Stage 1: what is possible with no training\n\n"
                 "Every table is read from `results/`; every figure is made by `scripts/analysis.py`. "
                 "The README says which script writes which file."),
    ("code", "import pandas as pd\nfrom IPython.display import Image, display\n"
             "pd.set_option('display.max_rows', 400)\npd.set_option('display.max_colwidth', 90)\n"
             "pd.set_option('display.float_format', lambda v: '%.4g' % v)\n"
             "frontier = pd.read_csv('results/classical_frontier.csv')\n"
             "cols = ['label', 'precision', 'field_reads_mean', 'ns_lowest', 'cold_ns_median',\n"
             "        'radial_median_um', 'radial_95_um', 'worst_band_median_um']"),
    ("markdown", "## 1. The classical frontier"),
    ("code", "display(Image('figures/frontier.png'))"),
    ("code", "pd.read_csv('results/cheapest_methods.csv')"),
    ("markdown", "## 2. The error of a fixed-step integrator and the grid of the map"),
    ("code", "display(Image('figures/error_against_the_number_of_steps.png'))"),
    ("code", "scan = pd.read_csv('results/step_length_scan.csv')\n"
             "equal = scan[scan.family == 'equal steps']\n"
             "equal.pivot_table(index='steps', columns=['map_of_the_field', 'sample'],\n"
             "                  values=['radial_median_um', 'radial_95_um'])"),
    ("code", "display(Image('figures/shifted_planes.png'))"),
    ("code", "shifted = pd.read_csv('results/shifted_planes.csv')\n"
             "shifted.pivot_table(index=['first_plane_mm', 'last_plane_mm'], columns='label',\n"
             "                    values='radial_median_um')"),
    ("markdown", "## 3. Field reads and time"),
    ("code", "both = frontier[frontier.family.isin(['RK4, equal steps', 'Nystrom, equal steps'])]\n"
             "both.pivot_table(index='steps', columns=['map_of_the_field', 'family'],\n"
             "                 values=['field_reads_mean', 'ns_lowest'])"),
    ("markdown", "## 4. The LHCb integrators at other tolerances"),
    ("code", "frontier[frontier.family.isin(['HLT2 adaptive Nystrom', 'HLT2 adaptive Cash-Karp',\n"
             "                               'Allen adaptive Nystrom'])][['map_of_the_field'] + cols]"),
    ("markdown", "## 5. The two readings of the map"),
    ("code", "two = pd.read_csv('results/two_readings_of_the_map.csv')\n"
             "bands = [c for c in two.columns if c.startswith('radial_median_um_') and c.endswith('_gev')]\n"
             "two[['label', 'field_reads_mean', 'radial_median_um', 'radial_95_um'] + bands]"),
    ("markdown", "## 6. Steps of unequal length"),
    ("code", "display(pd.read_csv('results/chosen_plans.csv'))\n"
             "pd.read_csv('results/cost_of_a_step_along_z.csv').pivot_table(\n"
             "    index='start_of_the_step_mm', columns='map', values='mean_contribution_to_the_final_error_um')"),
    ("markdown", "## 7. The floor of each layout"),
    ("code", "display(Image('figures/floor_of_the_layouts.png'))"),
    ("code", "floor = pd.read_csv('results/floor_of_the_layouts.csv')\n"
             "floor[['map_of_the_field', 'steps', 'stages', 'radial_median_um', 'radial_95_um',\n"
             "       'worst_band_median_um', 'converged']]"),
    ("markdown", "## 8. What a chained network would cost at target A"),
    ("code", "pd.read_csv('results/what_a_chained_network_costs.csv')"),
    ("markdown", "## 9. Every method of the stage"),
    ("code", "frontier.sort_values(['map_of_the_field', 'ns_lowest'])[['map_of_the_field'] + cols]"),
]


def main():
    notebook = nbformat.v4.new_notebook()
    for kind, source in CELLS:
        notebook.cells.append(nbformat.v4.new_markdown_cell(source) if kind == "markdown"
                              else nbformat.v4.new_code_cell(source))
    ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(
        notebook, {"metadata": {"path": HERE}})
    with open(os.path.join(HERE, "stage_1_no_training.ipynb"), "w") as handle:
        nbformat.write(notebook, handle)


if __name__ == "__main__":
    main()
