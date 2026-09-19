# Sim-CAMD

Computer-aided molecular design (CAMD) desktop software for **solvent** and **refrigerant** molecules. The interface supports group selection, molecular generation, property prediction, and screening.

## Requirements

| Item | Version / note |
|---|---|
| OS | Windows 10/11, 64-bit |
| Python | **3.9.x, 64-bit** (the `.pyd` modules will not load on 3.10+ or 32-bit Python) |
| GAMS | Installed locally, with the **Python API** available to this Python |
| Solver | **BARON** (called through Pyomo + GAMS) |

Core logic is shipped as compiled extensions:

- `functions_base.cp39-win_amd64.pyd`
- `ProductDesignFunction.cp39-win_amd64.pyd`

These files only work with **CPython 3.9 on 64-bit Windows**.

## Installation

1. Install [Python 3.9 64-bit](https://www.python.org/downloads/) (or Anaconda/Miniconda with a `python=3.9` environment).

2. Create and activate a virtual environment in the project folder (optional but recommended):

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Install GAMS and BARON. Make sure the GAMS Python API can be imported:

```bat
python -c "from gams import GamsWorkspace; print('GAMS OK')"
```

If this fails, add the GAMS `python` / `api` directories to `PYTHONPATH`, or use the Python that GAMS documents for your GAMS version.

`gams` is **not** listed in `requirements.txt` because it comes from the GAMS installation, not from PyPI.

## Run

From the project root (the folder that contains `mainwindow.py`):

```bat
python mainwindow.py
```

Keep the working directory at the project root. Data and result paths are relative (`data/…`, `results/…`, `UI/…`).

## Typical workflow

1. **Product type** — choose Solvent or Refrigerant.
2. **Targets** — select property constraints / objective.
3. **Step 2: Molecular generation** — select groups, set structural bounds, click **Generate molecules**.
4. **Property prediction** — choose a model type for each property and run prediction.
5. **Screening** — set operating conditions if needed, then generate solutions.

Generated tables are written under `results/temp file/`:

- `generated_molecules.xlsx`
- `property_prediction.xlsx`
- `solutions.xlsx`

GAMS/BARON working files (`.gms`, `.gdx`, `.lst`) appear under `results/` during a solve.

## Repository layout

```text
.
├── .gitignore
├── mainwindow.py
├── functions_base.cp39-win_amd64.pyd
├── ProductDesignFunction.cp39-win_amd64.pyd
├── requirements.txt
├── UI/
│   └── mainwindow_ui.ui
├── data/
│   └── stored/                 # group tables, GC/GPR models, training data
└── results/
    ├── .gitkeep
    └── temp file/
        └── .gitkeep            # folder is required; Excel outputs are gitignored
```

## Notes

- **Generate molecules** needs a working GAMS + BARON license. If BARON is missing or the model is infeasible, the interface should show a notice or error instead of a molecule table.
- Do not mix a different `scikit-learn` version than `requirements.txt`. The files in `data/stored/general GPR model/*.pkl` were saved with scikit-learn 1.6.x.

## License / citation

Add license and citation information here if you publish this repository with a paper.
