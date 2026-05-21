Moisture Prediction Project - Configured Runtime Package

1. Project Task
This package is configured for the paper/course experiment:
use loose-conditioning process parameters and historical HT inlet moisture
to predict the next 4 time steps of HT inlet cut-tobacco moisture.

2. Key Experiment Settings
- Source data: data/combined_result2.0.xlsx
- Historical window: 16
- Prediction horizon: 4
- Split: chronological 70% train, 15% validation, 15% test
- Models: MLP, CNN, LSTM, GRU, BiLSTM, CNN-LSTM, Proposed
- Ablations: Full Model, w/o Multi-scale CNN, w/o Attention, w/o Residual, w/o Auxiliary Loss

3. Folder Layout
- src/models.py: model definitions
- src/run_experiments.py: training, evaluation, metrics, and figures
- data/combined_result2.0.xlsx: source data copied from input_data for model
- results/: existing result files copied from the source project
- figures/: existing figure files copied from the source project
- outputs/: project check documents generated earlier
- checkpoints/: optional model checkpoints, created on demand when --save-checkpoints is used

4. Create Virtual Environment
Open PowerShell or cmd in this folder, then run:

python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt

No one-click setup/startup script is included. Create the environment manually
with the commands above.

5. Run Experiments
After installing dependencies:

.venv\Scripts\python.exe src\run_experiments.py

Optional quick comparison-only run:

.venv\Scripts\python.exe src\run_experiments.py --skip-ablation

Optional CPU run:

.venv\Scripts\python.exe src\run_experiments.py --device cpu

Optional checkpoint saving:

.venv\Scripts\python.exe src\run_experiments.py --save-checkpoints

6. Output Locations
The script writes tables to results/ and figures to figures/.
The default data path in run_experiments.py is already satisfied by this package:
data/combined_result2.0.xlsx

7. Notes
The original data are collected by time step in row order. This package therefore
uses row order as the implicit chronological index, consistent with the project note.

8. Completeness Check
See docs/project_completeness_check.md for the current project completeness status,
known verification results, and optional items still needed for formal release.
