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

9. Code Usage Statement
The code in this repository (including but not limited to all .py, .ipynb files) is provided solely for learning and demonstration purposes.
Without explicit written permission from the author, no individual or organization may:

Republish, copy, or distribute this code;

Use this code for any commercial purpose;

Cite or use this code in academic papers, reports, or other publications (except for private study notes that are not publicly shared);

Modify and re-release this code.

Dataset Usage Statement
The dataset in this repository (including raw data, processed data, sample data, etc.) is strictly prohibited from any form of access, download, copying, use, distribution, or modification.
The dataset is intended solely for the author’s local use and is not open to any third party. Even if permission is granted to use the code, that does not imply permission to use the dataset.

代码使用声明
本仓库中的代码（包括但不限于所有 .py、.ipynb 文件）仅供学习和展示使用。
未经作者明确书面许可，任何人或组织不得：

转载、复制、分发本代码；

将本代码用于任何商业用途；

在学术论文、报告或其他出版物中引用或使用本代码（仅限个人学习笔记中的非公开使用除外）；

基于本代码进行修改后重新发布。

数据集使用声明
本仓库中的数据集（包括原始数据、处理后的数据、样本数据等）严格禁止任何形式的访问、下载、复制、使用、分发或修改。
数据集仅供作者本人本地使用，不对任何第三方开放。即使获得了代码的使用授权，也不意味着获得了数据集的使用权。