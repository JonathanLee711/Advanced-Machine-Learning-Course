# 项目完整性检查

检查日期：2026-05-21

## 当前结论

该文件夹已经可以作为课程/论文实验复现型机器学习项目提交和继续维护。项目具备数据、模型代码、训练评估入口、依赖清单、实验结果、图表和检查/说明文档。

## 已具备内容

- 数据文件：`data/combined_result2.0.xlsx`
- 代码入口：`src/run_experiments.py`
- 模型定义：`src/models.py`
- 依赖清单：`requirements.txt`
- 实验结果：`results/`
- 实验图表：`figures/`
- 论文/检查材料：`docs/`、`outputs/`
- 项目元数据：`project_manifest.json`、`project_requirements.json`

## 已核对事项

- Excel 数据范围为 `A1:AL46378`，共 38 列、46377 条数据记录。
- 目标列为 `叶丝膨胀干燥_HT入口叶丝含水率`，与代码和需求配置一致。
- 代码设置为历史窗口 16、预测未来 4 步，按行顺序做 70%/15%/15% 时序划分。
- `src/models.py` 和 `src/run_experiments.py` 通过 Python 语法编译检查。
- `project_manifest.json` 和 `project_requirements.json` 可正常解析。
- 原空目录 `checkpoints/` 已清理；需要保存模型权重时，运行 `--save-checkpoints` 会重新创建。

## 还可补充但不是当前必需

- 可复现环境锁定文件，例如 `environment.yml` 或带精确版本的 `requirements-lock.txt`。
- 原始数据来源说明、采集口径、字段单位和脱敏说明。
- 最优模型权重文件；只有在提交方要求直接推理或复现实验结果时才需要。
- 自动化测试或轻量 smoke test；当前项目已有语法检查和数据结构核对，但未配置测试框架。
- 许可证、引用格式、作者/联系方式等正式发布材料。

## 运行提示

当前项目不包含一键启动脚本。按 `README.txt` 手动创建虚拟环境并安装 `requirements.txt` 后，可运行：

```powershell
.venv\Scripts\python.exe src\run_experiments.py
```
