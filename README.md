# 抓取序列的 QUBO 建模：代码与论文

本仓库是论文《面向料箱拣选的抓取序列 QUBO 建模：单调编码、位姿联合优化与闭环评估》的配套材料，包含论文全文、全部代码、结果数据与图，可离线复现。

## 论文

- `paper/paper.md` —— 论文全文（Markdown）
- `paper/paper.docx` —— 同一份的 Word 版，便于批注

## 代码

三个实验目录，前两个是论文 §5.1–5.3 的静态实验，第三个是 §5.4 的闭环实验。**不改动彼此的源文件**，后两个通过 `PYTHONPATH` 复用第一个的几何与求解器。

```
code/seq2d/src/       单个抓取选择 → 有序抓取 → 位姿联合优化
code/scaleup/src/     规模扫描（至 1860 变量）与滚动重规划
code/realistic/src/   真实场景闭环：非凸形状、指杆通道、噪声、失败、清障
code/*/results/       各组实验的 CSV
code/*/figures/       各组实验的图
code/*/NOTES.md       各实验的详细实验记录（论文写作时的原始笔记）
```

### 运行

Python 3.12，依赖见 `requirements.txt`：

```bash
pip install -r requirements.txt
```

**1. 编码验证（必须先跑）**

```bash
PYTHONPATH=code/seq2d/src python code/seq2d/src/_check_qubo.py     # 序贯编码，六例
PYTHONPATH=code/seq2d/src python code/seq2d/src/_check_cand.py     # 位姿编码，五例
```

两套编码都用穷举与全排列暴力解核对；不通过则后面的数字没有意义。

**2. 静态基准（论文表 1、表 3）**

```bash
PYTHONPATH=code/seq2d/src python code/seq2d/src/bench.py --seeds 3
PYTHONPATH=code/seq2d/src python code/scaleup/src/bench_scale.py --t-frac 0.5
```

**3. 位姿（论文表 2）**

```bash
PYTHONPATH=code/seq2d/src python code/seq2d/src/bench_cand.py
PYTHONPATH=code/seq2d/src python code/seq2d/src/bench_refine.py
```

**4. 闭环（论文表 4）**

```bash
PYTHONPATH=code/seq2d/src python code/scaleup/src/bench_rolling.py --seeds 5
PYTHONPATH=code/seq2d/src:code/realistic/src python code/realistic/src/bench2.py --objects 24 40 60 --seeds 3
```

Windows 下把 `PYTHONPATH` 换成：

```powershell
$env:PYTHONPATH = "code/seq2d/src;code/realistic/src"
```

**5. 单场景演示（不需要任何求解器依赖，只要 numpy）**

```bash
PYTHONPATH=code/seq2d/src python code/seq2d/src/demo.py
```

输出一段抓取序列，并验证"可行解的能量等于负收益"。

## 依赖

`numpy`、`pulp`（内含 HiGHS 绑定需 `highspy`）、`dimod`、`dwave-samplers`、`Pillow`。可选：`dwave-ocean-sdk`（连真机时）。

## 结果概览

| 实验 | 规模 | 主要结论 |
|---|---|---|
| 静态 | 30–360 变量，39 实例 | 精确解 0.01–0.43 s 最优；贪心差 0.83%；模拟退火差 2.11%（慢 2–3 倍）；路径积分模拟差 3.99%（慢 100–300 倍，4.8% 采样违反约束） |
| 位姿 | 15 实例 | 位姿与顺序联合优化 +9.8%（6/15 有效，最大 +64.7%）；局部精修再 +1.7% |
| 规模 | 45–1860 变量 | 差距 0→3.08%，精确解 10.8→1941 ms，贪心 0.010→0.146 ms |
| 闭环（简化） | 24–50 物体 | 开环清空率 42%，重规划 100%；三种规划器结果逐位相同 |
| 闭环（真实） | 24–60 物体 | 清单完成率：贪心 47%、MILP 75%、退火 75% |

量子退火的全部数字来自 `dwave-samplers` 的 CPU 模拟（路径积分蒙特卡洛），**未使用量子硬件**。`solvers.anneal_hardware` 提供了 D-Wave Leap 接口（需 `DWAVE_API_TOKEN`），本仓库不包含真机实验结果。

## 引用

若使用本仓库的编码或数据，请引用：

```bibtex
@misc{graspqubo2026,
  title  = {QUBO Modeling of Grasp Sequences: Monotone Encoding, Pose-Order Joint Optimization, and Closed-Loop Evaluation},
  author = {Wu, Yudu},
  year   = {2026},
  note   = {Code and paper, https://github.com/<user>/grasp-qubo}
}
```
