# 面向料箱拣选的抓取序列 QUBO 建模：单调编码、位姿联合优化与闭环评估

**摘要**　料箱拣选的核心是一串离散决策：抓哪一个、用什么位姿抓、先抓谁、抓空了怎么办。本文把这串决策写成二次无约束二值优化（QUBO），并给出一个可证明单调的编码：每个物体占一行、每一步占一列、另加一列表示"不抓"，前置关系只以正耦合出现，于是**任意可行解的能量恒等于负的收益**，罚权重只需大于某一下界即可，无需精细调参。在此编码上，我们把抓取位姿也纳入决策变量，使"低分但更好拿"的位姿可以被选中；在有位姿冲突的拥挤场景中，联合优化比固定最优位姿平均提升 9.8%（15 个实例中 6 个有效，最大 64.7%），在离散位姿邻域上再做局部精修可再得 1.7%。我们用精确整数规划、贪心剥离、模拟退火与路径积分（量子退火模拟）四种求解器在 39 个静态实例、15 个位姿实例、8 个规模实例（最大 1860 个变量）与两组闭环任务上做了同机同码对比。结论与直觉相反但清楚：在 30–400 个变量的区间，精确解在 0.01–2 s 内给出可证明最优，两种退火均更慢且更差；贪心剥离平均只差 0.8%，但当场景需要"先清障才能拿到清单目标"时，其完成率从 75% 掉到 47%，而会提前规划清障的方法维持在 75%。所有量子退火结果均为 CPU 上的路径积分模拟，本文不主张量子加速。代码、数据与图全部随文提供，可一键复现。

**关键词**　二次无约束二值优化；抓取序列；料箱拣选；量子退火；闭环重规划

## 1 引言

机器人从料箱里取物时，真正难的往往不是"怎么抓"，而是"先抓哪个"。夹爪需要一条干净的指间通道：若目标被上面的物体压住，就必须先把它移走；两个物体互相压住时谁也拿不到，只能放弃；而单次任务的动作预算有限，多抓一个挡路的非目标物体，就要少抓一个真正的目标。这类决策天然是离散、强耦合、需要反复重算的。

二次无约束二值优化（QUBO）是这类问题的通用语言：Ising 模型能表达大量 NP 难问题 [1]，量子退火与门模型变分算法都以它为输入 [3]。近年来 QUBO 已被用于检测后处理 [6]、点云配准 [7,8]、特征选择 [9,10]、任务规划 [4] 与运动规划 [12,13]。抓取任务上，公开可检索的工作集中在**抓取检测**（以深度图回归抓取矩形或六维位姿，如 [14,15,17]），而把"抓取位姿与抓取次序"联合写成 QUBO 的工作，据我们在 arXiv 与主要会议论文库的检索（截至 2026 年 10 月，检索式为 `abs:"QUBO" AND abs:"grasp"`）尚为空白。这正是本文的出发点。

本文的贡献有三条。**其一，一个可证明单调的编码。** 常见写法是"若 i 在 q、且其前置在 p，则 p<q，否则罚"，但当前置物体根本没被选中时，这一对变量都不存在，惩罚项为空，于是"抓了被挡的物体却没抓挡它的"反而是最低能量解。我们改为物体级 one-hot 加一列"不抓"，前置项只取正耦合，并让"不抓"列参与前置判定，从而规避上述空惩罚；在可行解上能量恒等于负收益，权重只需足够大（定理 1）。该编码在 6 组小实例上与穷举及全排列暴力解逐一核对通过。**其二，位姿作为决策变量。** 每个物体保留若干候选位姿，阻塞关系按位姿分别计算；在拥挤场景中这使"换一个姿势绕开"成为可能，平均增益 9.8%。**其三，一次诚实的闭环评估。** 我们比较了"只规划一次"与"滚动重规划"，并进一步把场景做复杂（非凸形状、指杆扫掠通道、感知噪声、抓取失败、可推、按清单拣选）。结论是：**规划器之间的差异在简单场景里会被噪声和重规划频率淹没，只有当任务需要主动清障时才显现**——贪心 47% 对规划 75%。

本文不主张量子优势。所有"量子退火"数字来自 CPU 上的路径积分蒙特卡洛模拟，硬件一次未参与；在本文规模上，精确解已足够快，退火的嵌入、链断裂与多次采样开销无法摊销。我们把这一边界写在 §7。

## 2 相关工作

**QUBO 与 Ising。** QUBO 与 Ising 模型等价，NP 难问题的多项式规模编码可参见 [1]，工程侧的建模技巧与罚函数讨论见 [2,3]。求解侧有精确分支定界与整数规划、模拟退火、量子退火、以及门模型上的 QAOA。近年工作 [9,10] 把特征选择写成 QUBO 并在退火机上验证；[11] 比较了 D-Wave 混合求解器、数字退火器与经典 MIP，结论是混合求解器**未能稳定胜过**经典最优解；这与我们在抓取序列上的观察一致。

**视觉与机器人中的组合优化。** 非极大值抑制被写成 QUBO 并在 D-Wave 上验证 [5]，点云配准有成对的 Q-Match [8] 与迭代局部 QUBO 的 QuCOOP [7]；运动规划与模型预测控制也已有 QUBO 形式 [12,13]。这些工作共同说明：QUBO 的用武之地在于**问题天然离散且规模适中**的环节，而不是替代通用优化器。

**抓取。** 抓取检测以 Dex-Net [17]、GG-CNN [14]、GR-ConvNet [15] 为代表，输出抓取矩形或位姿及其质量分；真实数据基准有 Cornell 与 Jacquard [16]。这些方法给出"候选抓取及其分数"，但**如何在一个料箱里按顺序执行**通常交给启发式（按分数贪心、按区域分区）。本文补的正是这一层：把候选抓取、阻塞关系与动作预算统一到一个二次型里。

## 3 问题定义

设料箱内物体集合 $O=\{1..M\}$，物体 $o$ 的候选抓取集合 $G_o$（本文 $|G_o|=3$ 或 $6$）。抓取 $g\in G_o$ 由中心、开口宽度与朝向确定，其矩形为 $R(g)$。

**阻塞关系。** 对抓取 $g$ 定义两条：其一，若同堆中物体 $o'$ 的实体中心高于 $o$，则 $o'$ 挡 $o$；其二，若夹爪接近时两只指杆扫过的通道与 $o'$ 相交，则 $o'$ 挡 $g$。记 $\mathrm{blk}(o,g)\subseteq O$。若 $o' \in \mathrm{blk}(o,g)$，则只有在 $o'$ 已被移走之后，$g$ 才可执行。

**决策变量。** 每个物体 $o$ 选择"在第 $s$ 步抓某个候选 $c$"（$0\le s<T$）或"不抓"（$s=T$），即 $z_{o,c,s}\in\{0,1\}$，满足 $\sum_{c,s} z_{o,c,s}=1$。动作预算 $T$ 为单次规划步数，故变量数 $M\cdot|G_o|\cdot(T+1)$。

**目标。** 最大化被选抓取的分数之和 $\sum_{o,c} \mathrm{score}(o,c)\,\mathbb{1}[s<T]$，受(a)每物体恰一个状态、(b)每步至多一个抓取、(c)前置关系三条约束。可行的序列定义为：按步序执行时，每一步的抓取在所有前置物体均已被移除。

该问题是**带前置约束的选择与排序**问题；其判定版本包含"加权最大独立集 + 线性序"作为特例，属 NP 难。

## 4 方法

### 4.1 单调编码

把 (a)(b)(c) 写成罚项，能量为

$$E(z)= -\sum_{o,c}\sum_{s<T} \mathrm{score}(o,c)z_{o,c,s} + \lambda\sum_o\Big(\sum_{c,s} z_{o,c,s}-1\Big)^2 + \lambda\sum_{s<T}\sum_{(o,c)<(o',c')} z_{o,c,s}z_{o',c',s} + \lambda\sum_{(o,c)}\sum_{o'\in \mathrm{blk}(o,c)}\sum_{s=0}^{T-1}\sum_{t=s}^{T} z_{o,c,s}z_{o',c',t}. $$

第二项展开后只有 $-\lambda$ 的对角与 $+2\lambda$ 的成对项；第三项是"步独占"；第四项是前置，其中 $t$ 取到 $T$ 是**关键**：若前置物体被放弃（$t=T$），该项仍然存在并被罚，从而堵住了"前置为空则不罚"的漏洞。

**定理 1（单调性）** 设 $\lambda> \max_{o,c}\mathrm{score}(o,c)\cdot\max(1, T)$，则对任意满足 (a)(b)(c) 的 $z$，有 $E(z)=-\sum \mathrm{score}\cdot z$；且任何违反约束的 $z$ 的能量严格大于最优可行解的能量。

*证明要点*：可行解上后三项全为 0；任一违反项增加至少 $\lambda$，而放弃或重排至多损失 $T\max\mathrm{score}$，取 $\lambda$ 足够大使违反不划算。∎

实践中取 $\lambda=2T$，在全部验证用例中均满足定理条件。该性质带来两个工程好处：一是无需调参；二是**能量可直接读作负收益**，便于在不同求解器间比较解的质量。

### 4.2 位姿联合优化

把每个物体的候选位姿集合扩大为 $\{( \delta, \theta)\}$（沿长轴的三个偏移与两个旋转），阻塞关系按候选分别计算。于是"分数略低但通道更干净"的位姿可被选中。变量数乘以候选数，代价在 §6.2 量化。

在其上再加一层**局部精修**：对已选序列，在每个位姿的九个邻域（沿长轴 $\pm0.05,\pm0.10$ 倍长、角度 $\pm12°$）里搜索，只要不新增阻塞且分数更高就接受。精修不改变序列，故无需重解，代价是序列长度的九倍次几何评估。

### 4.3 求解器

**精确整数规划**：以 pulp 调用 HiGHS，直接对 (a)(b)(c) 建模（前置写成 $z_{o,c,s}\le\sum_{t<s}z_{o',c',t}$，$s<T$）。**贪心剥离**：每轮在"当前无阻塞"的抓取中取分数最高者；由于移除物体只会放松约束，该启发式在纯剥离任务上接近最优。**模拟退火**：dwave-samplers 的 `SimulatedAnnealingSampler`，20–40 次读、600–2000 次扫描。**路径积分（量子退火模拟）**：`PathIntegralAnnealingSampler`，Trotter 切片 6–8。**硬件接口**：`anneal_hardware` 通过 D-Wave Leap 提交，需 API token，记录 `qpu_access_time` 与 `chain_break_fraction`；本文所有数字均未使用该路径。

### 4.4 闭环

滚动时域：每轮用带噪观测重建场景 → 规划长度 $T$ 的窗口 → 只执行前 $H$ 步 → 成功则移除、失败则重试或推一下挡路的物体 → 重新观测。观测含位置高斯噪声、掩膜膨胀与漏检；执行成功率为适配度、被挡数、是否碰撞易碎物与质量的函数。基线是"只规划一次"的开环策略。

## 5 实验设置

**环境**：Windows，单进程 CPU（Intel 级桌面处理器），Python 3.12；所有求解器在同一进程、同一份数据上计时，报告墙钟中位数；量子退火的计时只含采样，不含 QUBO→BQM 转换（该转换在 $N=200$ 变量上约 20 ms，已单独测量，故本文对退火的时间是**偏乐观**的。

**场景**：(i) 静态 39 例——堆叠族 7 个规模 × 3 种子 + 散乱族 6 个规模 × 3 种子，变量 30–360；(ii) 位姿 15 例——拥挤场景重叠上限 0.62，每物体 4 个候选位姿；(iii) 规模 8 例——$N=9..60$，窗口 $T=N/2$，最大 1860 变量、10524 个非零耦合；(iv) 闭环两组——简化（凸矩形、干净观测）与真实（非凸形状、通道、噪声、失败、推、清单拣选），24/40/60 物体 × 3 种子。

**验证**：小实例穷举全部 $2^n$ 状态并与全排列暴力最优核对（$n=9,12,18,24$）；大实例以整数规划最优为参照并检查三条不变量（编码序列能量 $=-\text{收益}$、解码可还原同一序列、随机 400 个状态中"可行⇒等式成立、不可行⇒能量更高"）。单次与序贯两套编码共 11 组用例全部通过。

## 6 结果

### 6.1 静态：精确解赢，且按规模触发

| 求解器 | 平均差距 | 最大差距 | 平均耗时 | 不可行率 |
|---|---|---|---|---|
| MILP/HiGHS | 0.00% | 0.00% | 48.8 ms | 0% |
| 贪心剥离 | 0.83% | 6.46% | 0.0 ms（微秒级） | 0% |
| 模拟退火 | 2.11% | 9.68% | 115 ms | 1.6% |
| 路径积分（量子退火模拟） | 3.99% | 17.91% | 1472 ms | 4.8% |

三点值得强调。第一，差距随规模出现：变量数 $\le150$ 时贪心几乎不差，$M=24$（360 变量）才掉到 6.46%，故小规模实验容易得出"贪心足够"的错误结论。第二，那 6.46% 可定位到具体一步：贪心第一步抓分数 0.822 的物体，最优解抓 0.534 的一个——后者压着三个分数更高的物体，最优解愿意先付这步"开通费"。第三，路径积分模拟有约 5% 的采样直接违反 one-hot 或前置约束，可用性还需再打折。

### 6.2 位姿：联合优化与精修

| 设置 | 平均收益 | 说明 |
|---|---|---|
| 固定最优位姿、只排顺序 | 4.31 | 基线 |
| 位姿 + 顺序联合 | 4.73（+9.8%） | 15 例中 6 例提升，最大 +64.7% |
| 再加局部精修 | 4.81（+1.7%） | 15 例全部提升，0.3%–2.5% |

典型机制：某实例中物体 5 与 14 在最优位姿下互相压住（谁都拿不到），改用第二个候选位姿后 5 只压住物体 3，于是"3→5→14"打通，抓取数 3→5、收益 2.16→3.55。代价是变量数乘以候选数与 2–9 倍的精确解耗时。堆叠场景该增益为 0，因为那里由层级规则主导、位姿无语义——这也说明该方法适用于**松散堆叠/拥挤**而非**整齐码垛**。

### 6.3 规模：窗口必须随场景增长

| N | 变量数 | 贪心差距 | MILP | 贪心 |
|---|---|---|---|---|
| 9 | 45 | 0.00% | 10.8 ms | 0.010 ms |
| 21 | 231 | 0.49% | 48.7 ms | 0.023 ms |
| 40 | 840 | 1.83% | 634 ms | 0.118 ms |
| 60 | 1860 | 3.08% | 1 941 ms | 0.146 ms |

若把窗口固定成 6 去跑 60 个物体，问题退化为"从 60 挑 6"，前置约束不再咬合、差距在 0–3% 间乱跳。**窗口与场景匹配**是这类实验的必要条件，否则测到的不是算法差异而是参数设置。

### 6.4 闭环

**简化场景（凸矩形、干净观测、无清障需求）。** 只规划一次：清理率 42%；每三步重规划：100%。三种规划器的清理结果**逐位相同**（抓走数、尝试数、总收益完全一致），耗时相差三个数量级（贪心 0.4 ms、精确 353 ms、退火 1269 ms）。此时规划器的选择被重规划频率淹没——杠杆在"多久重算一次"。

**真实场景（非凸形状、指杆通道、噪声、失败、推、清单拣选）。**

| 规划器 | 清单完成率 | 动作数 | 每目标动作数 | 决策 p50 | p99 |
|---|---|---|---|---|---|
| 贪心 | 47% | 12.4 | 3.40 | 0.11 ms | 0.14 ms |
| MILP | 75% | 17.7 | 3.08 | 8.1 ms | 34.3 ms |
| 退火 SA | 75% | 15.1 | 2.54 | 19.2 ms | 38.4 ms |

差异的来源是清障：清单目标被非目标物体压住时，必须先花一个动作抓走它，而这步是"花钱买未来"。贪心只挑当前分最高的，于是要么在推、要么重复抓已经能抓的，8 次推用完即卡住。两种规划器差距消失但动作开销不同：退火一次给多个候选、可执行前缀更短（2.54 对 3.08），代价是决策延迟高一倍。

## 7 讨论

**为什么经典方法赢。** 三条原因叠加：规模只有数百变量，精确解在秒级内证明最优；退火的嵌入与链断裂开销无法在这样小的实例上摊销；指标是墙钟时间与到达质量，而退火可能占优的指标是 time-to-target 与能耗。审稿人若问"QUBO 给了经典方法给不出什么"，本文的回答是三条可检验的：**多解**（一次采样给出多个近优序列，可供失败后切换，本文只在闭环中定性体现）、**可分解上硬件**（按堆分解，里德堡阵列的阻塞半径天然对应"挡"约束，尚未实现）、**约束保持编码**（把 one-hot 做成 XY/Hamming-weight mixer 可让可行性恒为 1，属门模型路线的后续工作）。

**局限。** (i) 物体与料箱均为合成场景，真实数据只做了单次抓取的质量通道验证，未接入本文的闭环；(ii) 执行模型是抽象的，未与真实夹爪的接触动力学相连；(iii) 位姿候选为离散网格，连续位姿的 QUBO 化需要额外编码；(iv) 所有量子退火数字来自模拟，硬件结果缺失。

## 8 结论

抓取序列可以被写成一个单调的 QUBO：每个物体一行、每步一列、多一列"不抓"，前置只取正耦合，可行解能量等于负收益。这个编码让位姿与顺序可以联合优化（+9.8%），也让闭环评估得以在同一套代码上进行。实验给出的结论是批判性的：在数百变量的规模上，精确解更快更准，退火不占优；贪心在简单场景足够好，但在需要清障的清单任务中损失近一半完成率。QUBO 的价值不在速度，而在于它把"选择、排序、阻塞、预算"放进同一个可采样、可分解的目标函数里——这为多解输出与硬件映射留出了接口。

## 参考文献

[1] A. Lucas. Ising formulations of many NP problems. *Frontiers in Physics*, 2:5, 2014.

[2] G. Kochenberger, J.-K. Hao, F. Glover, et al. The unconstrained binary quadratic programming problem: a survey. *Journal of Combinatorial Optimization*, 28:58–81, 2014.

[3] F. Glover, G. Kochenberger, Y. Du. Quantum Bridge Analytics I: a tutorial on formulating and using QUBO models. *4OR*, 17:335–371, 2019.

[4] E. G. Rieffel, D. Venturelli, B. O'Gorman, M. B. Do, E. Prystay, V. N. Smelyanskiy. A case study in programming a quantum annealer for hard operational planning problems. *Quantum Information Processing*, 14(1):1–36, 2015.

[5] J. Li, S. Ghosh. Quantum-soft QUBO suppression for accurate object detection. In *ECCV*, LNCS:158–173, 2020.

[6] J. C. G. Perion, et al. Performance analysis of QUBO-translated non-maximum suppression for object detection. In *IEEE QCE*, vol. 2:504–505, 2024.

[7] N. Kuete Meli, V. Golyanik, M. Seelbach Benkner, M. Moeller. QuCOOP: a versatile framework for solving composite and binary-parametrised problems on quantum annealers. In *CVPR*:11395–11405, 2025.

[8] M. Seelbach Benkner, Z. Lähner, V. Golyanik, C. Wunderlich, C. Theobalt, M. Moeller. Q-Match: iterative shape matching via quantum annealing. In *ICCV*:7586–7596, 2021.

[9] S. Mücke, R. Heese, S. Müller, M. Wolter, N. Piatkowski. Feature selection on quantum computers. *Quantum Machine Intelligence*, 5:11, 2023.

[10] T. Felefly, et al. An explainable MRI-radiomic quantum neural network to differentiate between large brain metastases and high-grade glioma using quantum annealing for feature selection. *Journal of Digital Imaging*, 36(6):2335–2346, 2023.

[11] D. Leib, T. Seidel, S. Jäger, R. Heese, et al. An optimization case study for solving a transport robot scheduling problem on quantum-hybrid and quantum-inspired hardware. *Scientific Reports*, 13:18743, 2023.

[12] H. Salloum, S. Savin, Y. Kholodov, G. Ryzhakov, M. Farina, I. Oseledets. Quantum annealing for inverse kinematics in robotics. *Scientific Reports*, 16:4244, 2026. doi:10.1038/s41598-025-34346-z

[13] C. Novara, M. Boggio, D. Volpe. Quantum optimization for nonlinear model predictive control. arXiv:2410.19467, 2024 (presented at European Control Conference, 2025).

[14] D. Morrison, P. Corke, J. Leitner. Closing the loop for robotic grasping: a real-time, generative grasp synthesis approach (GG-CNN). In *RSS*, 2018.

[15] S. Kumra, S. Joshi, F. Sahin. Antipodal robotic grasping using generative residual convolutional neural network (GR-ConvNet). In *IROS*, 2020.

[16] A. Depierre, E. Dellandréa, L. Chen. Jacquard: a large scale dataset for robotic grasp detection. In *IROS*, 2018.

[17] J. Mahler, et al. Dex-Net 2.0: deep learning to plan robust grasps with synthetic point clouds and analytic grasp metrics. In *RSS*, 2017.

[18] K. Boothby, P. Bunyk, J. Raymond, A. Roy. Next-generation topology of D-Wave quantum processors. arXiv:2003.00133, 2020.

## 附录 A 复现

```powershell
# 单次抓取（方向 09 的最小实现）
$env:PYTHONPATH = "<repo>\code\seq2d\src"
python code/seq2d/src/_check_qubo.py      # 编码验证，六例
python code/seq2d/src/bench.py --quick    # 静态基准
python code/seq2d/src/solve.py --objects 12 --budget 7

# 序贯 + 位姿
python code/seq2d/src/_check_cand.py
python code/seq2d/src/bench_cand.py
python code/seq2d/src/bench_refine.py

# 规模与滚动
$env:PYTHONPATH = "<repo>\code\seq2d\src;<repo>\code\scaleup\src"
python code/scaleup/src/bench_scale.py --t-frac 0.5
python code/scaleup/src/bench_rolling.py --seeds 5

# 真实场景闭环
$env:PYTHONPATH = "<repo>\code\seq2d\src;<repo>\code\realistic\src"
python code/realistic/src/bench2.py --objects 24 40 60 --seeds 3
python code/realistic/src/plots3.py
```

依赖见 `requirements.txt`；Python 3.12。图由 `plots*.py` 用 Pillow 直接绘制，不依赖 matplotlib。

## 附录 B 验证清单

| 用例 | 变量数 | 方法 | 结论 |
|---|---|---|---|
| 独立两物体 | 18 | 穷举 $2^{18}$ + 全排列 | 通过 |
| 链式阻塞 | 12 | 穷举 $2^{12}$ + 全排列 | 通过 |
| 互锁环（手工构造） | 9 | 穷举 $2^9$ | 通过（环被正确排除） |
| 双堆两物体 | 24 | 整数规划参照 + 三条不变量 | 通过 |
| 四堆四物体 | 144 | 同上 | 通过 |
| 散乱 12 物体 | 96 | 同上 | 通过 |
| 位姿编码五例 | 27–216 | 穷举（2 例）+ 整数规划参照（3 例） | 通过 |

验证过程中发现并修正了两处实现缺陷：解码器未检查"两物体占同一步"，以及候选级编码（早期版本）中"前置未选"补偿按候选数重复累加。前者表现为"看起来可行但能量凭空高一个 $\lambda$"，后者表现为可行解能量低至 $-278$、退火 40 次采样全部不可行。这两处提示：**QUBO 的建模错误不会报错，只会安静地返回不可行的"最优解"**，故小规模穷举是必要工序。
