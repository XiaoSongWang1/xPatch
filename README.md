<div align="center">
  <h2><b> (AAAI25) xPatch: 带有指数季节性趋势分解的双流时间序列预测 </b></h2>
</div>

<div align="center">

[![](https://img.shields.io/badge/arXiv:2412.17323-B31B1B?logo=arxiv)](https://arxiv.org/pdf/2412.17323)
[![](https://img.shields.io/badge/AAAI'25-xPatch-deepgreen)](https://ojs.aaai.org/index.php/AAAI/article/view/34270)
![](https://img.shields.io/github/last-commit/stitsyuk/xPatch)![](https://badges.pufler.dev/visits/stitsyuk/xPatch)![](https://img.shields.io/github/stars/stitsyuk/xPatch)![](https://img.shields.io/github/forks/stitsyuk/xPatch)

</div>

这是论文 [xPatch: 带有指数季节性趋势分解的双流时间序列预测](https://arxiv.org/pdf/2412.17323) 的官方实现。

## 模型概述

**指数化 Patch** (xPatch) 是一个利用指数分解的新型双流架构。

### 季节性趋势分解

**简单移动平均 (SMA)** 是先前数据点的未加权平均值。

<p align="center">
<img src="./figures/sma.png" alt="SMA示意图" style="width: 80%;" align=center />
</p>

**指数移动平均 (EMA)** 是一种指数平滑方法，它在平滑旧数据的同时，为较新的数据点分配更大的权重。

<p align="center">
<img src="./figures/ema.png" alt="EMA示意图" style="width: 80%;" align=center />
</p>

### 架构

双流架构由一个基于MLP的线性流和一个基于CNN的非线性流组成。

<p align="center">
<img src="./figures/xpatch.png" alt="xPatch架构图" align=center />
</p>

## 结果

### 在统一实验设置下的长期预测

在统一实验设置中，使用MSE指标，xPatch在60%的数据集上取得了最佳平均性能；使用MAE指标，则在70%的数据集上取得最佳。

<p align="center">
<img src="./figures/hyper-unified.png" alt="统一设置下的结果" style="width: 80%;" align=center />
</p>

### 在带超参数搜索下的长期预测

在超参数搜索设置中，使用MSE指标，xPatch在70%的数据集上取得了最佳平均性能；使用MAE指标，则在90%的数据集上取得最佳。

<p align="center">
<img src="./figures/hyper-search.png" alt="超参数搜索下的结果" style="width: 80%;" align=center />
</p>

### 在长回看窗口下的效率

我们探索了不同模型从更长的回看窗口中学习的能力。

<p align="center">
<img src="./figures/lookback.png" alt="不同回看窗口长度下的性能对比" style="width: 80%;" align=center />
</p>

### 双流网络 (Dual Flow Net)

我们探索了xPatch架构中双流网络的影响，并评估了每个流的贡献。四种可能的配置：
- **原始配置**: 季节性 -> 非线性流, 趋势 -> 线性流
- **反转配置**: 季节性 -> 线性流, 趋势 -> 非线性流
- **仅非线性流**: 季节性 -> 非线性流, 趋势 -> 非线性流
- **仅线性流**: 季节性 -> 线性流, 趋势 -> 线性流

<p align="center">
<img src="./figures/dual-flow.png" alt="双流网络消融实验结果" style="width: 80%;" align=center />
</p>

## "Drop-last" 技巧

最近的模型广泛采用了“drop-last”技巧，这在 [TFB](https://www.vldb.org/pvldb/vol17/p2363-hu.pdf) 论文中有很好的解释。由于为现有的基准模型重新运行所有不带此技巧的实验将非常复杂和耗时，我们选择依赖于它们官方论文中报告的基准结果来确保实验的公平性。这些报告的结果包含了原始作者应用的所有技巧和超参数搜索。我们的主要目标是使用这些已发表的结果（即使它们不完全公平），而不是自己重新进行基准实验。

然而，最近的工作 [TFB](https://www.vldb.org/pvldb/vol17/p2363-hu.pdf) 引入了一个新的、公平的基准测试，其中排除了“drop-last”技巧。作者们公平地重新评估了所有现有模型，包括全面的超参数搜索。因此，我们通过参考新的 [OpenTS](https://decisionintelligence.github.io/OpenTS/leaderboards/multivariate_time_series) 排行榜的基准结果，纳入了xPatch在没有“drop-last”技巧下进行的公平实验结果。

值得注意的是，xPatch在最长且最具挑战性的数据集——Weather、Traffic和Electricity——上，在所有预测长度上都达到了最先进的性能。

<p align="center">
<img src="./figures/hyper-fair.png" alt="公平实验（无drop-last）下的结果" style="width: 80%;" align=center />
</p>

## 开始使用

1.  安装 conda 环境: ```conda env create -f environment.yml```

2.  下载数据。您可以从 [Google Driver](https://drive.google.com/u/0/uc?id=1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP&export=download)、[Baidu Driver (提取码: i9iy)](https://pan.baidu.com/share/init?surl=r3KhGd0Q9PJIUZdfEYoymg&pwd=i9iy) 或 [Kaggle Datasets](https://www.kaggle.com/datasets/wentixiaogege/time-series-dataset) 下载数据集。所有数据集都已预处理，可以轻松使用。创建一个单独的文件夹 ```./dataset``` 并将所有文件放入该目录。

3.  训练模型。我们在 ```./scripts``` 文件夹下提供了所有基准测试的实验脚本。统一设置的脚本是 *xPatch\_unified*，超参数搜索的脚本是 *xPatch\_search*，而对于没有“drop-last”技巧的公平实验，您可以使用 *xPatch\_fair*。您可以通过以下方式复现实验：

```bash
bash scripts/xPatch_search.sh
```

所有实验都在单个 Quadro RTX 6000 GPU 上进行。您可以根据需要调整超参数（例如批量大小、patch长度、回看窗口和预测长度、指数分解的alpha参数）。我们还在 ```./ablation``` 中提供了基线模型和附录中消融实验的代码（EMA分解、反正切损失、sigmoid学习率调整方案、推理时间）。

## 致谢

我们感谢以下GitHub仓库的宝贵代码和努力：
- Autoformer (https://github.com/thuml/Autoformer)
- FEDformer (https://github.com/MAZiqing/FEDformer)
- ETSformer (https://github.com/salesforce/ETSformer)
- DLinear (https://github.com/cure-lab/LTSF-Linear)
- RLinear (https://github.com/plumprc/RTSF)
- PatchTST (https://github.com/yuqinie98/PatchTST)
- CARD (https://github.com/wxie9/CARD)
- TimeMixer (https://github.com/kwuking/TimeMixer)
- iTransformer (https://github.com/thuml/iTransformer)
- Time-Series-Library (https://github.com/thuml/Time-Series-Library)
- RevIN (https://github.com/ts-kim/RevIN)
- TFB (https://github.com/decisionintelligence/TFB)

## 联系方式

如果您有任何问题或疑虑，请通过 stitsyuk@kaist.ac.kr 联系我们或提交一个 issue。

## 引用

如果您在研究中发现这个仓库很有用，请考虑引用我们的论文，如下所示：

```
@inproceedings{stitsyuk2025xpatch,
  title={xPatch: Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend Decomposition},
  author={Stitsyuk, Artyom and Choi, Jaesik},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={19},
  pages={20601--20609},
  year={2025}
}
```