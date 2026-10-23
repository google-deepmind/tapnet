# Tracking Any Point (TAP): 任意点长周期物理追踪

[[`TAP-Vid`](https://tapvid.github.io/)] [[`TAPIR`](https://deepmind-tapir.github.io/)] [[`RoboTAP`](https://robotap.github.io/)] [[`博客文章`](https://deepmind-tapir.github.io/blogpost.html)] [[`BootsTAP`](https://bootstap.github.io/)] [[`TAPVid-3D`](https://tapvid3d.github.io/)] [[`TAPNext`](https://tap-next.github.io/)] [[`TRAJAN`](https://trajan-paper.github.io)] [[`TAPNext++`](https://tap-next-plus-plus.github.io/)]

[English](README.md) | [简体中文](README.zh.md)

https://github.com/google-deepmind/tapnet/assets/4534987/9f66b81a-7efb-48e7-a59c-f5781c35bebc

欢迎来到 Google DeepMind 任意点追踪（Tracking Any Point, TAP）官方开源代码库。这里汇聚了 TAP-Vid 与 TAPVid-3D 基准数据集、SOTA 级 TAPIR 追踪模型系列、以及面向真实世界机器人操作的 RoboTAP 扩展。

- **[TAP-Vid](https://tapvid.github.io)**：评估任意点追踪任务的标准基准，包含真实世界与合成视频中物理实体表面点的真实标注轨迹（Ground-truth）。
- **[TAPIR](https://deepmind-tapir.github.io)**：包含两个阶段的高效两阶段追踪算法：1）匹配阶段（Matching Stage），在所有其他帧上独立定位与查询点匹配的候选点；2）精细化阶段（Refinement Stage），基于局部相关性同步更新轨迹坐标与查询特征。该模型具备极高推理速度，并在 TAP-Vid 基准上显著超越此前所有算法。
- **[RoboTAP](https://robotap.github.io)**：利用 TAPIR 点轨迹通过高效现实世界模仿学习执行机器人抓取与操纵任务的系统。包含真实机器人操纵视频中的密集真值点标注数据集。
- **[BootsTAP](https://bootstap.github.io)**（基于自举训练的 TAP）：利用大规模无标注真实世界视频增强追踪准确率。通过对视频进行空间几何变换、图像扰动以及不同查询点选择，促使模型输出鲁棒且一致的预测。应用于 TAPIR 所衍生的 BootsTAPIR 在保持相似架构的同时大幅刷新了 TAP-Vid 评测表现。
- **[TAPVid-3D](https://tapvid3d.github.io)**：针对 3D 空间长周期任意点追踪任务的评估基准与度量体系，包含 4,000+ 段真实视频中计算生成的 100 万+ 条公制 3D 真值轨迹。
- **[TAPNext](https://tap-next.github.io)**：最新、最强、最快且结构极为简洁的追踪器。将 TAP 问题形式化为类似自回归的 Next Token Prediction，仅通过在网络中前向传递特征信息即可实现精准点追踪（最佳 TAPNext 权重基于 BootsTAP 流程微调）。
- **[TRAJAN](https://trajan-paper.github.io)**：首个点轨迹自编码器（TRAJectory AutoeNcoder）。TRAJAN 以一组支撑点轨迹为条件，重构留出的查询点轨迹。其学习到的嵌入空间可用于衡量生成式视频模型的真实感与物理时空一致性。
- **[TAPNext++](https://tap-next-plus-plus.github.io)**：TAPNext 的进阶增强权重，在长周期稳定追踪上实现了 40 倍的性能提升，支持跨遮挡长程追踪并具备强大的重检测（Re-detection）能力。

本仓库核心内容包括：

- [TAPNext / TAPNext++ / TAPIR / BootsTAPIR 演示](#演示体验-demos)：涵盖在线 **Colab 交互式演示** 与克隆代码后的本地 **实时摄像头实时推断演示**
- [TRAJAN 演示](#演示体验-demos)：在线 **Colab 演示**
- [TAP-Vid 基准测试](#tap-vid-基准)：评测**数据集**与评估**指标计算代码**
- [RoboTAP 基准测试](#robotap-基准)：机器人评估**数据集**与基于点轨迹的分割聚类算法
- [TAPVid-3D 基准测试](#tapvid-3d-基准)：3D 追踪评测**指标**与**评估脚本**
- [预训练权重 Checkpoints](#预训练权重-checkpoints)：涵盖 TAP-Net、TAPIR、BootsTAPIR、TAPNext 的 **JAX** 与 **PyTorch** 官方预训练权重
- [模型训练指南](#模型训练-training)：在 Kubric 合成数据集上训练 TAP-Net 与 TAPIR 的详细说明

## 演示体验 (Demos)

运行 TAP 系列模型最便捷的方式是直接使用我们的在线 Colab Notebook，亦可克隆本仓库并在自有硬件上运行实时摄像头推断。

### Colab 在线演示

1. <a target="_blank" href="https://colab.research.google.com/github/deepmind/tapnet/blob/main/colabs/torch_tapnextpp_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="TAPNext++"/></a> **TAPNext++**：微调版 BootsTAPNext 权重，支持超长时序追踪、遮挡追踪与丢失重检，已在 PointOdyssey 与 Kubric-1024 上微调。
2. <a target="_blank" href="https://colab.research.google.com/github/deepmind/tapnet/blob/main/colabs/tapnext_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="BootsTAPNext"/></a> **BootsTAPNext**：最强在线逐帧（Online/per-frame）TAPNext 模型（论文主报模型）。
3. <a target="_blank" href="https://colab.research.google.com/github/deepmind/tapnet/blob/master/colabs/torch_tapnext_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="BootsTAPNext PyTorch"/></a> **BootsTAPNext PyTorch**：PyTorch 原生复现版本，拥有与 JAX 模型完全一致的结构与权重。
4. <a target="_blank" href="https://colab.research.google.com/github/deepmind/tapnet/blob/master/colabs/tapir_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Offline TAPIR"/></a> **标准 TAPIR**：一次性处理全视频的最强离线 TAPIR / BootsTAPIR 模型。
5. <a target="_blank" href="https://colab.research.google.com/github/deepmind/tapnet/blob/master/colabs/causal_tapir_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Online TAPIR"/></a> **在线因果 TAPIR (Online TAPIR)**：时序因果结构，适用于流式实时点追踪，可在单卡 GPU 上实时运行。
6. <a target="_blank" href="https://colab.research.google.com/github/deepmind/tapnet/blob/master/colabs/tapir_rainbow_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="TAPIR Rainbow Visualization"/></a> **彩虹轨迹可视化 (Rainbow Visualization)**：自动前景/背景分割并补偿相机运动，直观展现物体在真实物理空间中的三维运动轨迹。
7. <a target="_blank" href="https://colab.research.google.com/github/deepmind/tapnet/blob/master/colabs/torch_tapir_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Offline PyTorch TAPIR"/></a> **标准 PyTorch TAPIR**：PyTorch 版本的离线 TAPIR 模型。
8. <a target="_blank" href="https://colab.research.google.com/github/deepmind/tapnet/blob/master/colabs/torch_causal_tapir_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Online PyTorch TAPIR"/></a> **在线 PyTorch TAPIR**：PyTorch 版本的因果在线 BootsTAPIR 模型。
9. <a target="_blank" href="https://colab.research.google.com/github/deepmind/tapnet/blob/master/colabs/trajan_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="TRAJAN"/></a> **TRAJAN**：基于输入点轨迹集合重构查询点运动轨迹的自编码器演示。
10. <a target="_blank" href="https://colab.research.google.com/github/deepmind/tapnet/blob/master/colabs/tapir_clustering.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Point Clustering"/></a> **RoboTAP 聚类演示**：RoboTAP 论文中所采用的轨迹点分割与物体聚类算法演示。
11. <a target="_blank" href="https://colab.research.google.com/github/deepmind/tapnet/blob/master/colabs/kubric_for_tapvid(3d).ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Kubric"/></a> **Kubric 用于 TAPVid(3D)**：展示如何利用 Kubric 数据集生成世界坐标系下的 2D 与 3D 真值点轨迹。

### 本地实时推断演示 (Live Demo)

克隆代码仓库：

```bash
git clone https://github.com/deepmind/tapnet.git
```

进入项目目录：

```bash
cd tapnet
```

安装 `tapnet` Python 包及推断运行依赖：

```bash
pip install .
```

下载预训练权重：

```bash
mkdir checkpoints
wget -P checkpoints https://storage.googleapis.com/dm-tapnet/causal_tapir_checkpoint.npy
```

将上级目录添加至 `PYTHONPATH`：

```bash
export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH
```

若使用 CUDA 加速，请确保驱动完备并安装与本地 CUDA/CUDNN 版本兼容的 JAX（可参考 [JAX 官方安装手册](https://github.com/jax-ml/jax#installation)）。

随后即可启动实时摄像头演示并交互式点击想要追踪的目标点：

```bash
cd ..
python3 ./tapnet/live_demo.py
```

在 Quadro RTX 4000（2018 年移动版 GPU）上测试，480x480 分辨率推断速度可达 ~17 FPS。

## 基准测试 (Benchmarks)

本仓库维护三个互相关联的评测基准：TAP-Vid、RoboTAP 以及 TAPVid-3D。

### TAP-Vid 基准

https://github.com/google-deepmind/tapnet/assets/4534987/ff5fa5e3-ed37-4480-ad39-42a1e2744d8b

[TAP-Vid](https://tapvid.github.io) 包含视频以及由人工手动标注或物理仿真器生成的点轨迹。目标是评估对任意固体物理表面上可追踪点的追踪能力。算法在某帧接收一个查询点输入，必须预测该点在所有其他帧上的轨迹坐标及可见性状态。该任务要求具备点级亚像素精度、长时序追踪能力，并适用于任意可形变物体表面。

更多关于下载、评测的细节请参阅 [TAP-Vid README](https://github.com/google-deepmind/tapnet/blob/main/tapnet/tapvid)。

### RoboTAP 基准

[RoboTAP](https://robotap.github.io/) 探索了点追踪模型在机器人领域的应用。数据集遵循与 TAP-Vid 相同的标注格式，但在真实机器人操纵视频中全部采用人工高精度标注，物体类别与环境多样性显著提升，包含 265 段评测视频。可参阅 [RoboTAP Colab Notebook](https://colab.research.google.com/github/deepmind/tapnet/blob/master/colabs/tapir_clustering.ipynb) 体验轨迹聚类分割。

### TAPVid-3D 基准

TAPVid-3D 是评估三维空间长周期任意点追踪任务（TAP-3D）的权威基准。涵盖 4,000+ 段真实世界视频及对应的公制 3D 物理轨迹。详细指引请参阅 [TAPVid-3D README](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid3d)。

### 坐标系规范说明 (A Note on Coordinates)

- 在离线存储的数据集中，`(x, y)` 坐标通常采用归一化栅格坐标：即 `(0, 0)` 为左上角像素的左上顶点，`(1, 1)` 为右下角像素的右下顶点。
- 代码内部会将其即时转换为常规栅格坐标（匹配 Kubric 格式）：`(0, 0)` 为左上角顶点，`(h, w)` 为右下角顶点，其中 $h$ 与 $w$ 分别为图像的高度与宽度。
- 2D 坐标按 `(x, y)` 顺序存储；而 3D 坐标通常采用 `(t, y, x)` 顺序存储，其中 $t$ 为帧时序坐标（$0$ 表示首帧，$0.5$ 表示第 1 帧与第 2 帧之间的时间中点）。在指标计算中单像素误差即会产生显著差异，请务必关注坐标格式。

## 预训练权重 (Checkpoints)

[<img alt="Checkpoints" src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo-with-title.png" width="200px">](https://huggingface.co/google/tapnet)

权重文件可由下方链接直接下载，亦可在 [HuggingFace (google/tapnet)](https://huggingface.co/google/tapnet) 获取。评测默认在 256x256 分辨率下进行，更高推断分辨率可进一步提升精度（BootsTAPIR 推荐 512x512）：

| 模型 | 预训练权重链接 | 配置文件 | 骨干网络 (Backbone) | 训练分辨率 | DAVIS First (AJ) | DAVIS Strided (AJ) | Kinetics First (AJ) | RoboTAP First (AJ) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TAP-Net** | [Jax](https://storage.googleapis.com/dm-tapnet/checkpoint.npy) | [tapnet_config.py](https://github.com/google-deepmind/tapnet/blob/main/configs/tapnet_config.py) | TSM-ResNet18 | 256x256 | 33.0% | 38.4% | 38.5% | 45.1% |
| **TAPIR** | [Jax](https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.npy) & [PyTorch](https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.pt) | [tapir_config.py](https://github.com/google-deepmind/tapnet/blob/main/configs/tapir_config.py) | ResNet18 | 256x256 | 58.5% | 63.3% | 50.0% | 59.6% |
| **Online TAPIR** | [Jax](https://storage.googleapis.com/dm-tapnet/causal_tapir_checkpoint.npy) | [causal_tapir_config.py](https://github.com/google-deepmind/tapnet/blob/main/configs/causal_tapir_config.py) | ResNet18 | 256x256 | 56.2% | 58.3% | 51.2% | 59.1% |
| **BootsTAPIR** | [Jax](https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.npy) & [PyTorch](https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt) | [tapir_bootstrap_config.py](https://github.com/google-deepmind/tapnet/blob/main/configs/tapir_bootstrap_config.py) | ResNet18 + 4 Convs | 256x256 + 512x512 | 62.4% | 67.4% | 55.8% | 69.2% |
| **Online BootsTAPIR** | [Jax](https://storage.googleapis.com/dm-tapnet/bootstap/causal_bootstapir_checkpoint.npy) & [PyTorch](https://storage.googleapis.com/dm-tapnet/bootstap/causal_bootstapir_checkpoint.pt) | [tapir_bootstrap_config.py](https://github.com/google-deepmind/tapnet/blob/main/configs/tapir_bootstrap_config.py) | ResNet18 + 4 Convs | 256x256 + 512x512 | 59.7% | 61.2% | 55.1% | 69.1% |
| **TAPNext** | [Jax](https://storage.googleapis.com/dm-tapnet/tapnext/bootstapnext_ckpt.npz) | [tapnext_demo.ipynb](https://github.com/google-deepmind/tapnet/blob/main/colabs/tapnext_demo.ipynb) | TrecViT-B | 256x256 | 65.25% | 68.9% | 57.3% | 64.1% |
| **TAPNext++** | [PyTorch](https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt) | [torch_tapnextpp_demo.ipynb](https://colab.research.google.com/github/deepmind/tapnet/blob/main/colabs/torch_tapnextpp_demo.ipynb) | TrecViT-B | 256x256 | 65.6% | - | 53.9% | 61.1% |
| **TAPNext++ (512×512)** | [PyTorch](https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt) | [VOTSp2026 README](https://github.com/google-deepmind/tapnet/blob/main/tapnet/tapnextpp/votsp2026/README.md) | TrecViT-B | 512x512 | 67.0% | 71.2% | - | - |
| **TRAJAN** | [Jax](https://storage.googleapis.com/dm-tapnet/trajan/track_autoencoder_ckpt.npz) | [trajan_demo.ipynb](https://github.com/google-deepmind/tapnet/blob/main/colabs/trajan_demo.ipynb) | - | - | - | - | - | - |

## 模型训练 (Training)

我们提供了基于 JAX 的 TAP-Net 和 TAPIR 完整训练与评测框架，详见 [training README](https://github.com/google-deepmind/tapnet/tree/main/tapnet/training)。

此外，社区开发者亦开发了 [TAPIR 的 PyTorch 训练版本 (echotracker)](https://github.com/riponazad/echotracker)，供开发者参考（注：该项目非 Google DeepMind 官方关联项目）。

## 论文引用 (Citing this Work)

若在研究中使用了 TAPNet、TAPIR、RoboTAP、BootsTAP、TAPVid-3D、TAPNext 或 TRAJAN，请使用以下 BibTeX 引用：

```bibtex
@article{doersch2022tap,
  title={{TAP}-Vid: A Benchmark for Tracking Any Point in a Video},
  author={Doersch, Carl and Gupta, Ankush and Markeeva, Larisa and Recasens, Adria and Smaira, Lucas and Aytar, Yusuf and Carreira, Joao and Zisserman, Andrew and Yang, Yi},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={13610--13626},
  year={2022}
}

@inproceedings{doersch2023tapir,
  title={{TAPIR}: Tracking any point with per-frame initialization and temporal refinement},
  author={Doersch, Carl and Yang, Yi and Vecerik, Mel and Gokay, Dilara and Gupta, Ankush and Aytar, Yusuf and Carreira, Joao and Zisserman, Andrew},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10061--10072},
  year={2023}
}

@article{vecerik2023robotap,
  title={{RoboTAP}: Tracking arbitrary points for few-shot visual imitation},
  author={Vecerik, Mel and Doersch, Carl and Yang, Yi and Davchev, Todor and Aytar, Yusuf and Zhou, Guangyao and Hadsell, Raia and Agapito, Lourdes and Scholz, Jon},
  journal={International Conference on Robotics and Automation},
  pages={5397--5403},
  year={2024}
}

@article{doersch2024bootstap,
  title={{BootsTAP}: Bootstrapped Training for Tracking-Any-Point},
  author={Doersch, Carl and Luc, Pauline and Yang, Yi and Gokay, Dilara and Koppula, Skanda and Gupta, Ankush and Heyward, Joseph and Rocco, Ignacio and Goroshin, Ross and Carreira, Jo{\~a}o and Zisserman, Andrew},
  journal={Asian Conference on Computer Vision},
  year={2024}
}

@article{koppula2024tapvid,
  title={{TAPVid}-{3D}: A Benchmark for Tracking Any Point in {3D}},
  author={Koppula, Skanda and Rocco, Ignacio and Yang, Yi and Heyward, Joe and Carreira, Jo{\~a}o and Zisserman, Andrew and Brostow, Gabriel and Doersch, Carl},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}

@article{zholus2025tapnext,
  title={TAPNext: Tracking Any Point (TAP) as Next Token Prediction},
  author={Zholus, Artem and Doersch, Carl and Yang, Yi and Koppula, Skanda and Patraucean, Viorica and He, Xu Owen and Rocco, Ignacio and Sajjadi, Mehdi S. M. and Chandar, Sarath and Goroshin, Ross},
  journal={arXiv preprint arXiv:2504.05579},
  year={2025}
}

@article{allen2025trajan,
  title={Direct Motion Models for Assessing Generated Videos},
  author={Allen, Kelsey and Doersch, Carl and Zhou, Guangyao and Suhail, Mohammed and Driess, Danny and Rocco, Ignacio and Rubanova, Yulia and Kipf, Thomas and Sajjadi, Mehdi S. M. and Murphy, Kevin and Carreira, Joao and van Steenkiste, Sjoerd},
  journal={arXiv preprint},
  year={2025}
}
```

## 开源协议与免责声明 (License and Disclaimer)

版权所有 2022-2026 Google LLC。

- TAPVid-3D 基准相关的软件及材料遵循 `tapvid3d/LICENSE` 中注明的许可协议。
- 本仓库中的所有其他软件代码均采用 [Apache 2.0 许可证](https://www.apache.org/licenses/LICENSE-2.0)。
- 本仓库发布并链接的所有预训练模型权重（包括 TAP-Net、TAPIR、BootsTAPIR、TAPNext、TAPNext++ 及 TRAJAN 权重）均采用 Apache 2.0 许可证。
- TAP-Vid 数据集的非代码材料（包括 TAP-Vid 标注、RGB-Stacking 视频与 RoboTAP 视频）遵循 [CC-BY 4.0 许可证](https://creativecommons.org/licenses/by/4.0/)。
- DAVIS 原始视频来自验证集，按原作者的知识共享协议授权；Kinetics 视频来自 YouTube 公开资源，受各自单独的许可协议约束。

本项目非 Google 官方产品。

---

> **说明**：本文档为基于英文原文的中文翻译版本。如翻译内容与英文原文存在不一致之处，请以 [英文原版 (README.md)](README.md) 为准。
