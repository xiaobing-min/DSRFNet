# Foggy-UAVCD and DSRF-Net

<p align="center">
  <strong>Benchmarking Robust Satellite-UAV Change Detection under Fog Degradation</strong>
</p>

<p align="center">
  <a href="#citation"><img src="https://img.shields.io/badge/ISPRS%20JPRS-Accepted-2ea44f" alt="Accepted by ISPRS Journal of Photogrammetry and Remote Sensing"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-Implementation-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch implementation"></a>
  <a href="https://github.com/xiaobing-min/DSRFNet"><img src="https://img.shields.io/badge/Code-GitHub-181717?logo=github" alt="GitHub repository"></a>
</p>

Official PyTorch implementation of **DSRF-Net** and data-access instructions for **Foggy-UAVCD**.

> **Accepted by ISPRS Journal of Photogrammetry and Remote Sensing.** The final DOI and publisher link will be added when they become available.

**Authors:** Xiu-Wen Huang<sup>1,&#42;</sup>, Qing-Ling Shu<sup>1,&#42;</sup>, Kai-Xuan Jiang<sup>2</sup>, Wei Lu<sup>1</sup>, Si-Bao Chen<sup>1,&dagger;</sup>, Jin Tang<sup>1</sup>, and Bin Luo<sup>1</sup>

<sup>1</sup> Anhui University &nbsp;&nbsp; <sup>2</sup> Wuhan University  
<sup>&#42;</sup> Equal contribution &nbsp;&nbsp; <sup>&dagger;</sup> Corresponding author

## Highlights

- **Foggy-UAVCD** contains 7,450 co-registered satellite-UAV image pairs for building change detection under controllable fog degradation.
- **DSRF-Net** follows a lightweight *rectify-then-fuse* design for asymmetric inputs: a clear pre-change satellite image and a fog-degraded post-change UAV image.
- **Cross-Modal Feature Rectification (CFR)** uses satellite priors for global statistical alignment and prior-guided restoration of contaminated UAV features.
- The **Spatio-Frequency Fusion Decoder (SFFD)** uses the **Dual-Domain Fusion Upsampler (DDFU)**, combining Semantic Flow Warping (SFW) and Spectral Detail Refinement (SDR) for boundary reconstruction.
- The MobileNetV2-based model has **3.43M parameters** and **1.84G FLOPs**.

## Dataset

### Foggy-UAVCD

| Split | Image pairs |
|---|---:|
| Train | 5,215 |
| Validation | 745 |
| Test | 1,490 |
| **Total** | **7,450** |

All samples are provided as 256 x 256 patches. Download the dataset from [Baidu Netdisk](https://pan.baidu.com/s/1wTyGVvRwN1Dxgr2RjsOF3Q?pwd=1124) (extraction code: `1124`) and organize it as follows:

```text
Foggy-UAVCD/
|-- train/
|   |-- A/          # Pre-change satellite images
|   |-- B/          # Post-change foggy UAV images
|   |-- label/      # Binary change masks
|   `-- list/
|       `-- train.txt
|-- val/
|   |-- A/
|   |-- B/
|   |-- label/
|   `-- list/
|       `-- val.txt
`-- test/
    |-- A/
    |-- B/
    |-- label/
    `-- list/
        `-- test.txt
```

Each list file contains one image filename per line. The same filename must exist under `A/`, `B/`, and `label/` for that split.

The paper additionally evaluates DSRF-Net on LEVIR-CD, SYSU-CD, WHU-CD, XiongAn, HTCD, MAHCD, and MT-Wuhan. Please obtain these datasets from their official sources and follow their respective licenses and terms.

## Repository Layout

```text
DSRFNet/
|-- dataset.py
|-- metric_tool.py
|-- Transforms.py
|-- utils.py
|-- requirements.txt
|-- models/
|   |-- MobileNetV2.py
|   |-- model.py
|   `-- resnet.py
`-- tools/
    |-- train.py
    |-- test.py
    |-- train.sh
    |-- test.sh
    `-- torchutils.py
```

## Installation

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/xiaobing-min/DSRFNet.git
cd DSRFNet
conda create -n dsrfnet python=3.10 -y
conda activate dsrfnet
```

The released code was developed with the PyTorch 1.8 generation. Install the PyTorch build matching your CUDA toolkit, then install the remaining dependencies:

```bash
pip install torch==1.8.0 torchvision==0.9.0
pip install matplotlib==3.5.0 numpy==1.21.2 opencv-python==4.5.4.60 \
  Pillow==9.1.0 scipy==1.7.3 tqdm==4.62.3 timm einops
```

`timm` and `einops` are imported by `models/model.py` but are not listed in the current `requirements.txt`; the command above installs them explicitly. On newer hardware, use a mutually compatible Python, PyTorch, torchvision, and CUDA combination.

The experiments reported in the paper were conducted on one NVIDIA GeForce RTX 3090 GPU.

## Prepare the Data Path

The current training and testing scripts use dataset aliases. Before running Foggy-UAVCD, replace the path assigned in the `uav-rs` branch of both `tools/train.py` and `tools/test.py` with the absolute path to your extracted `Foggy-UAVCD` directory.

For example, the mapping should resolve as follows:

```text
uav-rs -> /path/to/Foggy-UAVCD
```

Run all commands from the repository root because the scripts add the current directory to Python's import path.

## Training

```bash
python ./tools/train.py --file_root uav-rs --lr 5e-4 --max_steps 40000 --batch_size 32
```

The script writes checkpoints and logs to:

```text
results_uav-rs_iter_40000_lr_0.0005/
|-- best_model.pth
|-- checkpoint.pth.tar
`-- trainValLog.txt
```

The best model is selected by validation F1 score. The four decoder predictions are supervised with a BCE + Dice hybrid loss.

### Reproduction settings

The accepted paper reports the following common experimental protocol:

## Evaluation

The testing script looks for `best_model.pth` in the result directory determined by `--file_root`, `--max_steps`, and `--lr`. Keep these arguments identical to training:

```bash
python ./tools/test.py --file_root uav-rs --lr 5e-4 --max_steps 40000 --batch_size 32
```

Predicted change maps are saved under `Predict/uav-rs/`. The script reports Kappa, IoU, F1, Recall, and Precision and also saves the metrics to `results.mat`.

Pretrained weights are not bundled in the current repository snapshot. Place a compatible checkpoint at:

```text
results_uav-rs_iter_40000_lr_0.0005/best_model.pth
```

## Citation

If this project is useful for your research, please cite:

```bibtex
@article{huang2026foggyuavcd,
  title   = {Foggy-UAVCD and DSRF-Net: Benchmarking Robust Satellite-UAV Change Detection under Fog Degradation},
  author  = {Huang, Xiu-Wen and Shu, Qing-Ling and Jiang, Kai-Xuan and Lu, Wei and Chen, Si-Bao and Tang, Jin and Luo, Bin},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  year    = {2026},
  note    = {Accepted, in press}
}
```

Please update this entry with the final volume, pages, and DOI after the version of record is published.

## License and Data Use

The code is provided for academic research. A formal software license file is not included in the current repository snapshot; contact the authors before redistribution or commercial use. Dataset users must also comply with the terms of the original remote-sensing imagery providers.

## Contact

- Xiu-Wen Huang: [hxw@stu.ahu.edu.cn](mailto:hxw@stu.ahu.edu.cn)
- Qing-Ling Shu：[sql@stu.ahu.edu.cn](mailto:sql@stu.ahu.edu.cn)
- Si-Bao Chen: [sbchen@ahu.edu.cn](mailto:sbchen@ahu.edu.cn)



