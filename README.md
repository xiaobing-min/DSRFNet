Dataset Preparation

### Foggy-UAVCD

Download Foggy-UAVCD from (https://pan.baidu.com/s/1wTyGVvRwN1Dxgr2RjsOF3Q?pwd=1124 提取码: 1124 # DSRFNet
) and organize it as follows:

```text
Foggy-UAVCD/
|-- train/
|   |-- A/          # Pre-change satellite images
|   |-- B/          # Post-change foggy UAV images
|   |-- label/      # Binary change labels
|   `-- list/       # Training file lists
|-- val/
|   |-- A/
|   |-- B/
|   |-- label/
|   `-- list/
`-- test/
    |-- A/
    |-- B/
    |-- label/
    `-- list/
```

Set the dataset root in `configs/foggy_uavcd.yaml` or in the corresponding training and testing scripts.

### Public Benchmarks

The manuscript also evaluates DSRF-Net on the following public datasets:

- LEVIR-CD
- SYSU-CD
- WHU-CD
- XiongAn
- HTCD
- MAHCD
- MT-Wuhan

Please obtain these datasets from their official sources and comply with their respective licenses and terms of use. 
