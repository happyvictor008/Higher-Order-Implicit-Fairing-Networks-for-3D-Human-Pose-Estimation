# HIgh-order-GCNII
High-order GCN with Initial residual and Identity mapping on 3D Human Pose estimation 
### Results on Human3.6M

Under Protocol 1 (mean per-joint position error) and Protocol 2 (mean per-joint position error after rigid alignment).

| Method | 2D Detections | # of Epochs | # of Parameters | MPJPE (P1) | P-MPJPE (P2) |
|:-------|:-------:|:-------:|:-------:|:-------:|:-------:|
| Martinez et al. [1] | Ground truth | 200  | 4.29M | 44.40 mm | 35.25 mm |
| SemGCN | Ground truth | 50 | 0.27M | 42.14 mm | 33.53 mm |
| SemGCN (w/ Non-local) | Ground truth | 30 | 0.43M | 40.78 mm | 31.46 mm |
| HGCN   | Ground truth | 50 |  1.20M  | 39.52 mm | 31.07 mm |
| HGCNII(Ours)   | Ground truth | 50 |  1.20M  | **39.16 mm** | **30.83 mm** |

Results using Ground truth are reported. 
The results are borrowed from [SemGCN](https://github.com/garyzhao/SemGCN) and [High-order GCN](https://github.com/ZhimingZo/HGCN).
