# Notebooks DeepWeeds

[README bài làm](../../README.md) chứa môi trường, protocol, lệnh tái lập và resume. Output nhúng đã bỏ; kết quả thật ở `../../evidence/`, `../../curves/`, `../../predictions/` và Excel.

| Bước | Notebook | Trạng thái / cách dùng |
|---|---|---|
| Toàn bộ / Colab | [reproduce_colab.ipynb](reproduce_colab.ipynb) | CPU audit mặc định; bật REPRODUCE để chạy GPU trên workspace mới |
| EDA | [eda_deepweeds.ipynb](eda_deepweeds.ipynb) | Đã hoàn tất; Run All tạo lại từ dữ liệu |
| Pilot6.2 | [pilot_6_2.ipynb](pilot_6_2.ipynb) | Đã kiểm tra5 backbone/AMP |
| B01 | [ResNet50](train_B01_resnet50.ipynb) | 12 epoch, seed0, test tắt |
| B02 | [ResNeXt50](train_B02_resnext50_32x4d.ipynb) | 12 epoch, seed0, test tắt |
| B03 | [ConvNeXtTiny](train_B03_convnext_tiny.ipynb) | 12 epoch, seed0; alias T00 |
| B04 | [DeiTSmall](train_B04_deit_small_patch16_224.ipynb) | 12 epoch, seed0, test tắt |
| B05 | [EfficientNetB0](train_B05_efficientnet_b0.ipynb) | 12 epoch, seed0, test tắt |
| T00–T08 | [Ablation tổng](ablation_convnext_tiny.ipynb) | Đã hoàn tất; RUN_TRAINING=False để chỉ đọc |
| I00/I01/I02/I07/I08 | [Inference/latency](inference_T04_convnext_tiny.ipynb) | Đã hoàn tất; tắt RUN_QUALITY/RUN_LATENCY để chỉ đọc |
| Chung kết | [Final10](final_10_convnext_tiny.ipynb) | Đã hoàn tất3 seed/nhóm; tắt cả3 flag RUN_* để chỉ đọc |

RTX3060/12GB: cặp B03+B04 trước, B02+B05 sau, B01 cuối phù hợp VRAM pilot; cần kernel/process riêng. Tốc độ song song chưa đo. Benchmark luôn chạy riêng; T/final khuyến nghị tuần tự.

| ID | Notebook ablation riêng | Thay đổi |
|---|---|---|
| T01 | [Scratch](train_T01_convnext_scratch.ipynb) | Random initialization |
| T02 | [Frozen](train_T02_convnext_frozen.ipynb) | Chỉ classifier |
| T03 | [Color](train_T03_convnext_color.ipynb) | Color augmentation |
| T04 | [CutMix](train_T04_convnext_cutmix.ipynb) | α=1 |
| T05 | [LS](train_T05_convnext_label_smoothing.ipynb) | ε=0,1 |
| T06 | [Focal](train_T06_convnext_focal.ipynb) | γ=2 |
| T07 | [Weighted CE](train_T07_convnext_weighted_ce.ipynb) | Inverse frequency từ train |
| T08 | [Combination](train_T08_convnext_combination.ipynb) | CutMix+weighted CE, khóa từ val |

Final seed1/2, chỉ train+prepare val:

| Nhóm | Seed1 | Seed2 |
|---|---|---|
| T00 | [seed1](train_T00_convnext_seed1.ipynb) | [seed2](train_T00_convnext_seed2.ipynb) |
| F01 | [seed1](train_F01_convnext_seed1.ipynb) | [seed2](train_F01_convnext_seed2.ipynb) |

Seed0 reuse B03/T04. Chỉ master final mở test sau khi đủ sáu checkpoint/temperature đã seal. Không chạy master đồng thời với notebook riêng cùng ID. Không xóa marker hoặc đổi manifest/code để thử lại test. Evidence không dùng để resume; workspace mới có runs mới và prediction tham chiếu lưu riêng.
