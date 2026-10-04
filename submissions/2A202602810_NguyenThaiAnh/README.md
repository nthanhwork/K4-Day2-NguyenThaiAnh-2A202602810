# DeepWeeds — Nguyễn Thái Anh · 2A202602810

Đã hoàn tất EDA/sanity, năm backbone B01–B05, ablation T00–T08, inference I00/I01/I02/I07/I08 và chung kết ba seed cho baseline/final. Bài nộp gồm [results.xlsx](results.xlsx), [báo cáo Markdown](report.md), [PDF 8 trang](report.pdf), `curves/`, `code/`, README và `predictions/`. Log/cấu hình/protocol/evaluator nhỏ được lưu thêm tại [evidence/](evidence/README.md).

Final **F01 = ConvNeXt-Tiny `in12k_ft_in1k` + CutMix α=1 + CE + I07**, baseline **T00 = cùng ConvNeXt + basic CE + I00**. Seed 0/1/2, fold0, 3.507 ảnh test/seed; mean ± std mẫu (`ddof=1`):

| Cấu hình | Top-1 test | Macro-F1 test | ECE test |
|---|---:|---:|---:|
| T00 | 97,7474% ± 0,0754 điểm % | 97,1344% ± 0,0832 điểm % | 0,013120 ± 0,000845 |
| F01 | **98,1941% ± 0,1317 điểm %** | **97,7703% ± 0,1119 điểm %** | **0,005338 ± 0,001231** |

Δ macro-F1 = **+0,6359 điểm %**. I07 p95 batch1 đại diện seed0 **3,7609 ms** trên RTX3060: GPU crop/views → model → aggregation/calibration → softmax với input ở GPU, không gồm disk/CPU/H2D. Evaluator tự chấm mục I **19/20**, chưa phải điểm toàn bài. Số đầy đủ/từng seed ở Excel và evaluator.

## Notebook và link Colab

[Mở notebook tái lập trên Colab](https://colab.research.google.com/github/nthanhwork/K4-Day2-NguyenThaiAnh-2A202602810/blob/main/submissions/2A202602810_NguyenThaiAnh/code/notebooks/reproduce_colab.ipynb) · [file local](code/notebooks/reproduce_colab.ipynb) · [danh sách notebook](code/notebooks/README.md).

Link dùng repo GitHub công khai và nhánh `main`; **cần push commit chứa notebook lên GitHub trước khi link hoạt động**. Notebook mặc định kiểm tra số liệu đã nộp trên CPU; `REPRODUCE=True` chạy thực nghiệm mới trên GPU theo protocol đầy đủ. Colab/Kaggle cần Internet, GPU runtime, đủ thời gian và ổ lưu bền. Python/CUDA khác môi trường gốc có thể cho kết quả khác, không cam kết giống từng bit.

## Môi trường và cài đặt

Thực nghiệm gốc: Python **3.14.7**, torch **2.10.0+cu128**, torchvision **0.25.0**, timm **1.0.30**, numpy **2.5.3**, pandas **3.0.6**, CUDA **12.8**, RTX3060 **12GB**. Xem [requirements](code/requirements.txt), [lock phiên bản thực tế](code/requirements.lock.txt), và `evidence/runs/<exp_id>/seed<k>/environment.json`. Giữ torch/torchvision và CUDA wheel phù hợp phần cứng; dùng đúng lock để khớp môi trường gốc.

Từ gốc repo, trên máy mới:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r submissions/2A202602810_NguyenThaiAnh/code/requirements.txt
SUB="submissions/2A202602810_NguyenThaiAnh"
python -m ipykernel install --prefix "$PWD/.venv" --name deepweeds --display-name 'DeepWeeds (.venv)'
```

Các notebook local dùng kernel `.venv/bin/python`. Kiểm tra prediction/Excel không cần GPU hoặc checkpoint.

## Kiểm tra bài nộp và tạo lại sản phẩm bằng CPU

```bash
SUB="submissions/2A202602810_NguyenThaiAnh"
.venv/bin/python "$SUB/code/check_submission.py"
.venv/bin/python "$SUB/code/export_submission.py"
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s "$SUB/code/tests" -v
```

Exporter đọc evidence và CSV, tạo Excel/hình/báo cáo; không train, fit T hoặc forward test. Khi thiếu ảnh, giữ gallery đã nộp; cần ảnh gốc chỉ để tạo lại gallery. Hai bộ test chạy trong hai process riêng vì test gốc kiểm tra evaluator/bộ khung còn test bài làm kiểm tra implementation. Giữ `starter/` và `eval.py` nguyên bản.

Chấm lại bằng evaluator gốc và CSV nguồn offline:

```bash
.venv/bin/python eval.py score --pred "$SUB/predictions/F01_seed*_test.csv" \
  --test-csv "$SUB/evidence/labels/test_subset0.csv" \
  --labels "$SUB/evidence/labels/labels.csv" --tag F01 --out eval_out
.venv/bin/python eval.py score --pred "$SUB/predictions/T00_seed*_test.csv" \
  --test-csv "$SUB/evidence/labels/test_subset0.csv" \
  --labels "$SUB/evidence/labels/labels.csv" --tag T00 --out eval_out
.venv/bin/python eval.py grade --final "$SUB/predictions/F01_seed*_test.csv" \
  --baseline "$SUB/predictions/T00_seed*_test.csv" \
  --uncal "$SUB/predictions/F01uncal_seed*_test.csv" \
  --final-val "$SUB/predictions/F01_seed*_val.csv" \
  --test-csv "$SUB/evidence/labels/test_subset0.csv" \
  --val-csv "$SUB/evidence/labels/val_subset0.csv" \
  --labels "$SUB/evidence/labels/labels.csv" \
  --latency-p95-ms 3.760881850394071 --latency-method proper --out eval_out
```

## Dữ liệu và EDA

Giữ bốn CSV nguyên byte, URL/SHA256 tại [data_sources.json](code/data_sources.json); bản sao ở `evidence/labels/`. Train10.501, val3.501, test3.507, giao rỗng/hợp17.509. Không chia lại/sửa nhãn. Một sai khác nguồn: `20170714-110407-3.jpg` có Label0 trong train split nhưng Label1 trong labels.csv; giữ nhãn split, ghi trong báo cáo.

Ảnh không commit. Tải [images.zip từ Zenodo](https://zenodo.org/records/7939060/files/images.zip?download=1), giải nén để ảnh ở **`data/*.jpg`**, CSV ở **`data/labels/`**. Có thể copy CSV nguyên bản từ evidence hoặc tải lại bằng script:

```bash
.venv/bin/python "$SUB/code/setup_data.py"
.venv/bin/python "$SUB/code/validate_setup.py" --decode-images
```

Notebook [EDA](code/notebooks/eda_deepweeds.ipynb) lưu hình ở `curves/eda/`, thống kê/ghi chú ở `code/validation/eda/`. Pixel/ảnh mẫu dùng train; val/test chỉ kiểm tra split/phân bố/dedup. Negative52,02% train, imbalance9,03×; JPEG byte duplicate0. Sanity đạt initialCE2,208589, overfit18 ảnh train40 steps → evalCE0,014244/top-1 100%; bằng chứng ở `code/validation/sanity/`, `curves/sanity/`.

## Thứ tự thực nghiệm trên checkout mới

**Dùng workspace riêng để tái huấn luyện.** Bản clone có prediction tham chiếu; lưu chúng cùng curves/evidence/Excel/báo cáo ra thư mục khác trước khi mở test mới. Không copy protocol/markers trong evidence vào `runs/`. Notebook Colab tự tách bản tham chiếu.

1. Setup → EDA → sanity → pilot nếu cần kiểm tra VRAM. `sanity_checks.py` chỉ dùng train.
2. Chạy năm notebook B01–B05, 12 epoch/batch64/seed0, test tắt. Cặp B03+B04 trước, B02+B05 sau, B01 cuối phù hợp VRAM pilot RTX3060; tốc độ song song chưa đo. Chạy tuần tự dễ theo dõi nhất.
3. Chạy [ablation tổng](code/notebooks/ablation_convnext_tiny.ipynb): T00 alias B03, T01–T07 đổi riêng yếu tố, khóa T08 bằng val. T04 tốt nhất, T08 CutMix+weighted CE không cải thiện.
4. Chạy [inference](code/notebooks/inference_T04_convnext_tiny.ipynb): chỉ val; latency khi GPU không train. Chọn I07 theo F1, hòa xét NLL/p95. `submission_benchmark.py` bổ sung latency backbone bằng tensor tổng hợp.
5. Chạy [final10](code/notebooks/final_10_convnext_tiny.ipynb): khóa manifest, reuse seed0 T04/B03, train seed1/2 F01/T00, fit T riêng trên val, seal sáu checkpoint rồi test một lượt/model/seed và score/grade.
6. `export_submission.py --snapshot` lưu log nhỏ từ runs và tạo sản phẩm; `check_submission.py` kiểm tra trước nộp.

CLI phần8–10, từ gốc repo:

```bash
.venv/bin/python "$SUB/code/ablation_notebook.py" --run --auto-t08
.venv/bin/python "$SUB/code/inference_notebook.py"
.venv/bin/python "$SUB/code/submission_benchmark.py"
.venv/bin/python "$SUB/code/final_notebook.py"
.venv/bin/python "$SUB/code/export_submission.py" --snapshot
```

Nếu runtime mới chọn recipe/method khác, dừng để lập protocol mới trước test; helper final hiện tại dành cho T04+I07. Không sửa manifest/checkpoint/code sau khi test đã mở.

## Resume, đọc lại và quy tắc test

B/T tự resume hoặc đọc run đã hoàn tất. `last.pt` lưu optimizer/scheduler/scaler/RNG; `best.pt` được val chọn. Giữ cùng config/runtime, không đổi recipe để resume. Raw trainer val của các seed final mới ở `predictions/training/`, tránh calibrated CSV ghi đè đầu ra trainer.

Phần9 đã hoàn tất: tắt `RUN_QUALITY`, `RUN_LATENCY` để chỉ đọc. Phần10 đã hoàn tất: đặt **`RUN_TRAINING=False`, `RUN_TEST=False`, `RUN_EVALUATOR=False`** để xem/audit. Test entry hoàn tất dùng raw cache, không forward lại. Marker test có nhưng raw cache chưa hoàn chỉnh thì runner chặn thử lại; không xóa marker.

T scalar dương fit trên val từng seed; F01/F01uncal lấy cùng raw logits. Manifest/seal/markers/completion/hash ở `evidence/runs/finalization/ConvNeXt_Final_v1/`. Snapshot phục vụ kiểm tra; resume máy gốc cần giữ `runs/` và checkpoint/cache gốc.

## Cấu trúc và nộp

```text
submissions/2A202602810_NguyenThaiAnh/
  results.xlsx, report.md, README.md
  code/          implementation, notebooks, tests, validation
  curves/        EDA, sanity, mỗi run B/T/F, inference, phân tích test
  predictions/   CSV val/test; raw trainer val ở training/
  evidence/      CSV nguồn, log/config/protocol/evaluator, bảng và SHA256
```

Notebook đã bỏ output nhúng để giảm kích thước; bảng/hình/log vẫn nằm trong sản phẩm/evidence. Hướng dẫn/helper tạm được dọn sau khi nội dung cần thiết chuyển vào README/notebook. Giữ dữ liệu/checkpoint/cache gốc, `.gitignore` loại khỏi commit. Không xóa nguồn thực nghiệm đã khóa.

Nộp thư mục bài làm theo yêu cầu lớp; checkpoint/dataset không nằm trong Git. Kiểm tra link Colab sau khi push lên `main`. Báo cáo nêu rõ kết quả âm, giới hạn một fold/seed sàng lọc, pretraining, domain shift và phạm vi latency.
