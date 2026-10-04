"""Source-grounded Vietnamese report, generated alongside results.xlsx."""
from export_submission import EVIDENCE, FINAL, SESSION, SUB, markdown_table, prediction, read_json, text_style

import numpy as np
import pandas as pd


def write_report(frames):
    final = frames["Final"]
    f = final[(final.exp_id == "F01") & (final.seed == "mean")].iloc[0]
    b = final[(final.exp_id == "T00") & (final.seed == "mean")].iloc[0]
    delta = f.macro_f1_test - b.macro_f1_test
    noise = max(f.macro_f1_test_std, b.macro_f1_test_std)
    uncal = pd.read_csv(EVIDENCE / "tables/test_calibration.csv")
    before_ece = uncal[uncal.tag == "F01uncal"].ece.mean()
    classes = frames["PerClass"]
    pcf = classes[(classes.exp_id == "F01") & (classes.seed == "mean")]
    pcb = classes[(classes.exp_id == "T00") & (classes.seed == "mean")]
    env = read_json(EVIDENCE / "runs/B03/seed0/environment.json")
    t = frames["Training"].set_index("exp_id")
    inf = frames["Inference"].set_index("exp_id")
    backbones = frames["Backbones"].set_index("exp_id")
    valgap = f.macro_f1_test - f.macro_f1_val
    pred, metrics = prediction("F01_seed0_test.csv", "test")
    hard_forward = int(metrics["confusion"][0, 7])
    hard_reverse = int(metrics["confusion"][7, 0])
    top_pairs = pd.read_csv(EVIDENCE / "tables/F01_seed0_error_pairs.csv").head(5)
    best_seeds = final[(final.exp_id == "F01") & (final.seed != "mean")][["seed", "source_id", "temperature_fit_val", "macro_f1_val", "macro_f1_test", "top1_test", "ece_test"]]
    baseline_seeds = final[(final.exp_id == "T00") & (final.seed != "mean")][["seed", "source_id", "macro_f1_val", "macro_f1_test", "top1_test", "ece_test"]]
    counts = pd.read_csv(SUB / "code/validation/eda/class_counts.csv")
    report = f"""# DeepWeeds: backbone, công thức huấn luyện và suy luận

**Nguyễn Thái Anh · 2A202602810 · Track 4, Lab Day 2**
Fold 0 · GPU {env['gpu']} · seed sàng lọc 0 · seed chung kết 0, 1, 2.

## 1. Tóm tắt

Thực nghiệm phân loại chín lớp DeepWeeds bằng năm backbone, chín công thức ConvNeXt-Tiny
(gồm mốc), năm phương pháp suy luận và chung kết ba seed cho cả mốc lẫn cấu hình cuối.
Mọi lựa chọn và nhiệt độ đều dựa trên validation trước khi mở test.
Cấu hình F01 là ConvNeXt-Tiny `in12k_ft_in1k`, fine-tune với CutMix α=1 và CE,
suy luận một view FP32 với temperature scaling (I07).
Test đạt top-1 **{f.top1_test*100:.4f}% ± {f.top1_test_std*100:.4f} điểm %** và
macro-F1 **{f.macro_f1_test*100:.4f}% ± {f.macro_f1_test_std*100:.4f} điểm %** (mean ± std mẫu, ba seed).
Macro-F1 tăng **{delta*100:.4f} điểm %** so với T00 cùng backbone và số seed.
Độ trễ p95 đại diện seed0 là **{f.latency_p95_batch1_ms_representative:.4f} ms** trong phạm vi GPU pipeline.
Kết hợp CutMix và weighted CE không cải thiện; TTA tăng chi phí nhưng chưa tăng F1 ở seed sàng lọc.

## 2. Dữ liệu, thiết lập và tính chặt chẽ

DeepWeeds gồm 17.509 ảnh, tám loài cỏ và lớp Negative. Bài làm giữ nguyên CSV fold0 của tác giả:
train 10.501, validation 3.501, test 3.507; ba giao theo filename rỗng và hợp đủ 17.509.
Audit không thiếu ảnh hoặc có JPEG trùng byte; toàn bộ 10.501 ảnh train giải mã được, RGB 256×256.
Các thống kê pixel, ảnh mẫu và kiểm tra augmentation chỉ dùng train.

{markdown_table(counts, ['Species', 'train', 'val', 'test', 'all_splits', 'paper_reference'])}

`paper_reference` đối chiếu Table 1 của [Olsen và cộng sự (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6375952/).
Một sai khác nguồn được giữ nguyên: `20170714-110407-3.jpg` có nhãn 0 trong train split nhưng
nhãn 1 trong `labels.csv`; do đó tổng Chinee apple/Lantana lệch ±1 so với Table 1.
Không sửa nhãn hoặc chia lại. Negative chiếm 52,02% train, tỷ lệ lớp lớn nhất/nhỏ nhất 9,03×.
Macro-F1 chín lớp là tiêu chí chính; top-1 đi kèm chỉ số từng lớp để tránh diễn giải sai vì mất cân bằng.

![Phân bố lớp](curves/eda/class_distribution.png)

Công thức nền: input224; train RandomResizedCrop + hflip; val/test resize256 + CenterCrop224,
chuẩn hóa theo `pretrained_cfg`. AdamW với LR backbone 1e-4, head 1e-3,
weight decay0,05 (norm/bias được miễn), warmup1 epoch rồi cosine theo optimizer step,
12 epoch, batch64, gradient clip1, AMP khi train, không EMA/sampler cân bằng.
Checkpoint có macro-F1 val cao nhất được chọn; hòa lấy epoch sớm nhất.
`drop_last=True` khiến mỗi epoch train dùng 10.496/10.501 ảnh, thứ tự shuffle theo seed.
Đánh giá luôn dùng toàn bộ val/test, `eval()` và không gradient.

Môi trường thực tế: Python {env['python']}, torch {env['torch_build']}, CUDA {env['cuda_build']},
torchvision {env['versions']['torchvision']}, timm {env['versions']['timm']},
numpy {env['versions']['numpy']}, pandas {env['versions']['pandas']}.
Seed và deterministic algorithms được bật, cuDNN benchmark tắt. Phiên bản đầy đủ nằm trong
`code/requirements.lock.txt` và environment từng run. Thời gian train có thể chịu tranh chấp GPU,
vì vậy chỉ dùng như thời gian quan sát; latency được đo riêng khi không có compute job khác.

Sanity trên 18 ảnh train (2/lớp), ResNet-50 pretrained, seed42:
initial CE2,208589 gần ln9=2,197225; overfit40 updates đạt eval CE0,014244 và top-1 100%.
Đã kiểm tra gradient/optimizer/frozen state, focal γ=0 tương đương CE,
LS ε=0 tương đương CE, Mixup/CutMix trộn cả nhãn, CSV round-trip và eval không đổi model.
AMP có một số update bị GradScaler skip trong các epoch đầu; loss/trọng số vẫn hữu hạn.
Audit pilot tiếp theo đạt 820/820 updates không skip. Số skip từng run nằm trong history và bảng Training.

## 3. So sánh năm backbone

Các model dùng cùng split, seed0, 12 epoch, batch64 và recipe nền; tag trọng số được ghi riêng.
Các giá trị F1/top-1 trong bảng dưới là %, thời gian là giây train/epoch quan sát,
latency là p95 batch1 trong GPU pipeline, cùng input resident256 → crop224 → model → softmax.

{markdown_table(frames['Backbones'], ['exp_id', 'backbone', 'pretrained_tag', 'params_M', 'GMAC', 'macro_f1_val', 'top1_val', 'train_s_per_epoch_observed', 'latency_p95_batch1_ms'], percent=['macro_f1_val', 'top1_val'])}

![Đánh đổi backbone](curves/final_analysis/B_backbones_tradeoff.png)

B03 ConvNeXt-Tiny có macro-F1 val cao nhất, hơn ResNet-50
{(backbones.loc['B03','macro_f1_val']-backbones.loc['B01','macro_f1_val'])*100:.4f} điểm %,
và hơn DeiT-Small {(backbones.loc['B03','macro_f1_val']-backbones.loc['B04','macro_f1_val'])*100:.4f} điểm %.
Chi phí {backbones.loc['B03','params_M']:.3f}M tham số, {backbones.loc['B03','GMAC']:.3f} GMAC và
p95 {backbones.loc['B03','latency_p95_batch1_ms']:.3f} ms vẫn nằm trong ngân sách30–100ms đã nêu.
EfficientNet-B0 nhẹ và nhanh hơn, nhưng F1 thấp hơn rõ ở lần sàng lọc này.
Chọn B03 để tiếp tục vì cả chất lượng và chi phí phù hợp phần cứng hiện tại.

Đây là so sánh pipeline dùng các trọng số có sẵn, không cô lập riêng tác động kiến trúc:
ConvNeXt dùng `in12k_ft_in1k`, các backbone khác dùng tag và công thức pretraining khác.
GMAC là kết quả fvcore, một multiply-add tính một operation; các operator không hỗ trợ
được liệt kê trong `mac_profile.json`, nên số này không đại diện mọi phép toán thực thi.
Mỗi backbone chỉ có một seed; chưa có std để khẳng định ý nghĩa thống kê của các chênh lệch nhỏ.

## 4. Ablation công thức huấn luyện

T00 là alias B03/seed0, không train thêm. T01–T07 mỗi run chỉ đổi một yếu tố;
T08 được chọn từ validation để kiểm tra kết hợp. Ba trục chính là khởi tạo,
augmentation và loss, mỗi trục có ít nhất hai giá trị. Tất cả giữ backbone/split/seed0,
12 epoch, batch64 và LR giống mốc.

{markdown_table(frames['Training'], ['exp_id', 'axis', 'description', 'macro_f1_val', 'top1_val', 'delta_macro_f1_pp'], percent=['macro_f1_val','top1_val'])}

Fine-tune vượt frozen/scratch rõ trong ngân sách12 epoch. Scratch đạt
{t.loc['T01','macro_f1_val']*100:.4f}%, frozen {t.loc['T02','macro_f1_val']*100:.4f}%,
cho thấy pretraining và cập nhật backbone hữu ích trong thiết lập này; không suy ra scratch
sẽ kém khi được train lâu hơn. Color, label smoothing, focal và weighted CE đều được giữ
trong bảng dù mức tăng khác nhau. Weighted CE dùng inverse-frequency tính từ train,
chuẩn hóa weight về trung bình1; không sử dụng phân bố nhãn val/test để đặt weight.

CutMix T04 đạt {t.loc['T04','macro_f1_val']*100:.4f}%, tăng
{t.loc['T04','delta_macro_f1_pp']:.4f} điểm % và là recipe được chọn.
T08 = CutMix T04 + weighted CE T07 chỉ đạt {t.loc['T08','macro_f1_val']*100:.4f}%,
thấp hơn T04 {(t.loc['T04','macro_f1_val']-t.loc['T08','macro_f1_val'])*100:.4f} điểm %,
và gần mốc (Δ={t.loc['T08','delta_macro_f1_pp']:+.4f} điểm %).
Hai thành phần tốt riêng lẻ không cộng dồn ở seed0. Một giả thuyết là class weight
làm thay đổi đóng góp của nhãn đã trộn trong CutMix; cần lặp seed để kiểm chứng,
chưa quy kết đây là nguyên nhân đã được xác nhận.

Ngưỡng chọn hai thành phần T08 là Δ≥0,002 trên val, chỉ là ngưỡng thực dụng,
không phải phép kiểm định thống kê. Các ablation chưa có std riêng; std final phía dưới
chỉ gợi ý quy mô dao động, không thay thế độ nhiễu của từng T-run.
Loss train giữa CE/focal/weighted CE/CutMix có ý nghĩa khác nhau, nên so sánh bằng F1
và NLL thay vì so trực tiếp độ lớn loss. Đường cong từng run nằm ở `curves/T*_seed0_*.png`.

## 5. Suy luận, hiệu chuẩn và đánh đổi chi phí

Cả năm phương pháp dùng checkpoint T04/seed0, đủ 3.501 ảnh validation:
I00 một center crop; I01 crop gốc+lật ngang, trung bình xác suất;
I02 năm crop224 khác vị trí từ nguồn256, trung bình xác suất;
I07 scalar temperature scaling trên logits I00; I08 autocast FP16, không fusion.
ConvNeXt dùng LayerNorm nên Conv–BN fusion không áp dụng. I01/I02/I07/I08 là
bốn phương pháp ngoài mốc. T được fit bằng NLL trên val và không đổi argmax.

{markdown_table(frames['Inference'], ['exp_id','k_views','dtype','macro_f1','ece','nll','latency_p50_ms','latency_p95_ms','latency_p99_ms','throughput_images_per_s','relative_p50_cost'], percent=['macro_f1'])}

![Đánh đổi suy luận](curves/inference/I_T04_seed0/f1_vs_latency_p95_detail.png)

TTA hflip/5-crop không tăng macro-F1 ở seed0, dù NLL giảm; p50 tăng lần lượt
{inf.loc['I01','relative_p50_cost']:.2f}×/{inf.loc['I02','relative_p50_cost']:.2f}×.
I08 giữ F1 ở lần sàng lọc và tăng throughput batch32 lên
{inf.loc['I08','throughput_images_per_s']:.1f} ảnh/s, nhưng ECE
{inf.loc['I08','ece']:.6f} cao hơn I07 {inf.loc['I07','ece']:.6f}.
Chọn I07 theo quy tắc khóa trước: F1 giảm dần, hòa xét NLL tăng dần,
sau đó p95 và ID. T seed0={inf.loc['I07','temperature']:.8f}, ECE val
{inf.loc['I00','ece']:.6f}→{inf.loc['I07','ece']:.6f}; F1 và nhãn dự đoán giữ nguyên.
Không quy mức tăng F1 cho temperature scaling. I08 chưa đánh giá trên test,
vì vậy chưa kết luận triển khai FP16 sẽ giữ nguyên chất lượng test.

Benchmark trên RTX3060: 10 warmup, 100 lần đo, synchronize trước/sau mỗi lần,
batch1 và32. Timing bao gồm GPU crop/views, forward, aggregation/calibration và softmax;
input256 đã chuẩn hóa và ở GPU. Không gồm đọc ảnh, CPU preprocessing, H2D hoặc fit T offline.
p95 I07 {inf.loc['I07','latency_p95_ms']:.4f}ms là số đại diện cùng kiến trúc/pipeline seed0,
không phải end-to-end camera và không phải trung bình latency qua ba seed.
Chênh p95 nhỏ giữa I00/I07 có thể do nhiễu timing; p50 I07 thực tế cao hơn nhẹ.
Đề xuất I07 cho ngân sách30–100ms trên GPU đã đo. Khi chuyển robot/phần cứng khác,
cần đo lại toàn chuỗi camera→decision và kiểm tra chất lượng tại hiện trường.

## 6. Chung kết, từng lớp và phân tích lỗi

Manifest khóa recipe/seed/split/preprocessing/code/evaluator từ validation; sau đó
huấn luyện seed1/2 và seal đủ sáu checkpoint cùng nhiệt độ trước khi mở test.
Seed0 F01 reuse T04, T00 reuse B03, có hash/alias rõ. Mỗi model/seed forward toàn bộ
3.507 ảnh test đúng một lần (sáu lượt cho hai nhóm × ba seed), FP32, một view.
Dự đoán F01 đã hiệu chuẩn và F01uncal lấy từ cùng raw logits;
không fit T hoặc chọn checkpoint trên test. Export/score lại chỉ đọc cache/CSV.
Temperature từng seed F01 fit riêng trên val, T00 giữ T=1.

Final F01 (F1/top-1 là %, ECE là tỷ lệ):

{markdown_table(best_seeds, list(best_seeds.columns), percent=['macro_f1_val','macro_f1_test','top1_test'])}

Baseline T00:

{markdown_table(baseline_seeds, list(baseline_seeds.columns), percent=['macro_f1_val','macro_f1_test','top1_test'])}

| Cấu hình | Top-1 test (%) | Macro-F1 test (%) | ECE test | NLL test |
|---|---:|---:|---:|---:|
| T00 | {b.top1_test*100:.4f} ± {b.top1_test_std*100:.4f} | {b.macro_f1_test*100:.4f} ± {b.macro_f1_test_std*100:.4f} | {b.ece_test:.6f} ± {b.ece_test_std:.6f} | {b.nll_test:.6f} ± {b.nll_test_std:.6f} |
| F01 | {f.top1_test*100:.4f} ± {f.top1_test_std*100:.4f} | {f.macro_f1_test*100:.4f} ± {f.macro_f1_test_std*100:.4f} | {f.ece_test:.6f} ± {f.ece_test_std:.6f} | {f.nll_test:.6f} ± {f.nll_test_std:.6f} |

Mọi std dùng ddof=1. ΔF1={delta*100:.4f} điểm %, lớn hơn std lớn nhất hai nhóm
{noise*100:.4f} điểm % ({delta/noise:.2f}×). Đây là bằng chứng cải thiện nhất quán trong
ba seed đã đo, không phải kiểm định ý nghĩa thống kê hoặc đảm bảo trên miền mới.
Macro-F1 val final mean={f.macro_f1_val*100:.4f}%, thấp hơn test {valgap*100:.4f} điểm %;
chênh tuyệt đối {abs(valgap):.6f} nằm dưới0,02. Khác biệt độ khó giữa hai tập có thể góp phần;
không dùng test cao hơn để thay đổi cấu hình.

![Ma trận nhầm lẫn final](curves/final_analysis/F01_confusion_test.png)

Confusion count là **tổng ba seed, support10.521**, mỗi ảnh xuất hiện một lần/seed;
không gọi đó là 10.521 ảnh độc lập. Ma trận chuẩn hóa theo hàng mô tả tỷ lệ trên tổng count.
F1/precision/recall trong bảng là mean qua seed, không tính từ confusion cộng dồn.
Ma trận baseline lưu tại `curves/final_analysis/T00_confusion_test.png`.

{markdown_table(pcf, ['class','support_per_seed','precision','recall','recall_std','f1','f1_std'], percent=['precision','recall','recall_std','f1','f1_std'])}

![F1 từng lớp](curves/final_analysis/F01_vs_T00_per_class_test.png)

Recall Chinee apple tăng từ {pcb[pcb.class_id==0].recall.iloc[0]*100:.4f}% lên
{pcf[pcf.class_id==0].recall.iloc[0]*100:.4f}% ± {pcf[pcf.class_id==0].recall_std.iloc[0]*100:.4f} điểm %;
Snake weed từ {pcb[pcb.class_id==7].recall.iloc[0]*100:.4f}% lên
{pcf[pcf.class_id==7].recall.iloc[0]*100:.4f}% ± {pcf[pcf.class_id==7].recall_std.iloc[0]*100:.4f} điểm %.
Cả hai vượt các mốc tham chiếu88,5%/88,8% được đề bài lấy từ bài báo.
So sánh chỉ mang tính tham khảo vì bài báo dùng năm fold và thiết lập khác.
Precision Prickly acacia/Snake weed thấp hơn recall, cho thấy vẫn có false positive từ lớp khác.

Seed0 final có {(pred.y_true != pred.y_pred).sum()} ảnh sai; Chinee apple→Snake weed
{hard_forward} ảnh, chiều ngược lại {hard_reverse} ảnh. Năm cặp lỗi nhiều nhất seed0:

{markdown_table(top_pairs, list(top_pairs.columns))}

![Ảnh lỗi seed0](curves/final_analysis/F01_seed0_error_gallery.png)

Gallery chọn cố định từ CSV: ưu tiên cặp0↔7, tiếp theo lỗi của hai lớp khó,
rồi lỗi có confidence cao; filename/nhãn/confidence đầy đủ ở
`evidence/tables/F01_seed0_error_gallery.csv`, toàn bộ lỗi ở `F01_seed0_test_errors.csv`.
Ảnh nguyên bản256×256 chứa cây nhỏ xen nền lá/cỏ và vùng sáng tối; center crop có thể
làm mất bộ phận phân biệt. Các loài khác nhau cũng có cấu trúc lá/hình dạng tương tự.
Đây là giả thuyết từ quan sát lỗi, chưa kiểm chứng bằng crop/segmentation hoặc gán nhãn vị trí.
Confidence cao ở một số lỗi cho thấy hiệu chuẩn toàn tập không bảo đảm mọi ảnh tin cậy.
Phân tích này thực hiện sau khi khóa và chấm test, không đưa ngược vào chọn model của bài.

![Reliability test](curves/final_analysis/F01_test_reliability_before_after.png)

ECE test trung bình giảm **{before_ece:.6f}→{f.ece_test:.6f}** sau TS,
với NLL final {f.nll_test:.6f}. Reliability dùng15 bin đều và vẽ riêng từng seed;
bin ít ảnh có dao động lớn, số count được lưu trong CSV.
Nhiệt độ fit trên validation nên test chỉ dùng đánh giá độ chuyển giao hiệu chuẩn.
T scalar dương giữ thứ tự logits và nhãn dự đoán; cải thiện này thuộc chất lượng xác suất.

## 7. Kết luận, giới hạn và việc tiếp theo

F01 ConvNeXt-Tiny + CutMix + I07 là cấu hình được khóa trên validation và xác nhận
bằng ba seed test. Thay backbone/pretraining tạo chênh chất lượng lớn nhất trong bảng
sàng lọc; trong cùng ConvNeXt, CutMix tạo mức tăng F1 lớn nhất ở một yếu tố đã thử.
Suy luận I07 cải thiện hiệu chuẩn với chi phí gần một view, không tăng F1.
T08 là kết quả âm cần giữ lại; TTA chưa cho lợi ích F1 để bù chi phí ở seed này.
Lựa chọn realtime hiện tại là I07 FP32 trên RTX3060, p95 đại diện dưới ngân sách30–100ms;
offline cũng dùng I07 vì TTA đã thử chưa tăng F1. I08 là ứng viên throughput cần
kiểm chứng riêng trong thực nghiệm tương lai, chưa thay final đã khóa.

Giới hạn: chỉ một fold; B/T/I sàng lọc một seed; ba seed final còn ít và seed0 đã tham gia
chọn recipe, nên kết quả final chưa độc lập hoàn toàn khỏi quá trình sàng lọc.
Pretraining khác tag/dữ liệu ngăn kết luận nhân quả thuần về kiến trúc.
DeepWeeds chia ngẫu nhiên theo ảnh, không hold-out địa điểm; hiệu năng có thể lạc quan
khi triển khai ở vùng/mùa/camera mới. Không có kiểm chứng domain shift, mobile GPU,
end-to-end latency hoặc độ trễ thực của robot. ECE phụ thuộc binning và class imbalance.
Ảnh không trùng byte vẫn có thể gần nhau về cảnh; audit này chưa kiểm tra near-duplicates.

Tiếp theo nên lặp ablation quan trọng nhiều seed, đánh giá hold-out địa điểm trong protocol mới,
thu thập lỗi khó và kiểm chứng crop/segmentation, đo H2D/CPU/camera, rồi kiểm tra FP16
và hiệu chuẩn trên phần cứng triển khai. Các đề xuất này không thay đổi kết quả test đã báo cáo.

## 8. Phụ lục và khả năng tái lập

- [Excel7 sheet](results.xlsx): Backbones, Training, Inference, Final, PerClass, Latency, Summary.
- `evidence/runs/`: config, model/pretraining, history, summary, ablation decisions,
  protocol/hash/seal/test markers và evaluator gốc. `evidence/source_index.json` ghi SHA256 các bản sao.
- `predictions/`: đầy đủ val và test; F01/T00/F01uncal, seed0/1/2. Raw trainer val riêng trong `predictions/training/`.
- [PDF 8 trang](report.pdf), [README tái lập](README.md) và [notebook Colab](code/notebooks/reproduce_colab.ipynb).
- `code/export_submission.py`: tạo lại Excel/hình/báo cáo bằng CPU từ CSV/evidence,
  không load model, train, fit T hoặc forward test. Checkpoint/dataset/cache không commit.
- Các curve F01seed0/T00seed0 là biểu đồ từ history T04/B03 được reuse, không giả lập run mới.
- `evidence/runs/finalization/ConvNeXt_Final_v1/eval_out/grade_I.json`: tự chấm mục I
  **19/20**, không phải điểm toàn bài hoặc điểm giảng viên đã xác nhận.

Danh sách ID: B01ResNet50, B02ResNeXt50, B03ConvNeXtTiny, B04DeiTSmall,
B05EfficientNetB0; T00baseline, T01scratch, T02frozen, T03color, T04CutMix,
T05LS, T06focal, T07weightedCE, T08combination; I00center, I01hflip,
I02five-crop, I07TS, I08AMP; F01=T04+I07. Chi tiết config có trong Excel và evidence từng run.

Tham khảo dữ liệu: [Olsen et al., *DeepWeeds*, Scientific Reports9,2058(2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6375952/),
[CSV của tác giả](https://github.com/AlexOlsen/DeepWeeds/tree/master/labels),
và GUIDE/RUBRIC trong repo. Bộ khung `starter/` và `eval.py` giữ nguyên.
"""
    (SUB / "report.md").write_text(text_style(report), encoding="utf-8")
