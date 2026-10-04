"""Eight-page PDF companion to the detailed Markdown report, from the same tables."""
from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from PIL import Image

from export_submission import EVIDENCE, SUB, prediction, reliability, text_style


def write_pdf(frames, images_dir):
    final = frames["Final"]
    f = final[(final.exp_id == "F01") & (final.seed == "mean")].iloc[0]
    b = final[(final.exp_id == "T00") & (final.seed == "mean")].iloc[0]
    names = frames["PerClass"].drop_duplicates("class_id").sort_values("class_id")["class"].tolist()
    pdf = PdfPages(SUB / "report.pdf", metadata={"Title": "DeepWeeds — Nguyễn Thái Anh", "Author": "Nguyễn Thái Anh (2A202602810)"})

    def page(number, title):
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(.07, .967, "DEEPWEEDS  /  TRACK 4 · LAB DAY 2", fontsize=9, color="#33516F")
        fig.text(.07, .929, title, fontsize=16, weight="bold", color="#18324F")
        fig.text(.07, .023, "Nguyễn Thái Anh · 2A202602810  |  Fold 0  |  xem report.md và results.xlsx", fontsize=8, color="#33516F")
        fig.text(.92, .023, f"{number}/8", fontsize=8, ha="right")
        return fig

    def block(fig, text, y, width=100):
        text = text_style(text)
        wrapped = []
        for para in text.split("\n"):
            wrapped.extend(textwrap.wrap(para, width=width) or [""])
        fig.text(.07, y, "\n".join(wrapped), fontsize=9.6, va="top", linespacing=1.45)
        return y - len(wrapped) * .0167

    def table(fig, frame, columns, labels, box, percent=()):
        values = []
        for row in frame.to_dict("records"):
            values.append(["—" if pd.isna(row[col]) else f"{row[col]*100:.2f}" if col in percent
                           else f"{row[col]:.3f}" if isinstance(row[col], (float, np.floating)) else str(row[col]) for col in columns])
        ax = fig.add_axes(box)
        ax.axis("off")
        t = ax.table(cellText=values, colLabels=labels, cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
        t.auto_set_font_size(False)
        t.set_fontsize(8)
        for (row, col), cell in t.get_celld().items():
            cell.set_edgecolor("#D6DFE7")
            if row == 0:
                cell.set_facecolor("#18324F")
                cell.set_text_props(color="white", weight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#F0F4F8")

    def picture(fig, path, box):
        ax = fig.add_axes(box)
        ax.imshow(plt.imread(path))
        ax.axis("off")

    def finish(fig):
        pdf.savefig(fig)
        plt.close(fig)

    fig = page(1, "Dữ liệu, thiết lập và kết quả chính")
    block(fig, f"Final F01: ConvNeXt-Tiny in12k_ft_in1k + CutMix α=1 + CE + I07. Test top-1 {f.top1_test*100:.2f}% ± {f.top1_test_std*100:.2f} điểm %; macro-F1 {f.macro_f1_test*100:.2f}% ± {f.macro_f1_test_std*100:.2f} điểm %. Mean ± std mẫu (ddof=1), seed0/1/2. Baseline cùng backbone đạt F1 {b.macro_f1_test*100:.2f}%, Δ={(f.macro_f1_test-b.macro_f1_test)*100:.3f} điểm %.\n\nGiữ CSV fold0 nguyên byte: train10501, val3501, test3507; giao rỗng/hợp17509, không thiếu ảnh hoặc JPEG trùng byte. Negative52,02% train; tỷ lệ lớn nhất/nhỏ nhất9,03×. Một nhãn train nguồn khác labels.csv được giữ và ghi rõ trong report.md. Đối chiếu Table1 của Olsen et al. (2019).", .88)
    counts = pd.read_csv(SUB / "code/validation/eda/class_counts.csv")
    table(fig, counts, ["Species", "train", "val", "test", "paper_reference"], ["Lớp", "Train", "Val", "Test", "Table1"], [.07, .42, .86, .26])
    block(fig, "Thiết lập: input224, train crop+flip, val/test resize256→center crop224; AdamW LR backbone1e-4/head1e-3, WD0,05 (miễn norm/bias), warmup1 + cosine,12 epoch,batch64,AMP train,clip1,không EMA. Best checkpoint theo val F1, hòa chọn epoch sớm.\n\nRTX3060/12GB; Python3.14.7, torch2.10.0+cu128, torchvision0.25.0, timm1.0.30. Seed cố định, deterministic bật. Sanity overfit18 ảnh train/40 steps: evalCE0,014244/top-1 100%; loss ban đầu2,208589 gần ln9. Kiểm tra gradient/frozen/CutMix/focalγ=0/CSV đạt. Không dùng sanity đo chất lượng tổng quát.", .38)
    picture(fig, SUB / "curves/eda/class_distribution.png", [.1, .07, .8, .16])
    finish(fig)

    fig = page(2, "Năm backbone: chất lượng và chi phí")
    block(fig, "Cùng split/seed0/recipe12 epoch,batch64; khác kiến trúc và tag pretraining. F1/top-1 trong bảng là %. GMAC dùng fvcore, một multiply-add=1 operation; operator không hỗ trợ ghi trong evidence. Train s/epoch là số quan sát, có thể tranh chấp GPU.", .88)
    bb = frames["Backbones"].copy()
    bb["short"] = ["ResNet50", "ResNeXt50", "ConvNeXtTiny", "DeiTSmall", "EfficientNetB0"]
    table(fig, bb, ["exp_id", "short", "params_M", "GMAC", "macro_f1_val", "top1_val", "train_s_per_epoch_observed", "latency_p95_batch1_ms"],
          ["ID", "Model", "M params", "GMAC", "F1 val%", "Top1%", "s/epoch", "p95 ms"], [.07, .63, .86, .15], percent=["macro_f1_val", "top1_val"])
    block(fig, "Tag chính xác từng backbone được ghi ở sheet Backbones/model.json; ConvNeXt dùng in12k_ft_in1k. Vì dữ liệu/công thức pretraining khác, đây là so sánh pipeline có sẵn, không tách riêng tác động kiến trúc.\n\nChọn B03 vì F1 val cao nhất và độ trễ vẫn phù hợp ngân sách30–100ms trên GPU đã đo. EfficientNet nhẹ/nhanh hơn nhưng F1 thấp hơn. Các backbone chỉ sàng lọc một seed; chưa có std để kết luận các chênh lệch nhỏ có ý nghĩa thống kê.", .58)
    picture(fig, SUB / "curves/final_analysis/B_backbones_tradeoff.png", [.07, .16, .86, .28])
    block(fig, "Latency đo riêng:10 warmup,100 synchronized samples, input256 đã ở GPU→crop224→model→softmax. Không gồm CPU/disk/H2D; không so thời gian train đồng thời như latency. Xem Latency cho batch1/32 và raw timing samples trong validation/evidence.", .13)
    finish(fig)

    fig = page(3, "Ablation: ba trục và một kết hợp")
    training = frames["Training"].copy()
    training["short"] = ["Mốc B03", "Scratch", "Frozen", "Color", "CutMix α1", "LS ε0,1", "Focal γ2", "Weighted CE", "CutMix+WCE"]
    table(fig, training, ["exp_id", "axis", "short", "macro_f1_val", "delta_macro_f1_pp"],
          ["ID", "Trục", "Thay đổi", "F1 val%", "Δ điểm %"], [.07, .61, .86, .26], percent=["macro_f1_val"])
    block(fig, "T00 reuse B03. T01–T07 mỗi run chỉ khác một yếu tố; T08 kết hợp T04+T07 được chọn bằng val (ngưỡng Δ≥0,002 chỉ là thực dụng, không phải kiểm định). Giữ split/seed0/epoch/batch/LR. Weighted CE tính weight từ train.\n\nT04 CutMix là recipe tốt nhất. Scratch/frozen thấp hơn fine-tune trong12 epoch. T08 không cộng dồn lợi ích hai thành phần, thấp hơn T04 khoảng0,884 điểm %. Giả thuyết: weight và nhãn trộn thay đổi đóng góp từng lớp; cần lặp seed để kiểm chứng.\n\nChưa có std riêng cho các ablation, không dùng std final thay thế độ nhiễu T-run. Loss của các criterion có ý nghĩa khác nhau; đánh giá bằng F1/NLL.", .56)
    ax = fig.add_axes([.12, .12, .76, .25])
    for exp_id in ("B03", "T04", "T08"):
        h = pd.read_csv(EVIDENCE / "runs" / exp_id / "seed0/history.csv")
        ax.plot(h.epoch, h.val_macro_f1 * 100, label=exp_id)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro-F1 (%)")
    ax.legend()
    ax.grid(alpha=.2)
    finish(fig)

    fig = page(4, "Suy luận và temperature scaling")
    block(fig, "T04/seed0, đủ3501 ảnh val. I00 center224; I01 hai view gốc+lật; I02 năm crop từ nguồn256 (gộp xác suất); I07 T fit NLL trên val; I08 autocastFP16. ConvNeXt LayerNorm nên BN fusion không áp dụng. Ngoài mốc có bốn phương pháp.", .88)
    table(fig, frames["Inference"], ["exp_id", "k_views", "macro_f1", "ece", "nll", "latency_p95_ms", "throughput_images_per_s"],
          ["ID", "K", "F1 val%", "ECE", "NLL", "p95 ms", "img/s b32"], [.07, .65, .86, .15], percent=["macro_f1"])
    block(fig, "TTA giảm NLL nhưng chưa tăng F1 và tăng chi phí gần2×/5×. I08 giữ F1 ở val seed0, throughput tốt hơn nhưng ECE cao hơn I07; chưa đo test I08. Chọn I07 theo F1, hòa xét NLL→p95→ID. T scalar dương không đổi argmax; hiệu quả thuộc xác suất, không phải F1.\n\n10 warmup +100 lần sync trước/sau, batch1/32, GPU không có compute job khác. Scope: input resident256→crop/views→forward→gộp/calibration→softmax, bỏ CPU/disk/H2D và fitT offline. Chênh p95 nhỏ I00/I07 không chứng minh TS làm model nhanh hơn.", .6)
    picture(fig, SUB / "curves/inference/I_T04_seed0/f1_vs_latency_p95_detail.png", [.07, .12, .86, .31])
    block(fig, "Khuyến nghị I07 FP32 cho ngân sách30–100ms trên RTX3060. Đo end-to-end camera→decision và kiểm tra chất lượng lại khi chuyển sang robot/miền mới. I08 là ứng viên throughput cho protocol tương lai, chưa thay final đã khóa.", .11)
    finish(fig)

    fig = page(5, "Final ba seed và hiệu chuẩn test")
    block(fig, "Khóa manifest trên val, train/prepare đủ sáu mục rồi seal checkpoint/T trước test. T00seed0 reuse B03, F01seed0 reuse T04. Một raw test pass/model/seed; calibrated/uncal dùng cùng logits. Không fit T/chọn recipe trên test.", .88)
    table(fig, final, ["exp_id", "seed", "macro_f1_val", "macro_f1_test", "macro_f1_test_std", "top1_test", "ece_test"],
          ["ID", "Seed", "F1 val%", "F1 test%", "std pp", "Top1%", "ECE"], [.07, .6, .86, .21], percent=["macro_f1_val", "macro_f1_test", "macro_f1_test_std", "top1_test"])
    calibration = pd.read_csv(EVIDENCE / "tables/test_calibration.csv")
    before = calibration[calibration.tag == "F01uncal"].ece.mean()
    block(fig, f"Std mẫu ddof=1; mỗi seed3507 ảnh. ΔF1={(f.macro_f1_test-b.macro_f1_test)*100:.4f} điểm %, lớn hơn std lớn nhất hai nhóm {max(f.macro_f1_test_std,b.macro_f1_test_std)*100:.4f} điểm %. Chưa phải kiểm định ý nghĩa thống kê. Gap test−val mean={(f.macro_f1_test-f.macro_f1_val)*100:.4f} điểm %. ECE test mean {before:.6f}→{f.ece_test:.6f}. T fit riêng trên val từng seed, giữ nhãn dự đoán.", .55)
    ax = fig.add_axes([.12, .12, .76, .3])
    ax.plot([0, 1], [0, 1], "--", color="gray")
    for tag, label in (("F01uncal", "Before TS"), ("F01", "After TS")):
        bins = reliability(prediction(f"{tag}_seed0_test.csv", "test")[0])
        bins = bins[bins["count"] > 0]
        ax.plot(bins.confidence, bins.accuracy, "o-", label=label)
    ax.set_xlabel("Mean confidence (nonempty bins)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability test seed0 | 15 bins, T fitted on val", fontsize=10)
    ax.legend()
    finish(fig)

    fig = page(6, "Từng lớp và ma trận nhầm lẫn")
    pc = frames["PerClass"]
    pc = pc[(pc.exp_id == "F01") & (pc.seed == "mean")].sort_values("class_id")
    table(fig, pc, ["class", "support_per_seed", "precision", "recall", "f1", "f1_std"],
          ["Lớp", "Support", "P%", "R%", "F1%", "std pp"], [.07, .66, .86, .21], percent=["precision", "recall", "f1", "f1_std"])
    block(fig, "Chinee apple/Snake weed có recall cải thiện so với baseline (bảng đầy đủ ở PerClass). Precision Snake weed/Prickly acacia thấp hơn recall; còn false positive từ lớp khác. Metric từng lớp là mean/std qua seed, không tính trên confusion gộp.", .62)
    cm = sum(prediction(f"F01_seed{s}_test.csv", "test")[1]["confusion"] for s in (0, 1, 2))
    normalized = cm / cm.sum(1, keepdims=True) * 100
    ax = fig.add_axes([.2, .15, .63, .38])
    ax.imshow(normalized, cmap="Blues")
    ax.set_xticks(range(9), names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(9), names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("F01 row-normalized confusion (%) | sum across seeds", fontsize=10)
    for i in range(9):
        for j in range(9):
            ax.text(j, i, f"{normalized[i,j]:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if normalized[i,j] > 50 else "black")
    block(fig, "Confusion tổng ba seed có support10521 (mỗi ảnh một lần/seed), không phải10521 ảnh độc lập. Count và baseline matrix có ở curves/final_analysis và evaluator CSV; support trong bảng là3507/seed.", .075)
    finish(fig)

    fig = page(7, "Ảnh lỗi: cặp khó và độ tin cậy")
    errors = pd.read_csv(EVIDENCE / "tables/F01_seed0_error_gallery.csv").head(6)
    original_images = all((images_dir / name).exists() for name in errors.Filename)
    _, m = prediction("F01_seed0_test.csv", "test")
    block(fig, f"F01seed0 có {3507-int(np.trace(m['confusion']))} ảnh sai. Chinee apple→Snake weed {m['confusion'][0,7]} ảnh, chiều ngược lại {m['confusion'][7,0]} ảnh. Ảnh dưới lấy từ gallery: ưu tiên hai chiều cặp0↔7 rồi lỗi hai lớp khó. Không forward lại test.\n\nQuan sát: nền lá/cỏ rối, vùng sáng/tối, bộ phận cây khó phân biệt. Crop có thể mất chi tiết; các loài có cấu trúc lá gần nhau. Đây là giả thuyết, chưa kiểm chứng bằng segmentation/crop hay annotation vị trí. Một số lỗi confidence cao; ECE toàn tập tốt không đảm bảo mọi ảnh tin cậy.", .88)
    if not original_images:
        picture(fig, SUB / "curves/final_analysis/F01_seed0_error_gallery.png", [.07, .055, .86, .64])
    for k, row in enumerate(errors.itertuples() if original_images else []):
        left = .08 + (k % 3) * .3
        bottom = .37 if k < 3 else .065
        ax = fig.add_axes([left, bottom, .26, .255])
        path = images_dir / row.Filename
        with Image.open(path) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(f"{row.Filename}\n{row.true_class} → {row.predicted_class}\nConfidence {row.confidence:.3f}", fontsize=7)
        ax.axis("off")
    finish(fig)

    fig = page(8, "Kết luận, giới hạn và tái lập")
    block(fig, "Cấu hình tốt nhất: ConvNeXt-Tiny + CutMix + I07, được chọn trên val và xác nhận bằng ba seed test. Chênh lớn nhất trong sàng lọc đến từ backbone/pretraining; cùng ConvNeXt, CutMix là thay đổi đơn tăng F1 nhiều nhất. I07 cải thiện hiệu chuẩn, không tăng F1. T08 là kết quả âm cần giữ; TTA chưa bù được chi phí.\n\nRealtime: I07 FP32, p95 đại diện3,7609ms trên RTX3060 trong scope GPU đã khai báo. Chưa đo end-to-end camera/CPU/H2D hoặc robot khác. Offline cũng chọn I07 vì TTA chưa tăng F1. I08 cần kiểm chứng test theo protocol tương lai trước khi thay final.\n\nGiới hạn: một fold; B/T/I một seed; final ba seed còn ít và seed0 đã tham gia sàng lọc nên chưa hoàn toàn độc lập. Tag/dữ liệu pretraining khác ngăn kết luận nhân quả thuần về kiến trúc. Chia ngẫu nhiên theo ảnh không hold-out địa điểm; có thể lạc quan trên vùng/mùa/camera mới. Chưa audit near-duplicate hoặc domain shift. ECE phụ thuộc binning/mất cân bằng.\n\nHướng tiếp theo: lặp ablation quan trọng nhiều seed, protocol hold-out địa điểm, thu thập lỗi khó, thử crop/segmentation và đo toàn chuỗi cảm biến→decision; kiểm tra FP16/hiệu chuẩn trên GPU triển khai. Không dùng phân tích test để điều chỉnh model của kết quả hiện tại.\n\nTái lập: code/notebooks/reproduce_colab.ipynb; README có link Colab trỏ nhánh main (cần push commit trước khi mở). Mặc định CPU audit; bật REPRODUCE để chạy checkout/workspace mới theo B→T→I→lock final→seal→test once→export. Runtime khác có thể cho số khác, không cam kết giống từng bit.\n\nSản phẩm: results.xlsx7 sheet, report.md/PDF, curves, code, README, predictions. Evidence lưu config/history/summary/protocol/marker/evaluator và SHA256 nguyên byte. Checkpoint/dataset/cache không commit. Export/check chỉ đọc CSV/evidence, không train/fitT/forwardtest. Notebook đã bỏ output nhúng; bảng/hình/log vẫn giữ.\n\nNguồn: Olsen et al., DeepWeeds, Scientific Reports9,2058(2019), doi:10.1038/s41598-018-38343-3; pmc.ncbi.nlm.nih.gov/articles/PMC6375952/. Dữ liệu/ảnh từ tác giả (Zenodo/AlexOlsen/DeepWeeds); GUIDE/RUBRIC và eval.py gốc trong repo. Metric của bài báo khác thiết lập/fold nên chỉ tham khảo.\n\nEvaluator tự chấm mục I19/20; không phải điểm toàn bài hoặc điểm giảng viên xác nhận. Report.md có bảng/tag/cấu hình và phân tích đầy đủ hơn; mọi số trong PDF/Markdown xuất từ cùng bảng/evidence như Excel.", .88)
    finish(fig)
    if pdf.get_pagecount() != 8:
        raise ValueError("Expected an eight-page report")
    pdf.close()
