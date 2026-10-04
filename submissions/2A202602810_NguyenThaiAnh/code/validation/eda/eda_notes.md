# Ghi chú EDA DeepWeeds — fold 0

- Train/val/test: 10,501/3,501/3,507; union 17,509.
- Ba giao: {'train_val': 0, 'train_test': 0, 'val_test': 0}; không thiếu ảnh trong CSV.
- Lớp đa số train: Negative, chiếm 52.02%; imbalance 9.03x.
- Baseline lý thuyết luôn đoán lớp đa số trên nhãn TRAIN: top-1 0.5202, macro-F1 0.0760.
- Đã giải mã 10,501 ảnh train; lỗi 0.
- Thống kê RGB lấy từ 512 ảnh train với seed 42; giữ normalize của pretrained config.
- Duplicate byte groups: 0; cross-split: 0; checked=True.
- Sai khác nhãn giữa nguồn split/labels.csv: 1; giữ split nguyên bản.
- Ảnh mẫu mỗi lớp và ảnh augmentation/Mixup/CutMix nằm trong curves/eda/.

## Câu hỏi nhận xét trực quan cần bổ sung

- Cặp Chinee apple/Snake weed: chi tiết nào giúp phân biệt? Có crop làm mất chi tiết đó không?
- Negative gồm những kiểu nền/đối tượng nào trong các mẫu đã xem?
- Color/RandAugment/Mixup/CutMix giữ được dấu hiệu loài trong mẫu nào, phá hủy dấu hiệu trong mẫu nào?

Các nhận xét ảnh là giả thuyết; cần ablation trên val để kiểm chứng. Không dùng điểm test cho lựa chọn.
