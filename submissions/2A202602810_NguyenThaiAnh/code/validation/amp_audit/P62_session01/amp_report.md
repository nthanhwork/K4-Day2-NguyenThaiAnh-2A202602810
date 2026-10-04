# Kiểm tra AMP — P62_session01

Kết quả: PASS cho cả 5 model. Đã kiểm tra một epoch train tiếp theo từ trạng thái pilot trong bộ nhớ; không lưu đè checkpoint và không đánh giá val/test.

| Model | Skip epoch pilot | Scale đã thích nghi | Update / batch ở epoch kiểm tra | Skip mới |
|---|---:|---:|---:|---:|
| resnet50 | 2 | 16384 | 164/164 | 0 |
| resnext50_32x4d | 3 | 8192 | 164/164 | 0 |
| convnext_tiny | 6 | 1024 | 164/164 | 0 |
| deit_small_patch16_224 | 2 | 16384 | 164/164 | 0 |
| efficientnet_b0 | 1 | 32768 | 164/164 | 0 |

## Diễn giải
- Tổng 14/820 batch ở pilot bị bỏ qua bước optimizer; GradScaler giảm scale từ mặc định 65536 theo hệ số 0,5. Đây là cơ chế bảo vệ gradient overflow.
- Một số lần skip không nằm ở vài batch đầu; checkpoint growth_tracker cho thấy scale vẫn đang điều chỉnh trong epoch đầu. Không kết luận lỗi chỉ từ tổng skip.
- Epoch kiểm tra tiếp theo: toàn bộ 820 loss và norm gradient hữu hạn, 820 optimizer updates, không skip, scale giữ nguyên; trạng thái model hữu hạn.
- Scheduler tiến đúng số optimizer updates; hash cả 5 checkpoint gốc không đổi.
- Điều kiện skipped_updates == 0 trong notebook quá chặt. Đã sửa: khi pilot có skip, cần AMP audit hợp lệ cho đúng các pilot trước khi đánh dấu sẵn sàng.
- Sẵn sàng chạy B01–B05 với cấu hình T00 hiện tại. Theo dõi scale/skip và loss trong các epoch tiếp theo; không cần thay LR hoặc tắt AMP chỉ vì 14 skip này.
- Các số của epoch kiểm tra là chẩn đoán train, không đưa vào bảng kết quả backbone; checkpoint pilot vẫn ở epoch 1.

Tài liệu: [PyTorch AMP](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).
Chi tiết: summary.json, summary.csv và các *_steps.csv trong thư mục này.
