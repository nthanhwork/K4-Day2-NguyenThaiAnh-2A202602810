# Phần 6.2 — trạng thái pilot và ngân sách

- SESSION_ID: P62_session01; pilot hoàn thành 5/5.
- Phần 6.2 hoàn tất: True; sẵn sàng chuyển sang backbone: False.
- Horizon scheduler: 12 epoch; thực chạy: 1 epoch/model.
- Batch chung: 64; seed: 0; skipped updates: 14.
- Không đánh giá test; F1 pilot chỉ là kiểm tra, không chọn backbone.
- Epoch timing loại trừ download/profiling/checkpoint/plot và export val lần cuối.
- wall_seconds bao gồm toàn bộ lời gọi run; khi resume là thời gian của lời gọi tiếp tục.
- Ngân sách dự kiến: 311.4 phút; xem budget.json cho các giả định.

## Tiếp theo
Chạy B01–B05 theo cùng T00 đủ epoch, chọn backbone bằng macro-F1 val.
Không ghi các pilot này vào bảng kết quả 12 epoch.
