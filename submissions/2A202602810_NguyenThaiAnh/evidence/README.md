# Bằng chứng thực nghiệm

Các file trong `runs/` và `labels/` là bản sao nguyên byte của log, cấu hình, protocol, evaluator và CSV nguồn. `source_index.json` ghi SHA256 và đường dẫn nguồn. Đường dẫn tuyệt đối trong protocol phản ánh máy thực nghiệm; giữ nguyên để bảo toàn checksum.

Các bản sao phục vụ kiểm tra và tạo báo cáo, không đặt trở lại `runs/` để resume trên máy khác. Checkpoint và raw NumPy cache được giữ ở máy gốc, không commit. Dự đoán CSV đầy đủ nằm tại `../predictions/`. Máy mới có thể kiểm tra metric và tạo lại Excel/báo cáo bằng CPU; tái huấn luyện dùng một checkout sạch.
