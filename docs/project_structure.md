# Cấu trúc và phạm vi dự án

Dự án có hai pipeline thực nghiệm độc lập trên OPIXray. Chúng dùng loader, mô hình và metric khác nhau nên cần lưu kết quả riêng.

## Mã nguồn

- `ppd_resnet.py`: chuẩn bị crop, học bảng màu K-means, train backbone phân loại và đánh giá accuracy sạch qua các defense. Backbone được hỗ trợ: ResNet-50, EfficientNet-B0 và ViT-S.
- `attack_eval.py`: đánh giá robustness của checkpoint thuộc pipeline PPD bằng noise, FGSM, PGD và C&W; `--defense` chọn defense tại inference mà không đổi trọng số; kết quả gồm clean/robust accuracy, ASR và độ méo.
- `summarize_results.py`: đọc CSV của PPD, tạo bảng tổng hợp và biểu đồ.
- `visualize_def.py`: lưu ảnh ghép để kiểm tra ảnh trước/sau defense.
- `advlab/`: thư viện của pipeline phân loại khác, gồm model, loader, attack, defense và metric.
- `advlab_examples/`: lệnh train, đánh giá attack và đọc metadata checkpoint cho pipeline YOLO classification.

## Dữ liệu và palette

- `data/OPIXray/`: dữ liệu gốc cùng các file split; không chỉnh sửa dữ liệu này trong các bước tạo kết quả.
- `data/processed/cache/`: crop đã resize và nhãn, dùng lại để tránh đọc/cắt ảnh ở mỗi lượt chạy.
- `data/processed/palettes/`: palette 16/32/64/128 màu được học từ pixel train. Chỉ dùng split train để fit palette.

Cache và palette là artifact có thể tái tạo từ dữ liệu gốc, không phải mã nguồn.

## Cấu hình thí nghiệm

`experiments/ppd_classification/configs/` chứa một JSON cho mỗi backbone và `experiment_matrix.csv` liệt kê tích mô hình × defense × attack. Đây là manifest cho người chạy và báo cáo, không phải cấu hình được tự động nạp bởi script. Cập nhật trạng thái/seed/tham số sau mỗi lượt để phân biệt planned, completed và failed.

## Checkpoint

- `artifacts/checkpoints/ppd/`: backbone đã fine-tune cho nhánh PPD; tên file gồm kiến trúc và defense dùng khi train.
- `artifacts/checkpoints/yolov5_classification/`: checkpoint và JSON metadata của pipeline advlab.
- `assets/pretrained/`: trọng số pretrained đầu vào của YOLO classification, khác với checkpoint fine-tune trên OPIXray.

Metadata JSON ghi thông tin để người đọc biết checkpoint là gì; file `.pt` chứa trọng số để nạp vào mô hình.

## Kết quả

- `experiments/ppd_classification/metrics/`: CSV attack/eval, lịch sử train theo epoch, summary.
- `experiments/ppd_classification/figures/`: biểu đồ tổng hợp và ảnh ghép sau defense.
- `experiments/yolov5_classification/metrics/`: CSV tổng hợp, per-class, runtime của pipeline advlab.
- `experiments/yolov5_classification/figures/examples/`: ảnh minh họa được lưu bởi pipeline advlab.
- `archives/colab/`: ZIP dùng để chuyển input/output giữa máy local và Colab. Đây là bản lưu, không phải thư mục chương trình đọc trực tiếp.

Không gộp hai nhóm CSV cho tới khi cùng xác nhận model, split, epsilon, số bước và cách tính metric. `yolov5n-cls` là classifier; nếu sau này bổ sung detector YOLO, hãy đặt trong một nhánh riêng và dùng metric detection phù hợp.

## Physical robustness

`experiments/physical_robustness/` là bộ khung đánh giá phòng vệ dưới biến thiên thu nhận ảnh với mẫu thử vô hại và quy trình có kiểm soát. `protocol.md` ghi thiết bị, mẫu, split và metric; `manifests/`, `metrics/`, `figures/` lần lượt chứa metadata mẫu, số liệu và hình. Không có thư mục hoặc mã tối ưu vật thể để làm hệ thống soi chiếu bỏ sót vật bị cấm. YOLO checkpoint hiện có ghi `yolov5n-cls`, là classifier; nếu có detector object detection, đặt thành pipeline và metric riêng.

## Đường dẫn mặc định

Các script chính xác định thư mục dự án từ vị trí file mã nguồn. Có thể chạy từ root dự án hoặc truyền đường dẫn tuyệt đối. Nếu đổi lại cấu trúc, cập nhật các hằng số đường dẫn ở `ppd_resnet.py`, mặc định CLI ở `advlab_examples/`, và các đường dẫn ví dụ trong README cùng lúc.
