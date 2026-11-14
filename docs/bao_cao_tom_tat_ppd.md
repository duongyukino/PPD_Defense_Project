---
title: "Tổng quan đề tài và tiến độ triển khai"
subtitle: "Phòng thủ bằng chiếu bảng màu cho ảnh soi chiếu X-quang"
date: "12/11/2026"
lang: vi
---

> **Tóm tắt:** Dự án đã xây dựng pipeline PPD, các phương pháp đối chứng và ba dạng tấn công; đã có kết quả validation ban đầu trên ResNet-50 và EfficientNet-B0. Trong những kết quả PGD thích nghi được khảo sát, JPEG75 nổi bật khi giữ nguyên checkpoint ResNet huấn luyện trên ảnh sạch; PPD16m kết hợp checkpoint ResNet huấn luyện riêng là ứng viên đáng chú ý trong nhóm thứ hai. Đây chưa phải kết luận tổng quát hay kết quả test cuối.

## 1. Mục tiêu đề tài

### 1.1. Bài toán

Đề tài nghiên cứu ảnh đối kháng đối với mô hình phân loại ảnh soi chiếu hành lý X-quang và thử một phương pháp tiền xử lý nhằm tăng khả năng chống chịu. Mục tiêu là so sánh chất lượng phân loại trên ảnh sạch và ảnh bị tấn công, chứ không chỉ tạo ra ảnh đối kháng.

Ảnh soi chiếu có thể được biểu diễn bằng giả màu liên quan đến vật liệu. Ý tưởng nghiên cứu là phân bố màu hợp lệ của ảnh có thể tập trung trong một vùng hẹp hơn toàn bộ không gian RGB. Nếu nhiễu đối kháng đẩy một số màu ra khỏi vùng đó, có thể thử đưa màu trở về gần các màu đại diện đã học từ dữ liệu sạch.

**Đây là giả thuyết cần kiểm chứng, không phải bảo đảm:** nhiễu đối kháng không nhất thiết làm pixel rời khỏi tập màu; ảnh sạch cũng có thể có nhiều màu do nén và nội suy. Cần đo cả tác động của xử lý lên ảnh sạch và độ bền trước tấn công biết rõ phòng thủ.

### 1.2. Phương pháp dự kiến: Palette Projection Defense (PPD)

1. **Học bảng màu:** lấy mẫu pixel từ tập huấn luyện, dùng K-means để tìm khoảng 16–64 màu đại diện.
2. **Chiếu màu khi suy luận:** thay mỗi pixel bằng màu gần nhất trong bảng màu theo khoảng cách RGB.
3. **Lọc tùy chọn:** thêm bộ lọc trung vị kích thước 3×3 để làm mượt.

Sơ đồ khái quát:

```text
Ảnh sạch train -> lấy mẫu pixel -> K-means -> palette cố định
Ảnh đầu vào -> PPD [-> median 3×3] -> chuẩn hóa -> mô hình -> dự đoán
```

Median trong dự án thực hiện theo từng kênh RGB sau bước chiếu; vì vậy màu sau median không nhất thiết vẫn thuộc palette. PPD cũng có thể làm thay đổi chi tiết quan trọng và giảm accuracy ảnh sạch.

### 1.3. Thực nghiệm mong muốn

| Thành phần | Mục tiêu |
|---|---|
| Dữ liệu | Dùng SIXray, OPIXray hoặc PIDray; ưu tiên bài toán phân loại dễ triển khai. |
| Mô hình | So sánh 2–3 mô hình đã fine-tune, ví dụ ResNet-50, EfficientNet và ViT-S. |
| Tấn công | FGSM, PGD và C&W; khảo sát nhiều mức độ tấn công phù hợp với từng phương pháp. |
| Đối chứng | Không phòng thủ, JPEG, Gaussian blur, giảm bit-depth và PPD. |
| Kết quả | Accuracy trên ảnh sạch và ảnh đối kháng; so sánh cả tấn công không thích nghi và thích nghi. |
| Ablation | So sánh số màu K và tác động của median 3×3. |

Với FGSM/PGD, dự kiến dùng các mức nhiễu L∞ 2, 4 và 8/255. C&W đang triển khai theo chuẩn L2: `kappa` điều chỉnh biên logits, **không tương đương** mức epsilon L∞; cần báo cáo C&W riêng cùng độ lớn nhiễu và cấu hình tối ưu.

## 2. Phạm vi dự án hiện tại

### 2.1. Dữ liệu và bài toán thực tế

Dự án chọn **OPIXray**. Pipeline PPD sử dụng bounding box trong annotation để cắt vùng chứa vật thể, resize mỗi crop về 224×224 và phân loại vào một trong năm lớp: Folding Knife, Multi-tool Knife, Scissor, Straight Knife và Utility Knife.

Do đã biết bounding box trước khi phân loại, đây là **phân loại crop có sẵn vị trí**, không phải hệ thống tự phát hiện vật thể trên toàn ảnh. Do đó accuracy ở đây không phải mAP hay khả năng phát hiện vật thể trong một hệ thống sân bay hoàn chỉnh.

Mã có bước chia train/validation từ danh sách train theo tên ảnh, seed 42, tỷ lệ validation khoảng 10%; test lấy từ phần test của OPIXray. Bảng màu được học chỉ từ pixel của crop train. Khi báo cáo cuối cùng vẫn cần kiểm tra và ghi lại split thực tế, phân bố lớp, kích thước và nguồn gốc dữ liệu.

### 2.2. Hai nhánh mã riêng

- **Nhánh chính cho đề tài PPD:** `ppd_resnet.py` và `attack_eval.py`; dữ liệu, checkpoint và kết quả ở các thư mục tương ứng của `ppd_classification`.
- **Nhánh `advlab` / YOLO classification:** là pipeline phân loại khác, có loader và cách đánh giá riêng; `yolov5n-cls` là classifier, không phải detector. Không gộp kết quả hai nhánh khi chưa đồng nhất dữ liệu, mô hình và giao thức.
- Thư mục `experiments/physical_robustness/` là bộ khung riêng cho nghiên cứu độ bền trong điều kiện thu nhận có kiểm soát; không phải kết quả PPD chính của báo cáo này.

## 3. Những gì đã triển khai

### 3.1. Chuẩn bị dữ liệu và palette

- Đã có mã đọc annotation, cắt crop, tạo cache train/validation/test và lấy mẫu pixel từ train.
- Đã có mã học palette bằng `MiniBatchKMeans`: với K màu, học K−1 tâm từ pixel không phải nền và thêm một tâm đại diện nền sáng.
- Có file palette cho K = 16, 32, 64 và 128 trong `data/processed/palettes/`. K = 128 là mức bổ sung ngoài khoảng tối thiểu 16–64.
- `data/thu_mau.py` có các phép phân tích phân bố màu, sai số lượng tử hóa và khoảng cách tới các tâm palette; có thể dùng làm điểm xuất phát để kiểm chứng giả thuyết màu, nhưng sự tồn tại của mã chưa chứng minh giả thuyết đúng.

### 3.2. Phòng thủ và baseline

| Phương pháp | Trạng thái trong mã | Ý nghĩa |
|---|---|---|
| `none` | Có | Không thay đổi ảnh. |
| `jpeg75`, `jpeg50` | Có | Nén và giải nén JPEG theo mức chất lượng. |
| `blur0.5`, `blur1.0` | Có | Lọc Gaussian với sigma tương ứng. |
| `bit3`, `bit4` | Có | Giảm số mức màu mỗi kênh. |
| `ppd16`, `ppd32`, `ppd64`, `ppd128` | Có | Chiếu từng pixel về màu palette gần nhất. |
| Hậu tố `m`, ví dụ `ppd32m` | Có | Thêm median 3×3 sau phép chiếu. |

Mã cho phép thay defense khi đánh giá cùng một checkpoint. Điều này hữu ích để so sánh công bằng ảnh hưởng của bước tiền xử lý. Những checkpoint được huấn luyện riêng với defense là một thiết lập khác, cần báo cáo tách biệt.

### 3.3. Giải thích lựa chọn thông số

**Lưu ý:** Mã cho biết các giá trị đã sử dụng, nhưng chưa có bằng chứng chúng đã được tối ưu. Những lý do sau là diễn giải kỹ thuật hợp lý, không phải kết luận thực nghiệm về lựa chọn tốt nhất.

#### Kích thước ảnh 224×224

Crop được resize trực tiếp về 224×224 bằng nội suy bilinear. Kích thước thống nhất giúp tạo batch, so sánh mô hình và phù hợp với kích thước đầu vào phổ biến của các backbone pretrained ImageNet; biến thể ViT được dùng cũng có tên `vit_small_patch16_224`. Đây là điểm cân bằng thực dụng giữa chi phí và chi tiết, không chứng minh là tối ưu cho OPIXray.

Resize hình chữ nhật thành hình vuông có thể làm méo tỷ lệ; ảnh nhỏ có thể mất chi tiết và nội suy tạo màu trung gian. Palette được học từ crop gốc trước resize, trong khi PPD chạy sau resize, nên phân bố màu ở hai bước có thể khác nhau. Nên so sánh với giữ tỷ lệ rồi padding và thử kích thước khác phù hợp từng kiến trúc.

#### JPEG chất lượng 75: `jpeg75`

Ảnh được nén và giải nén JPEG với `quality=75`. **75 là tham số chất lượng encoder, không có nghĩa giữ 75% pixel hay thông tin.** JPEG biến đổi các hệ số ảnh, có thể giảm một phần nhiễu nhỏ; mức 75 là một baseline trung gian hợp lý giữa bảo toàn chi tiết và nén mạnh. Nén quá mạnh cũng có thể tạo artifact, làm mất biên hoặc chi tiết quan trọng.

Chưa có bằng chứng 75 là tối ưu. Nên so sánh các mức 50, 75 và 90 trên cùng mẫu và attack; với attack thích nghi phải kiểm tra chất lượng gradient xấp xỉ vì JPEG không khả vi.

#### Gaussian blur: `blur1.0`

Giá trị 1,0 là độ lệch chuẩn Gaussian σ tính theo pixel, **không phải kernel 1×1**. Với bán kính `ceil(3σ)`, code dùng kernel 7×7 và đệm phản chiếu. Lọc làm suy giảm biến thiên nhanh trong không gian nhưng cũng làm mờ biên, mất chi tiết; attack biết bộ lọc vẫn có thể tối ưu nhiễu vượt qua nó.

`blur1.0` là mức baseline để khảo sát, chưa chắc tối ưu. Có thể so sánh `blur0.5`, `blur1.0` và mức mạnh hơn trên cùng giao thức, đồng thời đo accuracy sạch.

#### Giảm bit: `bit3`

`bit3` lượng tử hóa mỗi kênh RGB về `2³ = 8` mức: 0, 1/7, …, 1. Về tổ hợp, có tối đa 512 màu RGB. Nhiễu nhỏ có thể bị loại nếu pixel vẫn nằm trong cùng khoảng làm tròn, nhưng pixel gần ranh giới có thể đổi mức; giảm bit mạnh cũng làm biến dạng ảnh sạch.

Khác với PPD học palette từ train, `bit3` dùng các mức cố định riêng cho từng kênh. Nên so sánh với `bit4` và các mức khác thay vì mặc định 3 bit tốt nhất.

#### Số màu K và median

K nhỏ nén màu mạnh, có thể giảm một số biến đổi nhưng dễ mất thông tin; K lớn thường giữ màu chi tiết hơn nhưng có thể giữ cả biến đổi không mong muốn. Không có bảo đảm quan hệ này luôn đúng. `K=32` là điểm giữa thuận tiện; K = 16, 32, 64 là các mức ablation đề xuất, và dự án còn có K = 128.

Median 3×3 xử lý từng kênh sau PPD, có thể giảm nhiễu nhưng cũng xóa chi tiết, và có thể tạo màu không còn thuộc palette. Trong cách học hiện tại, mã lấy tối đa 400 pixel mỗi crop train gốc, tìm `K−1` tâm từ pixel không phải nền rồi thêm tâm nền sáng; ngưỡng nền là `min(R,G,B) > 235`. Cần kiểm chứng cách lấy mẫu và ngưỡng, không mặc định mọi pixel sáng đều là nền.

### 3.4. Mô hình

`ppd_resnet.py` hỗ trợ ResNet-50, EfficientNet-B0 và ViT-S; đầu ra được chỉnh thành 5 lớp. Mặc định có thể dùng trọng số pretrained rồi fine-tune. Ảnh đi qua defense, chuẩn hóa và backbone theo thứ tự đó.

Đã xác nhận có checkpoint `resnet50_none.pt` và `efficientnet_b0_none.pt`. Lịch sử train EfficientNet ghi validation accuracy khoảng **98,18% ở epoch 8**. Mã hỗ trợ ViT-S thông qua `timm`, nhưng chưa thấy checkpoint `vit_small_none.pt`; không nên nói ViT-S đã hoàn thành huấn luyện hoặc benchmark chỉ vì có nhánh mã hỗ trợ.

### 3.5. Tấn công và cách đo

`attack_eval.py` đã có FGSM, PGD, C&W L2 không nhắm đích và nhiễu dấu ngẫu nhiên để đối chứng. Mã hỗ trợ hai kịch bản:

- **Không thích nghi / oblivious:** tạo ảnh tấn công trên backbone không bật defense, sau đó bật defense để đánh giá.
- **Thích nghi / adaptive:** đưa defense vào đường tấn công; với các phép không khả vi, dùng gradient xấp xỉ kiểu BPDA/STE. Với median, có các lựa chọn gradient thật, xấp xỉ bằng bộ lọc trung bình hoặc identity.

Mã lưu `clean_acc`, `robust_acc`, `asr`, `bb_acc` và một số đại lượng độ lớn nhiễu. `robust_acc` trong CSV là accuracy trên **toàn bộ ảnh sau tấn công**, kể cả mẫu đã sai trên ảnh sạch; đây không nhất thiết là robust accuracy theo nghĩa mẫu phải đúng ở cả hai điều kiện. `asr` là tỷ lệ mẫu ban đầu đúng nhưng sau tấn công sai, trên số mẫu ban đầu đúng. `post_l2` là chênh lệch giữa hai ảnh sau defense, không tự nó chứng minh đã xóa được bao nhiêu nhiễu.

Đã có lệnh `selftest` trong `ppd_resnet.py` để kiểm tra một số điều kiện hình dạng, miền giá trị và gradient; `tests/test_attacks.py` thuộc nhánh `advlab`. Tồn tại kiểm thử không đồng nghĩa toàn bộ pipeline PPD hoặc độ mạnh của attack đã được xác nhận đầy đủ.

FGSM dùng một bước gradient; PGD lặp trong miền nhiễu L∞ với các mức 2, 4 và 8/255 được khảo sát. Đây là các mức thử nghiệm, không phải giới hạn an toàn được chứng minh cho ảnh X-quang. Config đề xuất 30 bước PGD, còn mặc định trong code là 10 bước, nên phải ghi lệnh thực tế. C&W L2 dùng `kappa` điều chỉnh margin logits, `c` cố định và số bước hữu hạn; không giới hạn cùng epsilon L∞ như PGD. Nên báo cáo riêng các chuẩn nhiễu đạt được và tham số tối ưu.

## 4. Kết quả đã lưu: chỉ mang tính ban đầu

Các con số dưới đây được trích từ CSV hiện có, không phải chạy lại trong lần tổng hợp này. Chúng thuộc **validation**, thường dùng 200 crop cho FGSM/PGD và 100 crop cho C&W. CSV chưa lưu trực tiếp seed, danh sách mẫu, lệnh chạy đầy đủ hay hash checkpoint; cấu hình đề xuất không chứng minh mọi lượt đã thực hiện đúng cấu hình đó.

### 4.1. Ví dụ PGD trên PPD32m

Cả hai mô hình dùng checkpoint huấn luyện với `none`, sau đó áp dụng `ppd32m` khi đánh giá. Mỗi dòng là một file kết quả 200 crop validation. Các giá trị PGD lần lượt ứng với epsilon 2, 4, 8/255.

| Mô hình và kịch bản | Clean | PGD 2/255 | PGD 4/255 | PGD 8/255 |
|---|---:|---:|---:|---:|
| ResNet-50, không thích nghi | 74,5% | 75,0% | 71,5% | 69,0% |
| ResNet-50, thích nghi | 74,5% | 27,0% | 8,0% | 0,0% |
| EfficientNet-B0, không thích nghi | 49,0% | 47,0% | 49,5% | 51,5% |
| EfficientNet-B0, thích nghi | 49,0% | 8,5% | 1,0% | 0,0% |

**Cách đọc:** số cao dưới tấn công không biết defense không đủ để chứng minh phòng thủ an toàn. Các lượt thích nghi trong CSV này làm accuracy giảm mạnh. Do một số mẫu sai trên ảnh sạch có thể trở thành đúng sau tấn công, accuracy sau attack đôi khi cao hơn clean hoặc không giảm đơn điệu theo epsilon; đó không tự nó chứng minh mô hình được cải thiện.

Với EfficientNet-B0 không phòng thủ, một CSV khác ghi clean **99,0%** và PGD **0,0%** ở cả ba mức trên 200 crop validation. Kết quả này phản ánh mức dễ bị tấn công trong lượt chạy đó, không tự nó chứng minh kiến trúc EfficientNet bị lỗi. So sánh giữa các file vẫn cần xác minh cùng danh sách mẫu và cấu hình chạy.

### 4.2. FGSM và C&W

Các CSV gốc đã có kết quả FGSM; với `ppd32m` và checkpoint clean, đã có ví dụ C&W **không thích nghi** cho cả ResNet-50 và EfficientNet-B0. Với 100 crop validation:

| Mô hình | Clean trên 100 crop | C&W kappa 0 | C&W kappa 10 |
|---|---:|---:|---:|
| ResNet-50 + PPD32m | 72,0% | 70,0% | 69,0% |
| EfficientNet-B0 + PPD32m | 46,0% | 46,0% | 45,0% |

Không so sánh trực tiếp các tỷ lệ ở bảng này với bảng PGD 200 crop như thể chúng được đo trên cùng tập mẫu. Chưa thấy C&W thích nghi trong cặp CSV PPD32m vừa trích; còn phải kiểm tra và hoàn thiện ma trận thí nghiệm. Kết quả C&W với số bước hữu hạn, `c` cố định và không binary search không chứng minh không tồn tại nhiễu L2 hiệu quả hơn.

### 4.3. Ablation và baseline

Các bảng sau tổng hợp **PGD thích nghi**, validation, `n=200` cho mỗi dòng; epsilon là L∞. Tên `adapt30` cho biết lượt ResNet tương ứng dùng 30 bước; với median, `mean` chỉ gradient xấp xỉ qua median. CSV không lưu đầy đủ lệnh, seed, ID mẫu và hash checkpoint: chưa thể chứng minh mọi lượt là phép so sánh ghép cặp hoàn toàn.

#### ResNet-50: cùng checkpoint clean, thay defense lúc suy luận

| Defense | Clean | PGD 2/255 | PGD 4/255 | PGD 8/255 |
|---|---:|---:|---:|---:|
| `none` | 99,0% | 0,0% | 0,0% | 0,0% |
| `jpeg75` | 98,5% | **42,0%** | **10,5%** | **0,5%** |
| `blur1.0` | 97,0% | 0,0% | 0,0% | 0,0% |
| `bit3` | 91,5% | 0,5% | 0,0% | 0,0% |
| `ppd16` | 75,0% | 3,0% | 0,5% | 0,0% |
| `ppd16m` | 72,5% | 26,5% | **10,5%** | 0,0% |
| `ppd32` | 79,0% | 2,5% | 0,0% | 0,0% |
| `ppd32m` | 74,5% | 27,0% | 8,0% | 0,0% |
| `ppd64` | 78,0% | 1,0% | 0,0% | 0,0% |
| `ppd64m` | 78,0% | 19,5% | 3,0% | 0,0% |

Hàng `none` là PGD thông thường vì không có defense để thích nghi; các hàng còn lại là PGD thích nghi. Các file mang nhãn checkpoint `resnet50_none`. Trong bảng này, **JPEG75 cân bằng tốt nhất**: giữ clean 98,5% và đạt kết quả PGD cao nhất hoặc đồng cao nhất ở ba mức được trích. Tuy nhiên tại 8/255 chỉ còn 0,5%, và attack có thể chưa đủ mạnh. Median cho kết quả cao hơn PPD không median trong các lượt này, nhưng điều đó chưa chứng minh độ bền thực sự vì gradient qua median chỉ là xấp xỉ.

#### EfficientNet-B0: cùng checkpoint clean, thay defense lúc suy luận

| Defense | Clean | PGD 2/255 | PGD 4/255 | PGD 8/255 |
|---|---:|---:|---:|---:|
| `none` | 99,0% | 0,0% | 0,0% | 0,0% |
| `jpeg75` | 98,5% | 4,0% | 0,0% | 0,0% |
| `blur1.0` | 94,0% | 0,5% | 0,0% | 0,0% |
| `bit3` | 86,5% | 0,0% | 0,0% | 0,5% |
| `ppd16` | 26,0% | 3,5% | 2,5% | 1,0% |
| `ppd16m` | 19,0% | 6,0% | 4,5% | 4,0% |
| `ppd32` | 64,0% | 0,5% | 0,0% | 0,0% |
| `ppd32m` | 49,0% | **8,5%** | 1,0% | 0,0% |

Trong các hàng được kiểm tra, `ppd32m` đạt accuracy sau PGD 2/255 cao nhất, nhưng chỉ giữ 49,0% clean. `jpeg75` giữ clean cao hơn nhiều nhưng kết quả PGD thích nghi thấp. Vì thế không thể gọi một phương án là tốt nhất nếu chưa xác định tiêu chí cân bằng clean và độ bền. Accuracy có thể không giảm đơn điệu theo epsilon do cách tính trên toàn bộ ảnh và tính ngẫu nhiên của PGD; đó không phải bằng chứng defense mạnh hơn khi nhiễu tăng.

#### ResNet huấn luyện riêng với defense

Mỗi hàng dưới đây sử dụng **checkpoint khác nhau** được fine-tune với defense tương ứng. Đây là đánh giá cả mô hình lẫn defense, không cô lập riêng tác động của K; không ghép các số này thành cùng hàng với checkpoint `resnet50_none`.

| Checkpoint + defense | Clean | PGD 2/255 | PGD 4/255 | PGD 8/255 |
|---|---:|---:|---:|---:|
| `resnet50_ppd16m` + `ppd16m` | 98,0% | **82,5%** | **56,0%** | **15,0%** |
| `resnet50_ppd32m` + `ppd32m` | 98,0% | 67,5% | 33,0% | 6,5% |
| `resnet50_ppd64m` + `ppd64m` | 99,0% | 60,0% | 20,0% | 1,5% |
| `resnet50_jpeg75` + `jpeg75` | 98,5% | 41,5% | 4,0% | 0,0% |

**PPD16m là ứng viên đáng chú ý nhất trong nhóm và các lượt PGD đã khảo sát**, không phải bằng chứng PPD16m luôn thắng. Kết quả ở 8/255 còn thấp; cách xấp xỉ gradient ảnh hưởng đáng kể: cùng `resnet50_ppd16m` tại 8/255, file `mean` ghi 15,0%, còn file `identity` ghi 35,0%. Cần thử nhiều phương pháp tấn công và lấy kết quả xấu nhất theo từng mẫu khi có thể. Train với ảnh qua defense không đồng nghĩa adversarial training: vòng train này không tạo ảnh PGD.

Nguồn các bảng trong `experiments/ppd_classification/metrics/`: file `attack_resnet50_none_val_obl30.csv`; các file ResNet có tên bắt đầu bằng `attack_resnet50_none_as_` và kết thúc bằng `_val_adaptive_adapt30mean.csv`; các file EfficientNet bắt đầu bằng `attack_efficientnet_b0_none_as_` và kết thúc bằng `_val_adaptive_val_next.csv`. Nhóm checkpoint huấn luyện riêng dùng các file `attack_resnet50_` + tên defense + `_val_adaptive_adapt30mean.csv` hoặc `_val_adaptive_adapt30.csv`. Hàng EfficientNet `none` cũng có trong file bắt đầu bằng `attack_efficientnet_b0_none_as_none_` và kết thúc bằng `val_val_next.csv`. Tên và nguồn CSV cụ thể cần được lưu cùng cấu hình khi tái lập.

#### Kết luận tạm thời về phương pháp tốt nhất

Nếu chỉ đổi defense trên checkpoint ResNet sạch, ưu tiên kiểm tra sâu **JPEG75**; nếu cho phép fine-tune từng defense, ưu tiên **ResNet + PPD16m**. Hai kết luận có phạm vi khác nhau, không phải bảng xếp hạng chung. Trên EfficientNet, dữ liệu được trích cho thấy chưa có phương án nào vừa giữ sạch cao vừa chống PGD thích nghi tốt. Chưa đủ kết quả adaptive C&W, test cuối và kiểm chứng sức tấn công để khẳng định phương pháp thắng chung cuộc.

### 4.4. Bổ sung chỉ số cho bảng kết quả

#### Robust accuracy có điều kiện

Gọi `x` là ảnh sạch, `x_adv` là ảnh sau tấn công, `y` là nhãn đúng và `F` là toàn bộ pipeline mô hình kèm defense. **Robust accuracy có điều kiện** là tỷ lệ mẫu vẫn được dự đoán đúng sau tấn công, chỉ tính trên những mẫu mà `F(x)` đã dự đoán đúng:

```text
Robust accuracy có điều kiện = số mẫu đúng cả sạch và sau tấn công
                               / số mẫu đúng trên ảnh sạch
ASR = số mẫu đúng sạch nhưng sai sau tấn công / số mẫu đúng sạch
```

Với định nghĩa ASR hiện tại, hai tỷ lệ này cộng lại bằng 100% khi có ít nhất một mẫu sạch đúng. Nếu không có mẫu sạch đúng, cả hai tỷ lệ theo điều kiện phải được báo là **không xác định**, không quy ước là 0%. `attack_eval.py` hiện trả 0 cho ASR khi mẫu số bằng 0; không được diễn giải giá trị đó là ASR hợp lệ.

Cần báo cáo thêm cột này bên cạnh `robust_acc` hiện tại, vì `robust_acc` tính trên **toàn bộ mẫu sau attack**, kể cả mẫu sai khi sạch nhưng đúng sau tấn công. Ví dụ sau được suy ra từ ASR đã lưu, làm tròn đến hai chữ số thập phân, không phải thí nghiệm mới:

| Checkpoint clean + defense | PGD thích nghi | Clean | `robust_acc` toàn bộ | ASR | Robust accuracy có điều kiện |
|---|---|---:|---:|---:|---:|
| ResNet-50 + PPD32m | 2/255, n=200 | 74,5% | 27,0% | 63,76% | 36,24% |
| EfficientNet-B0 + PPD32m | 2/255, n=200 | 49,0% | 8,5% | 82,65% | 17,35% |

Nguồn trong `experiments/ppd_classification/metrics/`:

- ResNet: `attack_resnet50_none_as_ppd32m_val_adaptive_adapt30mean.csv`.
- EfficientNet: `attack_efficientnet_b0_none_as_ppd32m_val_adaptive_val_next.csv`.

Kết quả suy ra xấp xỉ vì ASR trong CSV đã làm tròn. Các bảng ở phần 4.1–4.3 vẫn giữ nguyên `robust_acc` toàn bộ, không phải giá trị có điều kiện.

#### PSNR và SSIM trên ảnh sạch qua phòng thủ

Đề xuất đo **PSNR và SSIM giữa `x` và `D(x)`** cho mỗi defense `D`: đây là mức biến đổi ảnh sạch do tiền xử lý, không phải độ bền trước tấn công. Đo trên cùng crop validation, sau resize 224×224 và trước chuẩn hóa ImageNet; cả hai ảnh cùng RGB, miền giá trị `[0,1]`, cùng kích thước. Giữ cố định thư viện, `data_range=1`, cách xử lý kênh và tham số SSIM, rồi ghi lại cấu hình để tái lập.

- **PSNR**, đơn vị dB: lớn hơn thường nghĩa là sai số bình phương nhỏ hơn. Tính theo từng ảnh rồi báo cáo thống kê đã định trước; trường hợp ảnh không đổi có MSE bằng 0 và PSNR vô hạn, cần ghi số ảnh này thay vì âm thầm loại bỏ hoặc gán giá trị tùy ý.
- **SSIM**: đo mức tương đồng cấu trúc theo cấu hình đã chọn; thông thường càng gần 1 càng giống. Hai chỉ số không tự chứng minh chi tiết quan trọng cho phân loại hoặc soi chiếu được bảo toàn.
- Báo cáo số mẫu, giá trị trung bình và độ phân tán cho từng defense, cùng ví dụ ảnh trước/sau nếu có thể. `none` cho ảnh không đổi nên PSNR là vô hạn và SSIM là 1 theo định nghĩa chuẩn.

**Trạng thái:** các bảng ở đây chưa có phép đo PSNR/SSIM trên cùng tập mẫu; cần đo trước khi điền số, không suy ra từ `post_l2` trong CSV.

#### C&W thích nghi và độ lớn nhiễu L2

Cần bổ sung C&W **thích nghi với pipeline có defense**, tách riêng khỏi C&W không thích nghi; ưu tiên hoàn thiện các tổ hợp PPD32m còn thiếu rồi mở rộng sang các defense khác. Không coi `kappa=0/10` là epsilon: đây là tham số margin, còn C&W L2 hiện không áp đặt cùng giới hạn L∞ như PGD.

Với mỗi lượt C&W, báo cáo `n`, clean accuracy, `robust_acc` toàn bộ, robust accuracy có điều kiện, ASR và **L2 trung bình của `x_adv - x`**; ghi rõ miền pixel và tập mẫu dùng để lấy trung bình. Nên báo cáo thêm L2 trên các mẫu tấn công thành công cùng số lượng thành công, vì trung bình trên toàn bộ ảnh cũng bao gồm các lần thất bại. Nếu không có mẫu thành công thì L2 trung bình trên nhóm thành công là không xác định. Ghi thêm `kappa`, `c`, số bước, learning rate, checkpoint, split và cấu hình gradient xấp xỉ.

**Trạng thái:** cặp CSV PPD32m ở mục 4.2 mới có C&W không thích nghi; một số baseline đã có kết quả C&W thích nghi, nhưng chưa đủ để kết luận cho toàn bộ ma trận. Không so sánh tỷ lệ thành công C&W với PGD như thể có cùng ngân sách nhiễu; nếu so các defense theo L2, phải nêu rõ quy tắc và điều kiện so sánh.

#### Nguyên tắc chọn cấu hình và đánh giá cuối

Học palette và huấn luyện trên train; dùng validation để chọn checkpoint, defense, K, median và giao thức đánh giá. Sau khi cố định toàn bộ lựa chọn, đánh giá các cấu hình đã chốt trên test **một lần theo giao thức**, không dùng kết quả test để tiếp tục chọn hoặc tinh chỉnh. Các giá trị ở mục 4 là validation, không phải kết quả test cuối.

## 5. Hạn chế và các điểm cần sửa trước khi kết luận

1. **Thiếu benchmark cuối trên test.** Kết quả được trích đang là validation trên tập con. Chọn cấu hình trên validation rồi báo cáo test riêng, tránh chọn lại cấu hình theo test.
2. **Chi phí ảnh sạch đáng chú ý.** Trong các CSV PPD32m vừa nêu, clean accuracy là 74,5% với ResNet và 49,0% với EfficientNet khi dùng checkpoint huấn luyện `none`. Cần so sánh với `none` trên đúng cùng mẫu để định lượng mức suy giảm.
3. **Tấn công thích nghi cần mạnh và được kiểm tra.** PGD hiện chỉ một lần khởi tạo và không lưu bước mạnh nhất; cần thử nhiều restart, nhiều số bước và kiểm tra hội tụ. BPDA là xấp xỉ, không tự động bảo đảm attack đã đủ mạnh.
4. **Không trộn checkpoint hoặc tập mẫu.** Checkpoint `resnet50_none` rồi thêm PPD lúc đánh giá khác với checkpoint `resnet50_ppd32m` được train theo defense. `summarize_results.py` hiện có thể ghép kết quả từ checkpoint khác nhau vào cùng hàng; không dùng trực tiếp bảng tổng hợp hiện tại làm kết luận.
5. **Thiếu metadata để tái lập.** Cần lưu lệnh, seed, split, danh sách mẫu, tên và hash checkpoint, phiên bản mã, số bước, tham số attack, cấu hình defense và thời gian. Manifest trong `configs/` là kế hoạch tham khảo, không được script tự nạp và trạng thái có thể đã cũ.
6. **Giới hạn của dữ liệu và giả thuyết.** Đánh giá trên crop 5 lớp không phản ánh đầy đủ phát hiện vật thể; cần phân tích phân bố lớp, sai số màu và độ méo ảnh sạch trước khi khẳng định giả thuyết palette.

## 6. Lộ trình tiếp theo đề xuất

1. **Chốt giao thức:** ghi rõ split, seed, danh sách crop, checkpoint, số mẫu và metric; đảm bảo so sánh các defense trên cùng mẫu và cùng checkpoint clean. Thí nghiệm huấn luyện cùng defense để riêng.
2. **Kiểm tra nền tảng:** kiểm tra dữ liệu, nhãn, phân bố lớp, rò rỉ split; bổ sung kiểm thử trực tiếp cho palette, PPD và attack thích nghi. Kiểm tra chuẩn nhiễu thực tế theo từng mẫu.
3. **Hoàn thiện validation:** chạy clean, FGSM, PGD và C&W cho ít nhất ResNet-50 và EfficientNet-B0, với `none`, JPEG, Gaussian blur, bit-depth và PPD. Báo cáo adaptive và oblivious riêng; tăng bước/restart PGD và kiểm tra C&W phù hợp. Ưu tiên kiểm tra sâu JPEG75 và PPD16m trong hai thiết lập riêng.
4. **Ablation có kiểm soát:** K = 16, 32, 64; mỗi K có và không có median; khảo sát JPEG 50/75/90, blur 0,5/1,0 và bit 3/4. Giữ nguyên dữ liệu, checkpoint và tham số attack trong mỗi nhóm; đo biến dạng ảnh sạch, thời gian và bộ nhớ của toàn bộ pipeline.
5. **Chọn cấu hình và đánh giá test:** sau khi chọn trên validation, chạy test theo giao thức đã chốt; báo cáo số mẫu, metric theo lớp nếu cần và độ bất định phù hợp.
6. **Tổng hợp có kiểm chứng:** sửa hoặc thay cách tổng hợp để mỗi hàng giữ đúng model, checkpoint, split, defense, attack, tham số và nguồn CSV. Bổ sung robust accuracy có điều kiện; đo PSNR/SSIM giữa ảnh sạch và ảnh sau defense; báo cáo C&W thích nghi kèm L2 và số mẫu thành công. Lưu biểu đồ, ảnh trước/sau defense và toàn bộ cấu hình đi kèm.
7. **Mô hình thứ ba nếu thời gian cho phép:** huấn luyện và đánh giá ViT-S theo cùng giao thức; hai mô hình đã có vẫn đáp ứng mục tiêu tối thiểu 2–3 mô hình nếu hoàn thành benchmark.

## 7. Nguồn tham khảo trong dự án

Các đường dẫn sau tính từ thư mục `adversarial-attack-project/`:

- `ppd_resnet.py`: dữ liệu, palette, defense, backbone, train/eval và selftest.
- `attack_eval.py`: attack, adaptive/BPDA, metric và xuất CSV.
- `data/tach_validation.py`, `data/thu_mau.py`: chia validation và phân tích màu.
- `docs/project_structure.md` và `experiments/ppd_classification/README.md`: cấu trúc, phạm vi và cách tổ chức thí nghiệm.
- `experiments/ppd_classification/configs/`: cấu hình dự kiến và ma trận công việc, không phải bằng chứng mọi thí nghiệm đã hoàn tất.
- `experiments/ppd_classification/metrics/`: lịch sử train và CSV gốc. Bảng PGD/C&W của PPD32m dùng các file có tiền tố `attack_resnet50_none_as_ppd32m_` hoặc `attack_efficientnet_b0_none_as_ppd32m_`. Phần tên còn lại lần lượt là `val_main.csv` hoặc `val_val_next.csv` cho kết quả không thích nghi; `val_adaptive_adapt30mean.csv` hoặc `val_adaptive_val_next.csv` cho kết quả thích nghi. Số liệu EfficientNet không defense lấy từ file có tiền tố `attack_efficientnet_b0_none_as_none_` và đuôi `val_val_next.csv`.
- `summarize_results.py`: công cụ tổng hợp hiện tại, cần kiểm tra và sửa cách gộp trước khi dùng cho kết luận.

**Kết luận:** Dự án đã có nền tảng thực nghiệm đúng hướng và dữ liệu ban đầu cho hai mô hình. Công việc còn lại quan trọng nhất là hoàn thiện đánh giá có kiểm soát, đặc biệt với attack thích nghi, rồi kiểm chứng trên test trước khi khẳng định hiệu quả của PPD.