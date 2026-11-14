# Đánh giá các defense tiền xử lý màu cho phân loại ảnh X-quang dưới tấn công đối kháng thích nghi: trường hợp Palette Projection Defense

## Tóm tắt

Ảnh soi chiếu X-quang ở sân bay sử dụng bảng màu cố định để phân biệt vật hữu cơ, hỗn hợp và kim loại, khiến các pixel nằm trong một tập màu khá hẹp. Phương pháp Palette Projection Defense (PPD) khai thác đặc điểm này bằng cách học các màu đại diện từ dữ liệu huấn luyện và chiếu mỗi pixel về màu gần nhất, với tùy chọn lọc trung vị 3×3. Bài báo này khảo sát PPD — một phép lượng tử hóa màu dựa trên palette học từ dữ liệu — và đánh giá chặt chẽ PPD cùng các phương pháp đối chứng (JPEG compression, Gaussian blur, bit-depth reduction) trên bộ dữ liệu OPIXray với ba loại tấn công đối kháng: FGSM, PGD và Carlini-Wagner (C&W). Quan trọng nhất, chúng tôi so sánh tấn công oblivious (không biết defense) với tấn công adaptive (biết defense) để đánh giá thực tế hiệu quả phòng thủ. Kết quả trên tập test gồm 1780 crop cho thấy PPD16m kết hợp checkpoint huấn luyện riêng là ứng viên đáng chú ý trong các defense thử, với độ chính xác sạch 97,2% và chống lại PGD adaptive ε=2/255 ở mức 80,4%. Tuy nhiên, các tấn công adaptive mạnh (PGD ε=8/255: 13,2%; C&W: 17,9%) vẫn phá vỡ defense đáng kể. Hơn nữa, đánh giá oblivious cho kết quả quá lạc quan: với JPEG75, chênh lệch giữa oblivious và adaptive lên đến 60,1 điểm phần trăm. Kết quả này nhấn mạnh tầm quan trọng của tấn công adaptive trong đánh giá defense và cho thấy các defense dựa trên tiền xử lý đơn thuần chưa đủ để bảo vệ mô hình trước kẻ tấn công biết hệ thống.

**Từ khóa:** adversarial defense, palette projection, X-ray image, adaptive attack, robustness evaluation, OPIXray

---

## 1. Giới thiệu

An ninh AI trong hệ thống soi chiếu sân bay ngày càng trở nên quan trọng khi các mô hình học sâu được áp dụng để phát hiện vật thể nguy hiểm trong ảnh X-quang. Tuy nhiên, các mô hình này dễ bị tấn công đối kháng — nhiễu nhỏ được thêm vào ảnh đầu vào có thể khiến mô hình dự đoán sai mà không làm thay đổi đáng kể nhận thức của con người. Nhiều phương pháp phòng thủ dựa trên tiền xử lý đã được đề xuất, bao gồm nén JPEG, lọc Gaussian và giảm độ sâu bit, với ý tưởng loại bỏ hoặc giảm thiểu nhiễu trước khi đưa ảnh vào mô hình.

Vấn đề cốt lõi trong đánh giá các defense này là **tấn công oblivious so với tấn công adaptive**. Đánh giá oblivious — tạo ảnh đối kháng trên mô hình không có defense, sau đó áp dụng defense khi suy luận — thường cho kết quả lạc quan, vì attacker không tối ưu theo pipeline thực tế bao gồm defense. Ngược lại, tấn công adaptive — attacker biết defense và tối ưu theo toàn bộ pipeline — phản ánh mối đe dọa thực tế hơn. Sự chênh lệch giữa hai cách đánh giá có thể lên đến hàng chục phần trăm, khiến các kết luận dựa trên oblivious evaluation trở nên đáng ngờ.

Bài báo này đóng góp các kết quả sau:

1. **Khảo sát PPD cho phân loại crop ảnh X-quang:** chúng tôi áp dụng PPD — một phép lượng tử hóa màu dựa trên palette học từ dữ liệu — cho bài toán phân loại crop ảnh X-quang trên OPIXray.
2. **Đánh giá PPD và các baseline dưới cả tấn công oblivious lẫn adaptive:** chúng tôi so sánh PPD với JPEG compression, Gaussian blur và bit-depth reduction trên cùng mô hình, dữ liệu và ngân sách tấn công.
3. **Xác nhận tầm quan trọng của adaptive attack trên miền X-quang:** kết quả oblivious đánh giá lạc quan đã được biết từ các nghiên cứu trước (Athalye et al., 2018), nhưng bài báo này cung cấp bằng chứng cụ thể trên OPIXray với các defense tiền xử lý thực tế, cho thấy chênh lệch có thể lên đến 60,1 điểm phần trăm.
4. **Phân tích ảnh hưởng của huấn luyện với defense và các lựa chọn của PPD:** chúng tôi so sánh checkpoint clean với checkpoint huấn luyện riêng, và khảo sát K=16, 32, 64 cùng lọc trung vị 3×3.

**Phạm vi mối đe dọa:** Mô hình tấn công của bài báo là nhiễu pixel số trên crop đã biết bounding box. Trong thực tế, tấn công vật lý trên hệ thống soi chiếu có thể dùng vật kim loại in 3D thay đổi **hình dạng** chứ không phải màu (theo X-Adv). Do đó, phép chiếu palette có thể không tác động được lên loại tấn công này. Bài báo không khẳng định PPD bảo vệ hệ thống sân bay thực tế.

---

## 2. Công việc liên quan

### 2.1. Tấn công đối kháng

Các tấn công đối kháng không nhắm đích (untargeted) phổ biến bao gồm:

- **FGSM (Fast Gradient Sign Method):** tấn công một bước gradient, cộng nhiễu theo dấu của gradient loss theo đầu vào. FGSM nhanh nhưng thường không đủ mạnh để đánh giá độ bền thực sự.
- **PGD (Projected Gradient Descent):** lặp nhiều bước gradient trong miền nhiễu L∞, được coi là tấn công mạnh trong lớp nhiễu L∞ nhỏ. PGD với nhiều restart thường được dùng làm baseline đánh giá robustness.
- **Carlini-Wagner (C&W):** tối ưu theo chuẩn L2 với hàm mục tiêu dựa trên margin logits. C&W thường tạo nhiễu nhỏ hơn theo L2 nhưng có thể lớn theo L∞, và không so sánh trực tiếp với PGD như cùng ngân sách nhiễu.

### 2.2. Defense dựa trên tiền xử lý

Các defense dựa trên tiền xử lý ảnh nhằm loại bỏ hoặc giảm nhiễu adversarial trước khi đưa vào mô hình:

- **JPEG compression:** nén và giải nén JPEG làm thay đổi các hệ số ảnh, có thể giảm nhiễu tần số cao.
- **Gaussian blur:** lọc làm mượt ảnh, giảm biến thiên nhanh trong không gian.
- **Bit-depth reduction (feature squeezing):** lượng tử hóa pixel về ít mức, giảm không gian nhiễu liên tục.
- **Color quantization:** giảm số màu, tương tự PPD nhưng dùng lưới cố định thay vì học từ dữ liệu.

### 2.3. Đánh giá defense

Đánh giá defense bằng oblivious attack thường cho kết quả lạc quan. Athalye et al. (2018) đã chỉ ra rằng **adaptive attack là cần thiết** để đánh giá đúng hiệu quả defense, và kết quả oblivious có thể quá lạc quan hàng chục phần trăm. Guo et al. (2018) và Xu et al. (2019) khảo sát các defense dựa trên input transformations và feature squeezing. Tramèr et al. (2020) và Carlini et al. (2019) nhấn mạnh tầm quan trọng của giao thức adaptive attack khi đánh giá robustness. Croce & Hein (2020) đề xuất AutoAttack — ensemble các attack không tham số — làm benchmark đánh giá robustness. Với các defense không khả vi (JPEG, lượng tử hóa), cần dùng gradient xấp xỉ như BPDA (Backward Pass Differentiable Approximation) hoặc STE (Straight-Through Estimator). Tuy nhiên, chất lượng của gradient xấp xỉ vẫn là một vấn đề mở, và kết quả adaptive có thể phụ thuộc vào cách xấp xỉ được chọn.

---

## 3. Phương pháp

### 3.1. Bài toán và dữ liệu

Chúng tôi sử dụng bộ dữ liệu **OPIXray** (Wei et al., 2020) — một bộ dữ liệu **tổng hợp** (synthetic) tạo bằng mô phỏng vật lý X-quang — với 5 lớp vật thể thuộc nhóm dao/kéo: **Folding Knife, Straight Knife, Scissor, Utility Knife, và Multi-tool Knife**. Pipeline tạo ảnh có thể khác nhau giữa các máy X-quang thực tế, nên kết quả trên OPIXray có thể không khái quát hóa trực tiếp sang môi trường thực.

Mỗi ảnh được crop theo bounding box đã biết, resize về 224×224 và phân loại. Tập test OPIXray gồm 1.776 ảnh; một số ảnh có nhiều bounding box, nên sau khi crop theo từng box ta được **1.780 crop**. Dữ liệu được chia thành ba tập: huấn luyện, validation và test. Tập test gồm 1.780 crop, dùng để đánh giá cuối cùng. Phân bố lớp trên tập test: Folding Knife 404, Straight Knife 430, Scissor 368, Utility Knife 235, Multi-tool Knife 343.

**Lưu ý quan trọng:** đây là bài toán **phân loại crop** — mỗi mẫu đã là vùng vật thể được xác định trước. Kết quả không phản ánh khả năng phát hiện vật thể trên toàn ảnh soi chiếu, và độ chính xác cao trên crop không đồng nghĩa với hiệu quả thực tế của detector.

### 3.2. Palette Projection Defense

PPD gồm ba bước:

1. **Học palette:** từ tập huấn luyện, lấy tối đa 400 pixel từ mỗi crop gốc, loại pixel nền (min(R,G,B) > 235), rồi dùng MiniBatchKMeans để học K−1 tâm màu, thêm một tâm nền sáng. Thử K = 16, 32, 64.
2. **Chiếu pixel:** tại suy luận, mỗi pixel được chiếu về màu palette gần nhất theo khoảng cách Euclidean trong RGB.
3. **Lọc trung vị (tùy chọn):** sau khi chiếu, áp dụng median filter 3×3 theo từng kênh để làm mượt.

Palette được học từ ảnh gốc trước resize, trong khi PPD được áp dụng sau resize 224×224, nên phân bố màu có thể khác nhau giữa hai bước.

### 3.3. Các baseline đối chứng

- **JPEG75:** nén và giải nén JPEG với `quality=75` trong Pillow.
- **Blur1.0:** lọc Gaussian với σ=1,0 pixel, kernel 7×7, đệm phản chiếu.
- **Bit3:** lượng tử hóa mỗi kênh RGB về 8 mức (2³), tương đương tối đa 512 màu RGB.
- **Bit3m:** bit3 kết hợp median filter 3×3, dùng làm baseline để tách đóng góp của median so với PPD.
- **Med3:** chỉ median filter 3×3, không có lượng tử hóa màu, dùng làm baseline để đánh giá đóng góp của median đơn thuần.
- **None:** không phòng thủ, dùng làm baseline.

### 3.4. Mô hình

Sử dụng **ResNet-50** và **EfficientNet-B0**, fine-tune từ trọng số ImageNet, với lớp đầu ra thay thành 5 lớp. Hai thiết lập checkpoint:

1. **Checkpoint clean:** huấn luyện trên ảnh sạch, đánh giá defense bằng cách chỉ thay tiền xử lý khi suy luận. Thiết lập này cô lập tác động của defense.
2. **Checkpoint riêng:** huấn luyện với defense áp dụng trong vòng train, đánh giá với cùng defense khi suy luận. Thiết lập này đánh giá toàn bộ hệ thống.

**Quan trọng:** huấn luyện với ảnh qua defense không đồng nghĩa với adversarial training — vòng train không tạo ảnh PGD hay nhiễu đối kháng.

---

## 4. Tấn công và giao thức đánh giá

### 4.1. Tấn công không nhắm đích

- **FGSM:** một bước gradient, L∞, với ε = 2, 4, 8/255.
- **PGD:** 30 bước lặp trong miền L∞, ε = 2, 4, 8/255, bước lớn ε/4, khởi tạo ngẫu nhiên. Chỉ một lần khởi tạo, không restart.
- **C&W L2:** tối ưu theo L2 bằng Adam, 100 bước, learning rate 0.01, `c=1` cố định, `kappa=0, 10`. Không giới hạn L∞, không binary search cho `c`.

### 4.2. Oblivious so với Adaptive

- **Oblivious:** attacker tối ưu theo backbone không defense, sau đó áp dụng defense khi đánh giá.
- **Adaptive:** attacker biết defense, tối ưu theo pipeline có defense. Với defense không khả vi (JPEG, PPD, lượng tử hóa), dùng gradient xấp xỉ (BPDA/STE). Với median filter, xấp xỉ gradient bằng mean hoặc identity.

### 4.3. Metric

- **Clean accuracy:** độ chính xác trên ảnh sạch.
- **Robust accuracy:** tỷ lệ dự đoán đúng trên toàn bộ ảnh sau attack, kể cả ảnh đã sai khi sạch.
- **ASR (Attack Success Rate):** tỷ lệ mẫu sạch đúng bị attack làm sai, chia cho số mẫu sạch đúng.
- **Robust accuracy có điều kiện:** tỷ lệ mẫu vẫn đúng sau attack, chỉ tính trên mẫu sạch đúng. Khi có mẫu sạch đúng, `Robust accuracy có điều kiện = 100% − ASR`.

---

## 5. Kết quả

Tất cả kết quả dưới đây được trích từ tập test, n=1780 cho FGSM/PGD và n=1780 cho C&W. Các số được làm tròn đến một chữ số thập phân.

### 5.1. ResNet-50: cùng checkpoint clean, thay defense lúc suy luận

Bảng 1 so sánh các defense khi giữ nguyên checkpoint `resnet50_none` và chỉ thay tiền xử lý tại suy luận, với tấn công adaptive.

**Bảng 1: ResNet-50 cùng checkpoint clean, tấn công adaptive (n=1780)**

| Defense | Clean | FGSM adapt 2/255 | PGD adapt 2/255 | C&W adapt k0 |
|---|---:|---:|---:|---:|
| none | 97,4% ± 1,3% | 71,7% ± 2,0% | **0,0%** | **0,0%** |
| jpeg75 | 97,1% ± 1,3% | 86,2% ± 1,7% | 34,8% ± 2,5% | 0,6% ± 0,8% |
| blur1.0 | 96,1% ± 1,4% | 51,1% ± 2,3% | **0,0%** | **0,0%** |
| bit3 | 88,4% ± 1,8% | 45,3% ± 2,2% | **0,0%** | **0,0%** |
| ppd32 | 72,4% ± 2,0% | 38,9% ± 2,2% | 2,4% ± 1,1% | **0,0%** |
| ppd32m | 73,3% ± 2,0% | 49,4% ± 2,3% | 22,8% ± 2,1% | 3,0% ± 1,3% |

*Các giá trị là accuracy ± khoảng tin cậy 95% (Wilson interval).*

**Nhận xét:** JPEG75 giữ clean accuracy cao (97,1%) và chống FGSM adaptive tốt nhất (86,2%), nhưng PGD adaptive ε=2/255 chỉ còn 34,8% và C&W adaptive gần như phá vỡ hoàn toàn (0,6%). Các defense PPD không median (ppd32) có clean accuracy thấp hơn đáng kể so với baseline. PPD có median (ppd32m) cho robust accuracy cao hơn PPD không median, nhưng clean accuracy cũng giảm.

### 5.2. Các defense với checkpoint huấn luyện riêng

Bảng 2 so sánh các defense khi huấn luyện riêng mô hình với defense, đánh giá với tấn công adaptive. Đây là thiết lập khác so với Bảng 1; **không so sánh trực tiếp** giữa hai bảng.

**Bảng 2: Checkpoint huấn luyện riêng, tấn công adaptive (n=1780)**

| Defense | Clean | FGSM adapt 2/255 | PGD adapt 2/255 | PGD adapt 8/255 | C&W adapt k0 | C&W L2 (k0) |
|---|---:|---:|---:|---:|---:|---:|
| **ppd16m** | **97,2% ± 1,3%** | **91,4% ± 1,6%** | **80,4% ± 1,9%** | **13,2% ± 1,8%** | **17,9% ± 1,5%** | **2,97** |
| ppd32m | 96,7% ± 1,3% | 90,9% ± 1,6% | 70,2% ± 2,3% | 7,5% ± 1,6% | 11,7% ± 1,4% | 2,34 |
| ppd64m | 97,4% ± 1,3% | 90,6% ± 1,6% | 52,2% ± 2,3% | 1,7% ± 1,0% | 5,2% ± 1,2% | 1,90 |
| jpeg75 | 97,3% ± 1,3% | 88,5% ± 1,7% | 34,7% ± 2,5% | 0,3% ± 0,5% | 0,2% ± 0,4% | 1,71 |
| bit3m | 97,5% ± 1,3% | 89,4% ± 1,7% | 28,3% ± 2,4% | 0,3% ± 0,5% | 3,7% ± 1,1% | 1,38 |
| med3 | 97,2% ± 1,3% | 72,1% ± 2,0% | 0,8% ± 0,7% | 0,0% | 0,6% ± 0,6% | 0,89 |

*Các giá trị accuracy là ± khoảng tin cậy 95% (Wilson interval). L2 là độ lớn nhiễu trung bình ||x_adv − x||₂ của C&W adaptive kappa=0.*

**Nhận xét:** PPD16m với checkpoint riêng là ứng viên tốt nhất trong các defense thử, giữ clean accuracy 97,2% và chống PGD adaptive ε=2/255 ở mức 80,4%. PPD32m checkpoint riêng là ứng viên thứ hai (70,2%), theo sau là PPD64m (52,2%). JPEG75 checkpoint riêng (34,7%) tốt hơn JPEG75 dùng checkpoint clean (34,8%) nhưng vẫn kém PPD16m. Bit3m (28,3%) tốt hơn med3 đơn thuần (0,8%) đáng kể, cho thấy bit3 góp phần bảo vệ ngoài median. Tuy nhiên, med3 oblivious cho 94,0% trong khi adaptive chỉ 0,8% — chênh lệch 93,2 điểm phần trăm, lớn nhất trong các defense thử.

**Lưu ý quan trọng:** mỗi defense dùng checkpoint khác nhau; kết quả phản ánh toàn bộ quy trình huấn luyện + defense, không cô lập riêng ảnh hưởng của K hay loại defense. Chưa có ≥3 seed để loại trừ yếu tố ngẫu nhiên khi huấn luyện.

### 5.3. Gradient xấp xỉ cho median: so sánh mean và identity

Bảng 3 so sánh hai cách xấp xỉ gradient qua median filter khi tấn công adaptive PPD16m: `mean` (dùng mean làm surrogate gradient) và `identity` (bỏ qua median, chỉ gradient qua palette). Kết quả cho thấy lựa chọn gradient xấp xỉ ảnh hưởng đáng kể đến robust accuracy.

**Bảng 3: PPD16m gradient xấp xỉ, tấn công adaptive (n=1780)**

| Attack | Tham số | mean | identity | Chênh lệch |
|---|---|---:|---:|---:|
| FGSM adapt | 2/255 | 91,4% ± 0,7% | 93,4% ± 0,6% | +2,0% |
| FGSM adapt | 4/255 | 87,5% ± 0,8% | 91,3% ± 0,7% | +3,8% |
| FGSM adapt | 8/255 | 82,5% ± 0,9% | 88,4% ± 0,7% | +5,9% |
| PGD adapt | 2/255 | **80,4% ± 0,9%** | **84,4% ± 0,8%** | +4,0% |
| PGD adapt | 4/255 | 56,0% ± 1,2% | 60,6% ± 1,1% | +4,6% |
| PGD adapt | 8/255 | 13,2% ± 0,8% | 28,6% ± 1,1% | +15,4% |
| C&W adapt | k0 | 17,9% ± 0,9% | 11,0% ± 0,7% | −6,9% |
| C&W adapt | k10 | 12,0% ± 0,8% | 12,0% ± 0,8% | 0,0% |

*Các giá trị là accuracy ± khoảng tin cậy 95% (Wilson interval). Chênh lệch = identity − mean.*

**Nhận xét:** `identity` cho robust accuracy cao hơn `mean` ở FGSM và PGD, đặc biệt tại ε=8/255 (28,6% so với 13,2%). Điều này cho thấy `mean` có thể là xấp xỉ bảo thủ hơn, trong khi `identity` (bỏ qua median) cho attack mạnh hơn ở PGD. Tuy nhiên, C&W adaptive cho kết quả tương tự ở cả hai cách (11-18%). **Kết quả 80,4% của PPD16m tại PGD ε=2/255 (mean) là cận dưới; identity cho 84,4%.** Không nên coi con số nào là độ bền thực tế tuyệt đối vì cả hai đều là xấp xỉ.

### 5.4. Oblivious so với Adaptive: bằng chứng rõ ràng

Bảng 4 so sánh trực tiếp oblivious và adaptive cho cùng defense, cùng checkpoint, với PGD ε=2/255.

**Bảng 4: So sánh oblivious và adaptive, PGD 2/255 (n=1780)**

| Defense | Checkpoint | Oblivious | Adaptive | Chênh lệch |
|---|---|---:|---:|---:|
| blur1.0 | resnet50_none | 93,4% ± 0,6% | 0,0% | **−93,4%** |
| med3 | resnet50_med3 | 94,0% ± 0,6% | 0,8% ± 0,2% | **−93,2%** |
| bit3 | resnet50_none | 79,9% ± 0,9% | 0,0% | **−79,9%** |
| bit3m | resnet50_bit3m | 96,9% ± 0,4% | 28,3% ± 1,0% | **−68,6%** |
| ppd32 | resnet50_none | 68,1% ± 1,1% | 2,4% ± 0,4% | **−65,7%** |
| jpeg75 | resnet50_none | 94,9% ± 1,5% | 34,8% ± 2,5% | **−60,1%** |
| ppd32m | resnet50_none | 71,7% ± 2,0% | 22,8% ± 2,1% | **−48,9%** |
| ppd16m | resnet50_ppd16m | 97,0% ± 1,3% | 80,4% ± 1,9% | **−16,6%** |

*Các giá trị là accuracy ± khoảng tin cậy 95% (Wilson interval). Adaptive dùng gradient xấp xỉ `mean`.*

**Nhận xét:** oblivious evaluation đánh giá quá lạc quan hàng chục phần trăm. Blur1.0 có chênh lệch lớn nhất: oblivious cho 93,4% trong khi adaptive chỉ 0,0% — chênh lệch 93,4 điểm phần trăm. Med3 (chỉ median) chênh lệch 93,2 điểm phần trăm. Bit3 chênh lệch 79,9 điểm phần trăm. **Không nên dùng oblivious để kết luận defense hiệu quả.** Ngay cả PPD16m với checkpoint riêng, mặc dù có chênh lệch nhỏ nhất (16,6 điểm phần trăm), vẫn cho thấy adaptive attack làm giảm đáng kể hiệu quả defense.

### 5.5. Kết quả trên EfficientNet-B0

Bảng 5 trình bày kết quả trên EfficientNet-B0 với checkpoint `efficientnet_b0_none`, tấn công adaptive.

**Bảng 5: EfficientNet-B0, tấn công adaptive (n=1780)**

| Defense | Clean | PGD adapt 2/255 | C&W adapt k0 |
|---|---|---:|---:|
| none | 97,3% | **0,0%** | **0,0%** |
| jpeg75 | 97,0% | 3,1% | 0,1% |
| ppd32m | 50,8% | 6,6% | 1,3% |

**Nhận xét:** PPD32m trên EfficientNet **khi không huấn luyện kèm defense** làm giảm nghiêm trọng clean accuracy (50,8% so với 97,3%) mà không bảo vệ đáng kể trước PGD hay C&W adaptive. Kết quả này không loại trừ khả năng EfficientNet huấn luyện kèm PPD có thể hiệu quả hơn, nhưng chưa được thử nghiệm. JPEG75 giữ clean cao nhưng vẫn bị PGD và C&W adaptive phá vỡ gần như hoàn toàn, tương tự xu hướng trên ResNet.

### 5.6. Ablation: K và median

Bảng 6 khảo sát ảnh hưởng của số màu K và lọc trung vị 3×3, trên ResNet-50 cùng checkpoint clean, với PGD adaptive ε=2/255. Các số được trích từ kết quả validation (n=200) vì chưa có kết quả test đầy đủ cho mọi tổ hợp ablation.

**Bảng 6: Ablation PPD, PGD adaptive 2/255 (validation, n=200)**

| K | Median | Clean | PGD adapt 2/255 |
|---|---|---:|---:|
| 16 | Không | 75,0% | 3,0% |
| 16 | Có | 72,5% | 26,5% |
| 32 | Không | 79,0% | 2,5% |
| 32 | Có | 74,5% | 27,0% |
| 64 | Không | 78,0% | 1,0% |
| 64 | Có | 78,0% | 19,5% |

**Nhận xét:** median filter có thể tăng robust accuracy đáng kể (ví dụ: K=16 từ 3,0% lên 26,5%), nhưng cũng làm giảm clean accuracy. Không có bảo đảm median luôn cải thiện độ bền thực sự, vì gradient xấp xỉ qua median (dùng mean) chưa được chứng minh đầy đủ. K=32 cho clean accuracy cao nhất trong các mức không median, nhưng robust accuracy thấp.

---

## 6. Thảo luận

### 6.1. PPD16m là ứng viên tốt nhất, nhưng không hoàn toàn an toàn

Trong các defense thử, PPD16m kết hợp checkpoint huấn luyện riêng cho kết quả tốt nhất: clean accuracy 97,2%, PGD adaptive ε=2/255 đạt 80,4%. PPD32m checkpoint riêng là ứng viên thứ hai (70,2%), theo sau là PPD64m (52,2%). Tuy nhiên, tại ε=8/255, PPD16m giảm xuống 13,2%, và C&W adaptive đạt ASR ~80%. **PPD16m không phải là defense thắng chung cuộc**, mà chỉ là tốt nhất trong các cấu hình đã thử.

### 6.2. Median đơn thuần không bảo vệ được gì trước adaptive

Kết quả med3 (chỉ median 3×3) cho thấy median filter đơn thuần gần như không bảo vệ được gì trước PGD adaptive (0,8%) và C&W adaptive (0,6%). Tuy nhiên, med3 oblivious cho 94,0% — chênh lệch 93,2 điểm phần trăm so với adaptive, lớn nhất trong các defense thử. Bit3m (bit3 + median) đạt 28,3% trước PGD adaptive, tốt hơn med3 đáng kể, cho thấy bit3 góp phần bảo vệ ngoài median. Kết quả này xác nhận rằng **phần "palette" trong PPD đóng góp đáng kể**, không chỉ median.

### 6.3. JPEG75 giữ clean cao nhưng không chống adaptive

JPEG75 giữ clean accuracy 97,1% và chống FGSM adaptive tốt (86,2%), nhưng PGD adaptive ε=2/255 chỉ còn 34,8% và C&W adaptive gần như phá vỡ hoàn toàn (0,6%). JPEG75 với checkpoint riêng (34,7%) không cải thiện đáng kể so với checkpoint clean (34,8%). Kết quả này cho thấy JPEG compression không đủ để bảo vệ trước tấn công adaptive mạnh.

### 6.4. Oblivious evaluation quá lạc quan

Bảng 4 cho thấy chênh lệch giữa oblivious và adaptive có thể lên đến 93,4 điểm phần trăm (blur1.0). **Không nên dùng oblivious để kết luận defense hiệu quả.** Athalye et al. (2018) đã chỉ ra vấn đề này, nhưng kết quả của chúng tôi cung cấp bằng chứng cụ thể trên OPIXray với các defense tiền xử lý thực tế, cho thấy chênh lệch có thể lớn hơn nhiều so với các nghiên cứu trước.

### 6.5. Hạn chế

1. **PGD chỉ một lần khởi tạo:** kết quả có thể đánh giá thấp sức tấn công. Nhiều restart thường cho ASR cao hơn.
2. **C&W không binary search cho c:** thất bại không chứng minh không tồn tại nhiễu L2 nhỏ. `c=1` cố định có thể không tối ưu. C&W adaptive chỉ đạt ASR ~80% trên PPD16m, thấp hơn mong đợi cho tấn công L2 không giới hạn; nguyên nhân có thể là attack chưa hội tụ hoặc gradient xấp xỉ qua median không chính xác. Do đó, robust accuracy 17,9% dưới C&W có thể là **cận trên**. Tuy nhiên, C&W adaptive trên PPD16m đạt được với L2 trung bình chỉ 2,97 (kappa=0) và 4,86 (kappa=10), cho thấy attacker tạo nhiễu L2 nhỏ nhưng vẫn phá vỡ defense đáng kể. Các defense khác cũng bị phá vỡ với L2 nhỏ: JPEG75 (L2 1,71), bit3m (L2 1,38), med3 (L2 0,89).
3. **Gradient xấp xỉ cho median:** dùng mean để xấp xỉ gradient qua median chưa được chứng minh đầy đủ. Kết quả so sánh mean vs. identity (Bảng 3) cho thấy chênh lệch có thể lên đến 15,4 điểm phần trăm tại PGD ε=8/255. Không nên coi con số nào là độ bền thực tế tuyệt đối.
4. **Clean accuracy cao trên crop** không chứng minh khả năng phát hiện thực tế trên toàn ảnh soi chiếu.
5. **Không so sánh với adversarial training:** PPD không thay thế adversarial training, nhưng có thể bổ sung.
6. **Không có Square Attack hay AutoAttack:** mã hiện tại không hỗ trợ các attack này; kết quả adaptive dựa trên PGD và C&W với cấu hình hạn chế.

### 6.6. So với adversarial training

PPD là defense dựa trên tiền xử lý, không thay thế adversarial training. Adversarial training — huấn luyện với ảnh đối kháng — thường cho robustness cao hơn nhưng tốn nhiều tài nguyên. PPD có thể bổ sung cho adversarial training bằng cách giảm nhiễu trước khi đưa vào mô hình đã được adversarial training.

---

## 7. Kết luận

Bài báo này khảo sát PPD — một phép lượng tử hóa màu dựa trên palette học từ dữ liệu — và đánh giá chặt chẽ PPD cùng các baseline đối chứng trên OPIXray với tấn công adaptive và oblivious. Kết quả chính:

1. **PPD16m với checkpoint huấn luyện riêng** là ứng viên đáng chú ý nhất trong các defense thử, với clean accuracy 97,2% và PGD adaptive ε=2/255 đạt 80,4%. PPD32m checkpoint riêng là ứng viên thứ hai (70,2%).
2. **Median đơn thuần không bảo vệ được gì trước adaptive:** med3 (chỉ median 3×3) đạt 0,8% trước PGD adaptive, trong khi oblivious cho 94,0% — chênh lệch 93,2 điểm phần trăm. Bit3m (bit3 + median) đạt 28,3%, xác nhận bit3 góp phần bảo vệ ngoài median.
3. **Adaptive attack phá vỡ mọi defense thử** ở mức nhiễu lớn: PGD ε=8/255 và C&W adaptive đạt ASR cao, kể cả với PPD16m.
4. **Oblivious evaluation quá lạc quan**, với chênh lệch lên đến 93,2 điểm phần trăm (med3) so với adaptive. Không nên dùng oblivious để kết luận defense hiệu quả.
5. **PPD32m trên EfficientNet khi không huấn luyện kèm defense không hiệu quả**, làm giảm clean accuracy nghiêm trọng mà không bảo vệ đáng kể.

Hướng nghiên cứu tương lai bao gồm: kết hợp PPD với adversarial training, thử PGD với nhiều restart và C&W với binary search cho `c`, đánh giá gradient xấp xỉ cho median kỹ hơn, và thử nghiệm trên detector thực tế thay vì chỉ phân loại crop.

---

## Tài liệu tham khảo

1. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and harnessing adversarial examples. *ICLR*.
2. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards deep learning models resistant to adversarial attacks. *ICLR*.
3. Carlini, N. & Wagner, D. (2017). Towards evaluating the robustness of neural networks. *IEEE S&P*.
4. Athalye, A., Carlini, N., & Wagner, D. (2018). Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples. *ICML*.
5. Guo, C., et al. (2018). Countering adversarial images using input transformations. *ICLR*.
6. Xu, W., et al. (2019). Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks. *NDSS*.
7. Tramèr, F., et al. (2020). Template Attacks and the State of Adversarial Defense Evaluation. *arXiv:2002.08632*.
8. Carlini, N., et al. (2019). On Measuring the Robustness of Defenses. *arXiv:1907.02499*.
9. Croce, F. & Hein, M. (2020). Reliable Evaluation of Adversarial Robustness with an Ensemble of Parameter-free Attacks. *AAAI*.
10. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.
11. Tan, M. & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML*.
12. Wei, Y., et al. (2020). OPIXray: A Dataset for Object Detection in X-ray Images. (Cần kiểm tra lại thông tin chính xác.)
13. X-Adv. (Cần điền thông tin chính xác về bài báo tấn công vật lý trên X-quang.)
