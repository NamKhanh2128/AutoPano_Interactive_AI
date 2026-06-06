# AutoPano Interactive AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green?logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![Kornia](https://img.shields.io/badge/Kornia-LoFTR-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Hệ thống ghép ảnh panorama tự động thế hệ mới, kết hợp Deep Learning (LoFTR) với Camera Calibration (Zhang Zhengyou) để tạo ra ảnh toàn cảnh liền mạch, chất lượng cao.**

</div>

---

## Mục lục

- [Giới thiệu](#giới-thiệu)
- [Kiến trúc Pipeline](#kiến-trúc-pipeline)
- [Công nghệ & Thuật toán](#công-nghệ--thuật-toán)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [Camera Calibration](#camera-calibration)
- [Thông số cấu hình](#thông-số-cấu-hình)
- [Kết quả & Đánh giá](#kết-quả--đánh-giá)

---

## Giới thiệu

**AutoPano Interactive AI** là một pipeline ghép ảnh panorama tự động hoàn chỉnh, được xây dựng trên nền tảng Hybrid AI — kết hợp sức mạnh của mạng nơ-ron sâu **LoFTR** với các thuật toán hình học máy tính cổ điển được tối ưu hóa cho panorama chuyên nghiệp.

### Điểm nổi bật

| Tính năng | Mô tả |
|---|---|
| 🤖 **AI Feature Matching** | LoFTR (Transformer-based) — tìm hàng nghìn điểm khớp ngay cả trên bề mặt ít texture |
| 🎯 **Camera Calibration** | Tích hợp phương pháp Zhang Zhengyou — undistort méo ống kính trước khi ghép |
| 🔗 **Robust Filtering** | MAGSAC++ — lọc nhiễu không cần ngưỡng cứng, chính xác hơn RANSAC |
| 🌐 **BFS Chain Warping** | Lan truyền ma trận tự động qua đồ thị — ghép n ảnh theo dây chuyền |
| 🎨 **Gaussian Feathering** | Blending mượt theo trọng số khoảng cách — không còn đường seam cứng |
| ✂️ **Auto Crop** | Tự động cắt viền đen sau khi ghép |
| 📊 **Báo cáo chi tiết** | Xuất report JSON/TXT đầy đủ các chỉ số đo lường |

---

## Kiến trúc Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                     AUTO PANO AI — HYBRID PIPELINE                  │
└─────────────────────────────────────────────────────────────────────┘

 [Ảnh đầu vào]
       │
       ▼
 ┌─────────────┐
 │  Bước 1.5   │  Camera Undistortion (Zhang Zhengyou)
 │  Undistort  │  cv2.remap + INTER_LANCZOS4 (alpha=0.5)
 └──────┬──────┘  → Loại bỏ méo hướng tâm & tiếp tuyến
        │
        ▼
 ┌─────────────┐
 │   Bước 2    │  Tiền xử lý CLAHE
 │  Preprocess │  Contrast Limited Adaptive Histogram Equalization
 └──────┬──────┘  → Tăng tương phản cục bộ, giữ chi tiết vùng tối/sáng
        │
        ▼
 ┌─────────────┐
 │  Bước 3-4   │  AI Feature Matching — LoFTR Neural Network
 │  Matching   │  + MAGSAC++ Outlier Rejection
 └──────┬──────┘  → Hàng nghìn keypoints chính xác; chọn Anchor Image
        │
        ▼
 ┌─────────────┐
 │   Bước 5    │  BFS Projective Warping
 │   Warping   │  Lan truyền chuỗi: H_AB × H_BC × ... × H → Canvas
 └──────┬──────┘  → Mọi ảnh ánh xạ lên hệ tọa độ canvas chung
        │
        ▼
 ┌─────────────┐
 │   Bước 6    │  Gaussian Feathering Blending
 │  Blending   │  weight = GaussianBlur(mask) → blend theo trọng số
 └──────┬──────┘  → Vùng chồng lấp mượt mà, không đường cắt cứng
        │
        ▼
 ┌─────────────┐
 │  Bước 6.5   │  Auto Crop + Unsharp Mask
 │  Finishing  │  findNonZero → boundingRect → crop viền đen
 └──────┬──────┘  Unsharp Mask (1.3× − 0.3×Gaussian) → tăng độ nét
        │
        ▼
 [panorama_result.jpg]
```

---

## Công nghệ & Thuật toán

### 1. Camera Undistortion (Bước 1.5)

Dựa trên phương pháp **Zhang Zhengyou (1999)** — camera calibration bằng bàn cờ checkerboard:

$$s\tilde{m} = K[R \mid t]\tilde{M}$$

Mô hình biến dạng được hiệu chỉnh:

$$x_{distorted} = x(1 + k_1r^2 + k_2r^4 + k_3r^6) + 2p_1xy + p_2(r^2 + 2x^2)$$

Kỹ thuật remapping cache (`initUndistortRectifyMap`) tính map1, map2 một lần — áp dụng O(N) cho toàn bộ batch, nhanh hơn `cv2.undistort` nhiều lần.

### 2. LoFTR Neural Network (Bước 3-4)

**LoFTR (Local Feature TRansformer)** là mạng Transformer nhận diện điểm khớp dense trực tiếp từ ảnh xám, không cần bước trích xuất keypoint riêng:

- Xử lý ở độ phân giải **1024px** (4× nhiều keypoints hơn 640px)
- Ngưỡng confidence > 0.5 (lấy nhiều điểm tốt)
- Trả về tọa độ đã scale về kích thước ảnh gốc

### 3. MAGSAC++ Robust Estimation (Bước 3-4)

Thay RANSAC bằng **MAGSAC++** để tính Homography:

```python
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC,
                              5.0, confidence=0.9999, maxIters=5000)
```

- Không cần ngưỡng inlier cứng — tự thích nghi theo phân phối noise
- Ngưỡng chấp nhận: `inliers ≥ 30` HOẶC `inlier_ratio ≥ 25%`

### 4. BFS Chain Warping (Bước 5)

Lan truyền ma trận Homography qua đồ thị kết nối:

```
H_global[neighbor] = H_global[curr] × H[neighbor → curr]
```

Anchor image (ảnh có tổng điểm ghép cao nhất) được đặt ở trung tâm canvas bằng ma trận dịch chuyển T.

### 5. Gaussian Feathering Blending (Bước 6)

```python
weight = GaussianBlur(binary_mask, sigma = max(H, W) × 0.15)
canvas_acc  += img × weight      # tích lũy weighted sum
result       = canvas_acc / weight_acc  # normalize
```

Vùng chồng lấp được blend theo gradient trọng số khoảng cách từ tâm ảnh — không còn đường seam cứng.

---

## Cấu trúc dự án

```
AutoPano_Interactive_AI/
│
├── src/                          # Source code chính
│   ├── main.py                   # Entry point — orchestrates toàn bộ pipeline
│   ├── calibration.py            # Undistort module (load .npz, remap batch)
│   ├── features.py               # Tiền xử lý CLAHE
│   ├── matching.py               # LoFTR + MAGSAC++ feature matching
│   ├── warping.py                # BFS Homography chain + auto_crop_canvas
│   ├── blending.py               # Gaussian Feathering Blending
│   └── logger.py                 # Metrics logger (JSON + TXT report)
│
├── data/
│   ├── input/                    # ← Đặt ảnh cần ghép vào đây (JPG/PNG)
│   ├── output/                   # Kết quả panorama + debug matches
│   │   └── matches_debug/        # Visualize keypoints AI tìm được
│   ├── logs/                     # Report.json + report.txt
│   └── calib/                    # camera_params.npz (từ calibration)
│
├── copy_calib.py                 # Helper: copy calibration params 1 lệnh
├── requirements.txt
└── README.md
```

---

## Yêu cầu hệ thống

| Thành phần | Yêu cầu tối thiểu | Khuyến nghị |
|---|---|---|
| **Python** | 3.8+ | 3.10+ |
| **RAM** | 8 GB | 16 GB |
| **GPU** | Không bắt buộc | NVIDIA CUDA (tăng tốc 5-10×) |
| **Dung lượng** | 3 GB (model LoFTR) | SSD |
| **OS** | Windows / Linux / macOS | — |

---

## Cài đặt

```bash
# 1. Clone repository
git clone https://github.com/NamKhanh2128/AutoPano_Interactive_AI.git
cd AutoPano_Interactive_AI

# 2. Tạo môi trường ảo (khuyến nghị)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 3. Cài đặt thư viện
pip install -r requirements.txt
```

> **Lưu ý:** Lần chạy đầu tiên, model LoFTR (~100MB) sẽ được tự động tải về. Cần kết nối internet.

---

## Hướng dẫn sử dụng

### Chạy nhanh (không cần calibration)

```bash
# 1. Đặt 2+ ảnh chồng lấn vào thư mục input
#    (chụp bằng cách xoay camera ngang, chồng lấp 20-50%)
copy anh1.jpg anh2.jpg anh3.jpg data\input\

# 2. Chạy pipeline
cd src
python main.py

# 3. Kết quả
#    → data/output/panorama_result.jpg
#    → data/logs/report.txt
```

### Chạy với Camera Calibration (chất lượng tốt nhất)

```bash
# Bước A: Calibrate camera (xem phần Camera Calibration bên dưới)
# Sau khi có camera_params.npz, copy sang AutoPano:

python copy_calib.py          # Tự động tìm và copy từ thư mục Calibration

# Bước B: Chạy bình thường — pipeline tự động undistort
cd src
python main.py
```

### Chạy trình xem web 360°

```bash
cd web_viewer
python -m http.server 8000
# Truy cập: http://localhost:8000
```

---

## Camera Calibration

Để đạt chất lượng tốt nhất (đặc biệt với ống kính điện thoại hoặc wide-angle), hãy thực hiện camera calibration trước khi ghép panorama.

### Yêu cầu

- **Bàn cờ calibration**: in bàn cờ 12×9 (11×8 inner corners), kích thước ô 17mm
- **Ảnh calibration**: 15-30 ảnh chụp bàn cờ ở các góc độ và khoảng cách khác nhau
- **Dự án calibration**: [Calibration-ZhangZhengyou-Method](https://github.com/NamKhanh2128/Calibration-ZhangZhengyou-Method)

### Quy trình

```bash
# Trong dự án Calibration-ZhangZhengyou-Method:

# 1. Đặt ảnh bàn cờ vào:
#    pic/RGB_camera_calib_img/

# 2. Chạy calibration
python calibrate.py
# → Xuất: camera_params.npz (K, D, RMS error)
# → Xuất: pic/reprojection_error.png (biểu đồ đánh giá)

# 3. Copy sang AutoPano (1 lệnh)
cd ../AutoPano_Interactive_AI
python copy_calib.py
```

### Chỉ số chất lượng calibration

| RMS (pixels) | Đánh giá |
|---|---|
| < 0.3 px | ✔✔✔ XUẤT SẮC |
| 0.3 – 0.5 px | ✔✔ TỐT |
| 0.5 – 1.0 px | ✔ CHẤP NHẬN ĐƯỢC |
| > 1.0 px | ✘ Cần thêm ảnh / kiểm tra lại |

### Tham số undistortion cho panorama

| alpha | Ý nghĩa | Khuyến nghị |
|---|---|---|
| `0.0` | Cắt sạch viền đen, mất pixel rìa | Ảnh thông thường |
| `0.5` | Cân bằng — giữ hầu hết pixels | **Panorama** ← mặc định |
| `1.0` | Giữ tất cả pixels, có viền đen nhỏ | Fisheye / GoPro |

---

## Thông số cấu hình

### matching.py

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `MAX_SIZE` | `1024` px | Độ phân giải LoFTR — cao hơn = nhiều keypoints hơn |
| `confidence` | `> 0.5` | Ngưỡng độ tin cậy AI |
| `min_inliers` | `30` | Số inliers tối thiểu để chấp nhận cặp ghép |
| `min_ratio` | `0.25` | Tỷ lệ inlier tối thiểu |

### blending.py

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `sigma_ratio` | `0.15` | Độ rộng feathering (0.1=nhẹ, 0.15=vừa, 0.25=mạnh) |

### calibration.py

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `alpha` | `0.5` | Hệ số giữ pixel sau undistort |
| Interpolation | `INTER_LANCZOS4` | Nội suy chất lượng cao nhất |

---

## Kết quả & Đánh giá

### Pipeline so sánh

| Tính năng | Phiên bản cũ | Phiên bản mới |
|---|---|---|
| Feature matching | SIFT/ORB + FLANN | **LoFTR Deep Learning** |
| Outlier rejection | RANSAC | **MAGSAC++** |
| Camera distortion | Không xử lý | **Undistort (Zhang Zhengyou)** |
| Blending | Ghép đè trực tiếp | **Gaussian Feathering** |
| Canvas cleanup | Không | **Auto Crop** |
| Calibration export | Không | **camera_params.npz** |

### Chỉ số báo cáo (data/logs/report.txt)

- Số điểm AI tìm được (LoFTR keypoints)
- Số inliers sau MAGSAC++ và tỷ lệ inlier/keypoints
- Thời gian xử lý từng bước (ms)
- Tổng thời gian pipeline

---

## Thư viện sử dụng

| Thư viện | Mục đích |
|---|---|
| `opencv-python` | Xử lý ảnh, undistort, warping, blending |
| `numpy` | Tính toán ma trận và số học |
| `torch` + `torchvision` | Deep Learning runtime cho LoFTR |
| `kornia` | LoFTR model và các thuật toán thị giác AI |

---

## License

MIT License © 2026 NamKhanh2128
