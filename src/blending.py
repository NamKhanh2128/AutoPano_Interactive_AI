"""
blending.py — Thành phẩm (Blending) cho AutoPano_Interactive_AI

Nâng cấp từ "ghép đè trực tiếp" lên Gaussian Distance-Weight Blending:

Vấn đề của ghép đè:
    canvas[mask] = warped[mask]  →  Đường seam cắt thẳng, lộ rõ sự chênh lệch sáng/màu

Giải pháp — Gaussian Feathering:
    Mỗi ảnh được gán trọng số dựa trên khoảng cách từ pixel đến tâm ảnh:
        weight(x,y) = Gaussian blur của mask nhị phân
    Vùng chồng lấp = blend tuyến tính theo trọng số:
        output = Σ(w_i × img_i) / Σ(w_i)
    
    Hiệu ứng: Càng gần tâm ảnh → trọng số cao → ảnh "thống trị" vùng đó
              Ở viền giao nhau → trọng số = nhau → blend mượt mà

Quy trình:
    1. Warp từng ảnh lên canvas (Projective Warping)
    2. Tính mask nhị phân (pixel có dữ liệu)
    3. GaussianBlur mask → weight map (feathering kernel sigma lớn)
    4. Tích lũy: canvas_float += img × weight
    5. Normalize: canvas_float / sum_weights (tránh chia 0)
"""

import cv2
import numpy as np
import time
from logger import pano_logger
from warping import warp_to_canvas


# ==============================================================================
def compute_weight_map(warped: np.ndarray, sigma_ratio: float = 0.15) -> np.ndarray:
    """
    Tính weight map cho một ảnh đã warp dựa trên khoảng cách đến tâm.

    Phương pháp:
        1. Tạo mask nhị phân (pixel có dữ liệu = 1)
        2. GaussianBlur với sigma lớn → weight giảm dần ra viền
        3. Kết quả: pixel ở tâm ảnh = trọng số cao, viền = thấp

    Args:
        warped      : Ảnh BGR đã warp lên canvas
        sigma_ratio : Tỉ lệ sigma / kích thước ảnh (0.15 = feathering vừa)

    Returns:
        weight : float32 array [0.0, 1.0], cùng kích thước với warped
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    mask_f = mask.astype(np.float32) / 255.0

    # Kích thước sigma: càng lớn → feathering càng rộng, blend càng mượt
    h, w = warped.shape[:2]
    sigma = max(h, w) * sigma_ratio

    # GaussianBlur mask → weight decay từ tâm ra viền
    weight = cv2.GaussianBlur(mask_f, (0, 0), sigma)

    # Đảm bảo vùng không có pixel = 0
    weight *= mask_f
    return weight


# ==============================================================================
def stitch_images(
    images: list,
    canvas_shape: tuple,
    homographies: dict,
    sigma_ratio: float = 0.15
) -> np.ndarray:
    """
    Ghép ảnh panorama với Gaussian Feathering Blending.

    Thay vì ghép đè (overwrite), tích lũy:
        canvas_acc[y,x,c] += pixel[c] × weight[y,x]
        weight_acc[y,x]   += weight[y,x]
    Rồi normalize:
        result[y,x,c] = canvas_acc[y,x,c] / weight_acc[y,x]

    Args:
        images       : List ảnh BGR đã tiền xử lý (CLAHE + undistort)
        canvas_shape : (height, width) kích thước canvas
        homographies : Dict {img_idx → H_canvas} ma trận chiếu lên canvas
        sigma_ratio  : Feathering strength (0.1=nhẹ, 0.15=vừa, 0.25=mạnh)

    Returns:
        result : Ảnh panorama BGR đã blended, dtype uint8
    """
    t_start = time.time()

    h_canvas, w_canvas = canvas_shape[:2]

    # Accumulator: float32 để tránh overflow khi cộng nhiều ảnh
    canvas_acc  = np.zeros((h_canvas, w_canvas, 3), dtype=np.float64)
    weight_acc  = np.zeros((h_canvas, w_canvas),    dtype=np.float64)

    print("[*] Đang tiến hành Blending (Gaussian Feathering)...")

    for i in range(len(images)):
        if i not in homographies or images[i] is None:
            continue

        # (5) Projective Warping từ module warping.py
        warped = warp_to_canvas(images[i], homographies[i], canvas_shape)

        # Tính weight map — feathering decay từ tâm ảnh ra viền
        weight = compute_weight_map(warped, sigma_ratio=sigma_ratio)   # [H,W] float32

        # Tích lũy weighted sum
        weight_3c = weight[:, :, np.newaxis]                           # [H,W,1]
        canvas_acc  += warped.astype(np.float64) * weight_3c
        weight_acc  += weight.astype(np.float64)

        print(f"   [+] Ảnh {i}: Đã blend với Gaussian feathering (sigma_ratio={sigma_ratio})")

    # Normalize: chia cho tổng trọng số (tránh chia 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        result_f = np.where(
            weight_acc[:, :, np.newaxis] > 1e-6,
            canvas_acc / weight_acc[:, :, np.newaxis],
            0.0
        )

    result = np.clip(result_f, 0, 255).astype(np.uint8)

    t_end = time.time()
    pano_logger.log_blending(t_end - t_start)
    print(f"[+] Blending hoàn tất trong {(t_end-t_start)*1000:.1f}ms")

    return result