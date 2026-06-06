"""
calibration.py — Module Calibration & Undistortion cho AutoPano_Interactive_AI

Nhiệm vụ:
    Nạp thông số camera đã calibrate (K, D) từ file .npz
    (được xuất từ dự án Calibration-ZhangZhengyou-Method)
    và undistort batch ảnh đầu vào trước khi đưa vào pipeline ghép panorama.

Lý thuyết:
    Ảnh từ camera thực tế luôn bị méo ống kính (lens distortion):
      - Radial distortion  : x_d = x(1 + k1·r² + k2·r⁴ + k3·r⁶)
      - Tangential distortion: x_d += 2p1·xy + p2(r² + 2x²)

    Nếu không undistort trước:
      - Các đường thẳng trong thực tế bị cong trên ảnh
      - LoFTR vẫn tìm được keypoints nhưng homography tích lũy sai số
      - BFS warping chain: lỗi H_AB × H_BC × ... nhân dần → artifact nặng

    Sau khi undistort với K_new (getOptimalNewCameraMatrix):
      - Ảnh phẳng hình học hoàn toàn
      - Homography chính xác hơn nhiều
      - Seam line mịn hơn

Pipeline sử dụng:
    1. load_camera_params(calib_file)    → K, D, img_size
    2. build_undistort_maps(...)          → map1, map2, K_new
    3. undistort_batch(images, ...)       → [undistorted images], K_new
    4. apply_remap(img, map1, map2)       → single image undistort
"""

import cv2
import numpy as np
import os
from typing import Tuple, Optional, List


# ==============================================================================
def load_camera_params(calib_file: str) -> Tuple[np.ndarray, np.ndarray, tuple]:
    """
    Nạp thông số camera calibration từ file .npz.
    File này được tạo bởi Calibration-ZhangZhengyou-Method/calibrate.py

    Args:
        calib_file : Đường dẫn đến file .npz (ví dụ: data/calib/camera_params.npz)

    Returns:
        K        : Ma trận nội tại (Intrinsic Matrix) 3×3  [float64]
        D        : Vector hệ số biến dạng [k1,k2,p1,p2,k3] [float64]
        img_size : (width, height) kích thước ảnh lúc calibrate

    Raises:
        FileNotFoundError: Nếu file không tồn tại
    """
    if not os.path.exists(calib_file):
        raise FileNotFoundError(
            f"[Calibration] Không tìm thấy file calibration: {calib_file}\n"
            f"  → Hãy chạy Calibration-ZhangZhengyou-Method/calibrate.py\n"
            f"  → Rồi copy camera_params.npz vào data/calib/"
        )

    data = np.load(calib_file, allow_pickle=True)
    K = data['K']
    D = data['D']
    img_size = tuple(data['img_size'].tolist())   # (width, height)
    rms = float(data['rms'][0])
    method = str(data['method'][0])

    print(f"[Calibration] ✔ Đã nạp thông số camera từ: {calib_file}")
    print(f"              RMS={rms:.4f}px | size={img_size} | model={method}")
    print(f"              fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
    print(f"              D = {D.ravel()}")

    return K, D, img_size


# ==============================================================================
def build_undistort_maps(
    K: np.ndarray,
    D: np.ndarray,
    img_size: Tuple[int, int],
    alpha: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tính toán remapping maps để undistort ảnh nhanh qua cv2.remap().

    Kỹ thuật: Thay vì gọi cv2.undistort() mỗi lần (tính map mới mỗi lần),
    ta tính sẵn map1, map2 một lần → apply O(N) cho toàn bộ batch.

    Công thức:
        K_new = getOptimalNewCameraMatrix(K, D, size, alpha)
        map1, map2 = initUndistortRectifyMap(K, D, None, K_new, size, CV_16SC2)
        img_out = remap(img, map1, map2, INTER_LANCZOS4)

    Args:
        K        : Intrinsic matrix 3×3
        D        : Distortion coefficients
        img_size : (width, height)
        alpha    : 0.0 = cắt sạch viền đen (mất pixel rìa)
                   0.5 = cân bằng ← khuyến nghị cho panorama
                   1.0 = giữ tất cả pixels (có viền đen nhỏ)

    Returns:
        map1, map2 : Remapping arrays (CV_16SC2 — fastest integer maps)
        K_new      : Ma trận nội tại đã điều chỉnh sau undistort
    """
    K_new, roi = cv2.getOptimalNewCameraMatrix(K, D, img_size, alpha, img_size)
    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, None, K_new, img_size, cv2.CV_16SC2
    )
    print(f"[Calibration] Đã tính remapping maps (alpha={alpha})")
    print(f"              K_new: fx={K_new[0,0]:.1f}  fy={K_new[1,1]:.1f}  "
          f"cx={K_new[0,2]:.1f}  cy={K_new[1,2]:.1f}")
    return map1, map2, K_new


# ==============================================================================
def apply_remap(
    img: np.ndarray,
    map1: np.ndarray,
    map2: np.ndarray,
    interpolation: int = cv2.INTER_LANCZOS4
) -> np.ndarray:
    """
    Áp dụng remapping để undistort một ảnh đơn lẻ.

    Dùng BORDER_REPLICATE thay vì BORDER_CONSTANT (màu đen) để tránh
    viền đen làm nhiễu blending ở bước ghép panorama.

    Args:
        img           : Ảnh BGR đầu vào
        map1, map2    : Remapping maps từ build_undistort_maps()
        interpolation : Kiểu nội suy pixel (LANCZOS4 = chất lượng tốt nhất)

    Returns:
        Ảnh đã undistort, cùng kích thước với ảnh đầu vào
    """
    return cv2.remap(
        img, map1, map2, interpolation,
        borderMode=cv2.BORDER_REPLICATE
    )


# ==============================================================================
def get_adjusted_K(
    K: np.ndarray,
    D: np.ndarray,
    img_size: Tuple[int, int],
    alpha: float = 0.5
) -> np.ndarray:
    """
    Trả về K_new (ma trận nội tại sau undistort) mà không cần tính maps.
    Hữu ích cho cylindrical projection và các bước tính toán sau undistort.

    Args:
        K, D, img_size : Camera parameters
        alpha          : Giống build_undistort_maps

    Returns:
        K_new : Ma trận nội tại đã hiệu chỉnh
    """
    K_new, _ = cv2.getOptimalNewCameraMatrix(K, D, img_size, alpha, img_size)
    return K_new


# ==============================================================================
def undistort_batch(
    images: List[np.ndarray],
    calib_file: str,
    alpha: float = 0.5
) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    """
    Undistort toàn bộ danh sách ảnh đầu vào dùng thông số từ file calibration.

    Workflow tối ưu:
        1. Nạp K, D từ .npz
        2. Tính remapping maps 1 lần (từ ảnh đầu tiên)
        3. Apply remap cho từng ảnh — O(n × pixels) thay vì tính lại maps mỗi lần

    Args:
        images     : List ảnh BGR (đã méo)
        calib_file : Đường dẫn file .npz
        alpha      : 0.5 (khuyến nghị cho panorama)

    Returns:
        undistorted_images : List ảnh đã undistort
        K_new              : Ma trận nội tại sau undistort (None nếu lỗi)
    """
    if not os.path.exists(calib_file):
        print(f"[Calibration] ⚠ Không tìm thấy file calibration: {calib_file}")
        print(f"[Calibration]   → Bỏ qua undistort, dùng ảnh gốc (pipeline vẫn chạy).")
        return images, None

    try:
        K, D, calib_img_size = load_camera_params(calib_file)
    except Exception as e:
        print(f"[Calibration] ⚠ Lỗi đọc calibration: {e}")
        return images, None

    # Xác định kích thước ảnh thực tế đầu vào
    valid_imgs = [img for img in images if img is not None and img.size > 0]
    if not valid_imgs:
        return images, None

    h0, w0 = valid_imgs[0].shape[:2]
    actual_size = (w0, h0)

    # Nếu kích thước ảnh khác kích thước calibration → cần scale K
    if actual_size != calib_img_size:
        sx = w0 / calib_img_size[0]
        sy = h0 / calib_img_size[1]
        K_scaled = K.copy()
        K_scaled[0, 0] *= sx   # fx
        K_scaled[1, 1] *= sy   # fy
        K_scaled[0, 2] *= sx   # cx
        K_scaled[1, 2] *= sy   # cy
        print(f"[Calibration] ⚠ Kích thước ảnh ({actual_size}) ≠ calib size ({calib_img_size})")
        print(f"[Calibration]   → Đã scale K theo tỉ lệ ({sx:.3f}, {sy:.3f})")
        K_use = K_scaled
    else:
        K_use = K

    # Tính remapping maps 1 lần
    map1, map2, K_new = build_undistort_maps(K_use, D, actual_size, alpha)

    # Áp dụng cho toàn bộ batch
    undistorted = []
    count_ok = 0
    for i, img in enumerate(images):
        if img is None or img.size == 0:
            undistorted.append(img)
            continue

        if img.shape[1] != w0 or img.shape[0] != h0:
            img = cv2.resize(img, actual_size)

        dst = apply_remap(img, map1, map2, cv2.INTER_LANCZOS4)
        undistorted.append(dst)
        count_ok += 1

    print(f"[Calibration] ✔ Undistort thành công {count_ok}/{len(images)} ảnh "
          f"(alpha={alpha}, LANCZOS4)")

    return undistorted, K_new
