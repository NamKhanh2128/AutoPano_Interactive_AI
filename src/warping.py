"""
warping.py — Projective Warping & Canvas Management cho AutoPano_Interactive_AI

Cải tiến so với phiên bản cũ:
  1. auto_crop_canvas() — Tự động cắt viền đen sau khi ghép panorama
  2. warp_cylindrical()  — Chiếu ảnh lên mặt trụ (giảm méo "bow-tie" panorama dài)
  3. warp_to_canvas()    — Giữ nguyên (INTER_CUBIC, chất lượng cao)
  4. calculate_canvas_size() — BFS chain warping (giữ nguyên logic, cải thiện print)
"""

import cv2
import numpy as np
from typing import Tuple, Optional


# ==============================================================================
def warp_to_canvas(
    img: np.ndarray,
    H_canvas: np.ndarray,
    canvas_shape: tuple
) -> np.ndarray:
    """
    Biến đổi ảnh (Warping) vào không gian canvas dựa trên ma trận Homography.

    Dùng INTER_CUBIC: nội suy bicubic — cân bằng tốt giữa tốc độ và chất lượng
    (LANCZOS4 sắc nét hơn nhưng chậm hơn 4× khi canvas lớn).

    Args:
        img          : Ảnh BGR đầu vào
        H_canvas     : Ma trận homography 3×3 (ảnh → canvas)
        canvas_shape : (height, width[, channels])

    Returns:
        warped_img : Ảnh đã warp lên canvas, cùng kích thước canvas_shape[:2]
    """
    h_canvas, w_canvas = canvas_shape[:2]
    warped_img = cv2.warpPerspective(
        img, H_canvas, (w_canvas, h_canvas),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    return warped_img


# ==============================================================================
def warp_cylindrical(
    img: np.ndarray,
    K: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chiếu ảnh lên mặt trụ (Cylindrical Projection).

    Mục đích:
        Panorama từ nhiều ảnh chụp xoay ngang (rotation panorama) sẽ mượt
        hơn nhiều nếu chiếu lên hình trụ trước khi ghép. Hình trụ loại bỏ
        méo "bow-tie" (ảnh hai đầu bị kéo rộng) khi có nhiều ảnh ghép nối.

    Công thức chiếu trụ:
        x_c = f × arctan((x - cx) / f)
        y_c = f × (y - cy) / √((x-cx)² + f²)
    Trong đó f = focal length từ ma trận K.

    Args:
        img : Ảnh BGR đầu vào
        K   : Ma trận nội tại (Intrinsic Matrix) 3×3

    Returns:
        dst   : Ảnh đã chiếu lên mặt trụ
        K_cyl : Ma trận nội tại mới sau chiếu trụ (để dùng cho các bước sau)
    """
    h, w = img.shape[:2]
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Tạo lưới tọa độ trụ
    x_cyl = np.arange(w, dtype=np.float32) - cx
    y_cyl = np.arange(h, dtype=np.float32) - cy

    # Chiếu ngược từ tọa độ trụ → tọa độ ảnh phẳng (inverse mapping cho cv2.remap)
    theta = x_cyl / fx                            # góc ngang
    X = np.tan(theta)                             # X = tan(θ)

    x_src = fx * X + cx                           # x trên ảnh gốc
    r_vec = np.sqrt(X**2 + 1.0)                   # ||[X, 0, 1]||

    # Grid 2D
    map_x = np.tile(x_src, (h, 1)).astype(np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    for row in range(h):
        Y = (row - cy) / fy
        y_src = fy * Y / r_vec + cy
        map_y[row, :] = y_src.astype(np.float32)

    dst = cv2.remap(img, map_x, map_y, cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # K_cyl: giữ nguyên fy, cx, cy — fx thực tế vẫn là f của trụ
    K_cyl = K.copy()

    return dst, K_cyl


# ==============================================================================
def auto_crop_canvas(canvas: np.ndarray) -> np.ndarray:
    """
    Tự động cắt bỏ viền đen (zero-pixel) xung quanh sau khi ghép panorama.

    Phương pháp:
        1. Chuyển sang grayscale
        2. Threshold để tách vùng có dữ liệu (pixel > 0)
        3. findContours → bounding rect của vùng lớn nhất
        4. Cắt ảnh theo bounding rect

    Xử lý edge case:
        - Nếu boundingRect quá nhỏ (< 10% diện tích) → trả về ảnh gốc (tránh crop lỗi)
        - Thêm padding nhỏ (2px) để tránh cắt mất edge pixel

    Args:
        canvas : Ảnh panorama BGR (có thể có viền đen)

    Returns:
        Ảnh đã crop sạch viền đen
    """
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Tìm bounding box của toàn bộ vùng sáng
    # Dùng findNonZero thay vì findContours để nhanh hơn
    coords = cv2.findNonZero(thresh)
    if coords is None:
        print("[!] auto_crop: Canvas rỗng — trả về ảnh gốc.")
        return canvas

    x, y, rw, rh = cv2.boundingRect(coords)

    h_full, w_full = canvas.shape[:2]

    # Kiểm tra sanity: bounding rect phải chiếm ít nhất 5% diện tích canvas
    area_ratio = (rw * rh) / (w_full * h_full)
    if area_ratio < 0.05:
        print(f"[!] auto_crop: Bounding rect quá nhỏ ({area_ratio:.1%}) — bỏ qua crop.")
        return canvas

    # Padding nhỏ để không cắt sát edge
    pad = 2
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w_full, x + rw + pad)
    y2 = min(h_full, y + rh + pad)

    cropped = canvas[y1:y2, x1:x2]
    print(f"[+] Auto crop: {w_full}×{h_full} → {x2-x1}×{y2-y1} "
          f"(cắt {pad}px padding, coverage={area_ratio:.1%})")
    return cropped


# ==============================================================================
def calculate_canvas_size(
    images: list,
    anchor_idx: int,
    match_matrix: np.ndarray,
    H_matrix: dict
) -> Tuple[tuple, dict, np.ndarray]:
    """
    Thuật toán BFS (Breadth-First Search) lan truyền ma trận Homography.

    Nguyên lý:
        Ảnh anchor (trung tâm) chỉ cần dịch chuyển vào giữa canvas (ma trận T).
        Các ảnh xung quanh lan truyền qua chuỗi:
            H_global[neighbor] = H_global[curr] × H[neighbor→curr]
        Cho phép ghép n ảnh (4-5+) theo dây chuyền thay vì mọi ảnh kết nối thẳng với anchor.

    Canvas size:
        - Chiều ngang: n × w_anchor (đủ rộng cho panorama ngang)
        - Chiều dọc: 3 × h_anchor (đủ cho cả góc nghiêng và vertical parallax)

    Args:
        images       : List ảnh BGR (display images sau CLAHE)
        anchor_idx   : Index ảnh làm gốc (được chọn bởi identify_anchor_image)
        match_matrix : Ma trận điểm ghép n×n
        H_matrix     : Dict {(i,j) → H_ij} ma trận homography các cặp

    Returns:
        canvas_shape : (height, width) canvas
        homographies : Dict {img_idx → H_canvas} ma trận chiếu từng ảnh lên canvas
        T            : Ma trận dịch chuyển offset của anchor vào canvas
    """
    anchor_img = images[anchor_idx]
    h, w = anchor_img.shape[:2]
    n = len(images)

    # Canvas đủ rộng
    canvas_w = w * n
    canvas_h = h * 3

    # Ma trận dịch chuyển đặt anchor vào giữa canvas
    offset_x = (canvas_w - w) // 2
    offset_y = (canvas_h - h) // 2
    T = np.array(
        [[1, 0, offset_x],
         [0, 1, offset_y],
         [0, 0, 1]],
        dtype=np.float64
    )

    homographies = {anchor_idx: T}
    visited = {anchor_idx}
    queue = [anchor_idx]

    print("[*] Đang tính toán BFS Projective Warping chain...")

    while queue:
        curr = queue.pop(0)
        for neighbor in range(n):
            if neighbor not in visited and match_matrix[curr, neighbor] > 0:
                H_neighbor_to_curr = H_matrix[(neighbor, curr)]
                H_global = homographies[curr] @ H_neighbor_to_curr
                homographies[neighbor] = H_global
                visited.add(neighbor)
                queue.append(neighbor)
                print(f"   [+] Chuỗi: Ảnh {neighbor} → Ảnh {curr} → Canvas")

    unconnected = [i for i in range(n) if i not in homographies]
    if unconnected:
        print(f"   [!] Cảnh báo: Ảnh {unconnected} không kết nối được vào panorama chain.")

    return (canvas_h, canvas_w), homographies, T