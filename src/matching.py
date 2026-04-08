import cv2
import numpy as np
import time
import torch
import kornia
from kornia.feature import LoFTR
from logger import pano_logger

import ssl
import urllib.request

# Cấu hình bỏ qua xác thực SSL Tắt trên Windows (để tải Model PyTorch không bị chặn)
ssl._create_default_https_context = ssl._create_unverified_context

# Cấu hình tính toán cục bộ/AI
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[AI] Đang tải mô hình mạng nơ-ron LoFTR lên {device} (Lần đầu sẽ mất tí thời gian)...")
try:
    loftr = LoFTR(pretrained='outdoor').to(device)
    loftr.eval()
except Exception as e:
    print(f"[!] Lỗi khi nạp LoFTR: {e}. Vui lòng kiểm tra quyền truy cập mạng.")
    
def compute_loftr_matches(img1, img2):
    t_start = time.time()
    
    # ---------- (2.1) TIỀN XỬ LÝ NHÁNH AI ----------
    # Thu nhỏ ảnh để vừa bộ nhớ (VRAM/RAM) và tăng tốc mạng nơ-ron
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    scale1 = 640.0 / max(h1, w1)
    scale2 = 640.0 / max(h2, w2)
    
    # Resize chuẩn AI
    img1_sm = cv2.resize(img1, (int(w1*scale1), int(h1*scale1)))
    img2_sm = cv2.resize(img2, (int(w2*scale2), int(h2*scale2)))
    
    gray1 = cv2.cvtColor(img1_sm, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_sm, cv2.COLOR_BGR2GRAY)

    # Đưa vào kiểu Tensor
    t_img1 = torch.from_numpy(gray1).unsqueeze(0).unsqueeze(0).float() / 255.0
    t_img2 = torch.from_numpy(gray2).unsqueeze(0).unsqueeze(0).float() / 255.0
    
    t_img1 = t_img1.to(device)
    t_img2 = t_img2.to(device)

    # ---------- (2.2) ĐỐI SÁNH TRỰC TIẾP BẰNG AI ----------
    input_dict = {"image0": t_img1, "image1": t_img2}
    with torch.no_grad():
        correspondences = loftr(input_dict)
        
    mkpts0_sm = correspondences['keypoints0'].cpu().numpy()
    mkpts1_sm = correspondences['keypoints1'].cpu().numpy()
    confidence = correspondences['confidence'].cpu().numpy()

    # Lọc ngưỡng độ tự tin AI
    valid = confidence > 0.6
    mkpts0_sm = mkpts0_sm[valid]
    mkpts1_sm = mkpts1_sm[valid]
    
    # Scale trả lại tọa độ trên ảnh Đồ họa kích thước thật
    mkpts0 = mkpts0_sm / scale1
    mkpts1 = mkpts1_sm / scale2
    
    t_end = time.time()
    print(f"   [AI] LoFTR Core tìm được {len(mkpts0)} điểm ghép giữa 2 ảnh. ({(t_end-t_start)*1000:.1f}ms)")
    
    return mkpts0, mkpts1, t_end - t_start


def get_matches_for_pair(idx1, idx2, images):
    img1 = images[idx1]
    img2 = images[idx2]

    # --- Gọi hàm AI ---
    mkpts1, mkpts2, t_match = compute_loftr_matches(img1, img2)
    pano_logger.log_matching(f"{idx1}-{idx2}", len(mkpts1), len(mkpts1), t_match)

    if len(mkpts1) < 20: 
        return None, None, None

    # ---------- (3) LỌC NHIỄU MAGSAC++ ----------
    src_pts = np.float32(mkpts1).reshape(-1, 1, 2)
    dst_pts = np.float32(mkpts2).reshape(-1, 1, 2)

    # MAGSAC++ tìm ma trận (Robust hơn RANSAC, không cần ngưỡng chia cắt Inlier cứng)
    try:
        # Hỗ trợ CV2 4.5.0+ 
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, 5.0, confidence=0.9999, maxIters=5000)
    except AttributeError:
        # Fallback lại RANSAC nếu ai dùng bản CV2 quá lỗi thời
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    inliers = 0
    inlier_ratio = 0.0
    if mask is not None:
        inliers = np.sum(mask)
        inlier_ratio = inliers / len(mkpts1)
        
    pano_logger.log_homography(f"{idx1}-{idx2}", int(inliers), inlier_ratio)
    
    # Nới lỏng ngưỡng tin cậy vì AI ghép bạo dạn hơn các hàm dò góc
    if inlier_ratio < 0.25:
        return None, None, None

    good_matches = list(range(len(mkpts1)))  
    return H, good_matches, inlier_ratio


def identify_anchor_image(images):
    """
    Xác định ảnh làm gốc (Anchor Image) 
    """
    n = len(images)
    print("[*] Đang xây dựng đồ thị ghép nối bằng Trí Tuệ Nhân Tạo (LoFTR + MAGSAC++)...")
    match_matrix = np.zeros((n, n), dtype=np.float32)
    H_matrix = {}

    for i in range(n):
        for j in range(i + 1, n):
            H, good, ratio = get_matches_for_pair(i, j, images)
            if H is not None:
                score = len(good) * ratio
                match_matrix[i, j] = score
                match_matrix[j, i] = score
                H_matrix[(i, j)] = H
                # Ma trận nghịch đảo cho chiều ngược lại j -> i
                H_matrix[(j, i)] = np.linalg.inv(H)
                print(f"   [+] Cặp {i}↔{j}: Số điểm tốt={len(good)}, Tỷ lệ inlier={ratio:.3f}, Điểm={score:.1f}")

    scores_per_image = np.sum(match_matrix, axis=1)
    anchor_idx = int(np.argmax(scores_per_image))

    print(f"[+] Ảnh trung tâm (Anchor) được chọn là ảnh thứ: {anchor_idx} (Tổng số điểm: {scores_per_image[anchor_idx]:.2f})")
    
    return anchor_idx, match_matrix, H_matrix