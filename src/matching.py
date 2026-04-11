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
    
    # Nâng lên 1024.0 (Độ phân giải lý tưởng nhất của LoFTR) thay vì 640.0
    # Điều này giúp tăng số lượng điểm neo lên gấp bội (2x - 4x Keypoints)
    MAX_SIZE = 1024.0 
    
    scale1 = MAX_SIZE / max(h1, w1)
    scale2 = MAX_SIZE / max(h2, w2)
    
    # Resize chuẩn AI & Bắt buộc quy tròn về bội số của 8 (yêu cầu tối ưu của thuật toán Transformers)
    w1_sm, h1_sm = int(w1*scale1) // 8 * 8, int(h1*scale1) // 8 * 8
    w2_sm, h2_sm = int(w2*scale2) // 8 * 8, int(h2*scale2) // 8 * 8
    
    # Cập nhật lại scale thật sau khi làm tròn để tránh sai số toạ độ
    real_scale1_w, real_scale1_h = w1_sm / w1, h1_sm / h1
    real_scale2_w, real_scale2_h = w2_sm / w2, h2_sm / h2
    
    img1_sm = cv2.resize(img1, (w1_sm, h1_sm))
    img2_sm = cv2.resize(img2, (w2_sm, h2_sm))
    
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

    # Lọc ngưỡng độ tự tin AI (Hạ nhẹ xuống 0.5 thay vì 0.6 để lấy thêm nhiều điểm tốt)
    valid = confidence > 0.5
    mkpts0_sm = mkpts0_sm[valid]
    mkpts1_sm = mkpts1_sm[valid]
    
    # Scale trả lại tọa độ trên ảnh Đồ họa kích thước thật, dùng real_scale độc lập XY
    mkpts0 = np.empty_like(mkpts0_sm)
    mkpts0[:, 0] = mkpts0_sm[:, 0] / real_scale1_w
    mkpts0[:, 1] = mkpts0_sm[:, 1] / real_scale1_h
    
    mkpts1 = np.empty_like(mkpts1_sm)
    mkpts1[:, 0] = mkpts1_sm[:, 0] / real_scale2_w
    mkpts1[:, 1] = mkpts1_sm[:, 1] / real_scale2_h
    
    t_end = time.time()
    print(f"   [AI] LoFTR Core tìm được {len(mkpts0)} điểm ghép giữa 2 ảnh. ({(t_end-t_start)*1000:.1f}ms)")
    
    return mkpts0, mkpts1, t_end - t_start


def draw_loftr_matches(img1, img2, mkpts1, mkpts2, mask, idx1, idx2):
    import os
    # Biến đổi array về cấu trúc KeyPoint mặc định của OpenCV để dùng lệnh hiển thị có sẵn
    kps1 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in mkpts1]
    kps2 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in mkpts2]
    
    # Tạo ánh xạ 1-1 cho các điểm đã tìm thấy
    matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _imgIdx=0, _distance=0) for i in range(len(mkpts1))]
    matchesMask = mask.ravel().tolist() if mask is not None else None
    
    # Xuất màn hình Inliers (xanh lá rực mượt), loại bỏ điểm nhiễu 
    match_img = cv2.drawMatches(img1, kps1, img2, kps2, matches, None, 
                                matchColor=(0, 255, 0), # Màu đường nối thành công
                                singlePointColor=(0, 0, 255), 
                                matchesMask=matchesMask,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                                
    # Lưu file
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(BASE_DIR, "data", "output", "matches_debug")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"AI_matches_{idx1}_{idx2}.jpg")
    cv2.imwrite(out_path, match_img)

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
    
    # Nới lỏng ngưỡng tin cậy: AI sinh ra hàng nghìn điểm nên tỷ lệ inlier có thể thấp, 
    # nhưng nếu số điểm inliers tuyệt đối đủ lớn (>= 30) thì vẫn là một cặp ghép rất tốt.
    if inliers < 30 and inlier_ratio < 0.25:
        return None, None, None

    # NẾU CẶP ẢNH CÓ CHỒNG LẤP ĐẠT TIÊU CHUẨN -> XUẤT ẢNH VISUALIZE CÁC KEYPOINTS ĐỂ CHỨNG MINH 
    draw_loftr_matches(img1, img2, mkpts1, mkpts2, mask, idx1, idx2)

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