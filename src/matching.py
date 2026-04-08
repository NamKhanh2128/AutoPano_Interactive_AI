import cv2
import numpy as np
import time
from logger import pano_logger

def get_matches_for_pair(idx1, idx2, all_kps, all_des):
    t_start = time.time()
    
    des1 = all_des[idx1]
    des2 = all_des[idx2]
    kp1 = all_kps[idx1]
    kp2 = all_kps[idx2]

    # Kiểm tra điều kiện đầu vào an toàn
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None, None, None

    # Do SIFT trả về float32, đề phòng trường hợp fallback sang ORB (uint8), ta ép kiểu
    if des1.dtype != np.float32: des1 = np.float32(des1)
    if des2.dtype != np.float32: des2 = np.float32(des2)

    # ---------- (3) GHÉP CẶP ĐẶC TRƯNG ----------
    # Khởi tạo thuật toán FLANN (Fast Library for Approximate Nearest Neighbors)
    # Index_params 1: KDTree, search_params=50: Số lần duyệt cây tìm kiếm
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Sử dụng KNN (k=2) lấy 2 hàng xóm gần nhất
    matches = flann.knnMatch(des1, des2, k=2)

    # Lọc điểm nối bằng Lowe's Ratio Test (Ngưỡng 0.7)
    good_matches = []
    for m in matches:
        if len(m) == 2:
            m1, m2 = m
            if m1.distance < 0.7 * m2.distance:
                good_matches.append(m1)
    # ---------------------------------------------
    
    t_match = time.time() - t_start
    pano_logger.log_matching(f"{idx1}-{idx2}", len(matches), len(good_matches), t_match)

    if len(good_matches) < 20: 
        return None, None, None

    # ---------- (4) LỌC NHIỄU & MA TRẬN ----------
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # RANSAC tìm ma trận chiếu Homography H
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    inliers = 0
    inlier_ratio = 0.0
    if mask is not None:
        inliers = np.sum(mask)
        inlier_ratio = inliers / len(good_matches)
        
    pano_logger.log_homography(f"{idx1}-{idx2}", int(inliers), inlier_ratio)
    # ---------------------------------------------
    
    if inlier_ratio < 0.4:
        return None, None, None

    return H, good_matches, inlier_ratio

def identify_anchor_image(images, all_des, all_kps):
    """
    Xác định ảnh làm gốc (Anchor Image) với nhiều cặp Inliers nhất.
    """
    n = len(images)
    print("[*] Đang xây dựng đồ thị ghép nối (FLANN + RANSAC)...")
    match_matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            H, good, ratio = get_matches_for_pair(i, j, all_kps, all_des)
            if H is not None:
                score = len(good) * ratio
                match_matrix[i, j] = score
                match_matrix[j, i] = score
                print(f"   [+] Cặp {i}↔{j}: Số điểm tốt={len(good)}, Tỷ lệ inlier={ratio:.3f}, Điểm={score:.1f}")

    scores_per_image = np.sum(match_matrix, axis=1)
    anchor_idx = int(np.argmax(scores_per_image))

    print(f"[+] Ảnh trung tâm (Anchor) được chọn là ảnh thứ: {anchor_idx} (Tổng số điểm: {scores_per_image[anchor_idx]:.2f})")
    
    return anchor_idx, match_matrix