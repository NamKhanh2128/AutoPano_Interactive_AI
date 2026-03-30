import cv2
import numpy as np

def identify_anchor_image(images, all_des, all_kps):
    """
    Chọn ảnh anchor (trung tâm) bằng cách:
    - so sánh tất cả cặp ảnh
    - dùng BFMatcher + Lowe ratio
    - kiểm tra homography + inlier ratio
    """
    n = len(images)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    match_matrix = np.zeros((n, n), dtype=np.float32)

    print("[*] Đang xây dựng đồ thị đối sánh chéo để tìm Ảnh Trung Tâm...")

    for i in range(n):
        for j in range(i + 1, n):
            ki = all_kps[i]
            kj = all_kps[j]
            di = all_des[i]
            dj = all_des[j]

            if di is None or dj is None or len(ki) < 10 or len(kj) < 10:
                continue

            knn = bf.knnMatch(di, dj, k=2)
            good = []
            for m, n_match in knn:
                if m.distance < 0.6 * n_match.distance:  # Giảm từ 0.65 xuống 0.6
                    good.append(m)

            if len(good) < 30:  # Tăng từ 25 lên 30
                continue

            src_pts = np.float32([ki[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kj[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0, maxIters=2000, confidence=0.995)

            inlier_ratio = 0.0
            if mask is not None and len(mask) > 0:
                inliers = int(mask.sum())
                inlier_ratio = inliers / len(good)

            score = 0.0
            if H is not None and inlier_ratio >= 0.5:  # Tăng từ 0.45 lên 0.5
                score = len(good) * inlier_ratio
            else:
                score = len(good) * 0.35  # fallback
            match_matrix[i, j] = score
            match_matrix[j, i] = score

            print(f"   [PAIR] {i}↔{j}: good={len(good)}, inlier={inlier_ratio:.3f}, score={score:.1f}")

    scores_per_image = np.sum(match_matrix, axis=1)
    anchor_idx = int(np.argmax(scores_per_image))

    print(f"[+] Bảng điểm per image: {scores_per_image}")
    print(f"[+] Chọn ảnh anchor: {anchor_idx} (max score={scores_per_image[anchor_idx]:.1f})")

    return anchor_idx, match_matrix