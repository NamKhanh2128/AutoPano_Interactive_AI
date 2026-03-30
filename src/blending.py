import cv2
import numpy as np

def stitch_to_anchor(images, anchor_idx, all_kps, all_des):
    anchor_img = images[anchor_idx]
    h_anchor, w_anchor = anchor_img.shape[:2]
    
    # Tạo Canvas rộng hơn (gấp 4 lần để an toàn)
    h_canvas, w_canvas = h_anchor * 4, w_anchor * 4
    canvas = np.zeros((h_canvas, w_canvas, 3), dtype=np.uint8)
    
    # Đặt ảnh tâm vào giữa
    offset_x, offset_y = w_anchor * 1.5, h_anchor * 1.5
    canvas[int(offset_y):int(offset_y)+h_anchor, int(offset_x):int(offset_x)+w_anchor] = anchor_img
    
    T = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)
    bf = cv2.BFMatcher()
    
    print("[*] Đang ghép các ảnh xung quanh vào Canvas với blending mượt...")
    stitched_count = 0
    for i in range(len(images)):
        if i == anchor_idx: continue
            
        matches = bf.knnMatch(all_des[i], all_des[anchor_idx], k=2)
        good_matches = [m for m, n_match in matches if m.distance < 0.6 * n_match.distance]  # Đồng bộ với matching
        
        if len(good_matches) < 30:  # Đồng bộ threshold
            print(f"[!] Bỏ qua ảnh {i} do không đủ điểm khớp ({len(good_matches)} < 30).")
            continue
            
        src_pts = np.float32([all_kps[i][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([all_kps[anchor_idx][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        inlier_ratio = np.sum(mask) / len(good_matches) if mask is not None else 0
        if H is None or inlier_ratio < 0.5:  # Đồng bộ threshold
            print(f"[!] Bỏ qua ảnh {i} do homography yếu (inlier={inlier_ratio:.3f}).")
            continue
            
        H_canvas = T @ H 
        warped_img = cv2.warpPerspective(images[i], H_canvas, (w_canvas, h_canvas))
        
        # Blending mượt: Feathering (linear blend ở edges)
        gray_warped = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_warped, 1, 255, cv2.THRESH_BINARY)
        
        # Mở rộng mask để tạo feathering zone
        kernel = np.ones((5, 5), np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=2)
        mask_feather = cv2.GaussianBlur(mask_dilated.astype(np.float32) / 255, (21, 21), 0)
        
        # Blend
        canvas_bg = cv2.bitwise_and(canvas, canvas, mask=(1 - mask_feather).astype(np.uint8))
        warped_fg = cv2.bitwise_and(warped_img, warped_img, mask=mask_feather.astype(np.uint8))
        canvas = cv2.add(canvas_bg, warped_fg)
        
        stitched_count += 1
        print(f"[+] Ghép thành công ảnh {i} (good={len(good_matches)}, inlier={inlier_ratio:.3f}).")
    
    print(f"[+] Tổng ghép {stitched_count} ảnh.")
    return canvas

def crop_black_borders(img):
    print("[*] Đang tự động cắt tỉa viền đen với morphological operations...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Morphological opening để loại bỏ noise
    kernel = np.ones((5, 5), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return img
        
    # Chọn contour lớn nhất, nhưng thêm padding để tránh cắt quá sát
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    padding = 10  # Thêm padding 10px
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1] - x, w + 2 * padding)
    h = min(img.shape[0] - y, h + 2 * padding)
    
    return img[y:y+h, x:x+w]