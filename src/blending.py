import cv2
import numpy as np
import time
from logger import pano_logger
from warping import warp_to_canvas

# ---------- (6) THÀNH PHẨM (BLENDING) ---------- 
# Sự kết hợp giữa Seam-finding cơ bản (dựa vào khoảng cách tới tâm / mask intersection)
# và Multi-band Blending / Mượt viền (Feathering).

def stitch_images(images, canvas_shape, homographies):
    t_start = time.time()
    
    # Khởi tạo Canvas rỗng
    canvas = np.zeros((*canvas_shape, 3), dtype=np.uint8)
    canvas_mask = np.zeros(canvas_shape, dtype=np.float32)
    
    print("[*] Đang tiến hành Thành phẩm (Blending) các mảng ảnh lại với nhau...")
    
    for i in range(len(images)):
        if i not in homographies:
            continue
            
        # (5) PROJECTIVE WARPING (từ module trước)
        warped = warp_to_canvas(images[i], homographies[i], canvas_shape)
        
        # (6.1) SEAM FINDING (Đường nối)
        # Sử dụng mặt nạ khoảng cách (Distance Transform) để tạo đường nối mềm
        # Ảnh ở tâm điểm sẽ có trọng số cao, biên (rìa) ảnh có trọng số thấp
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        # Chuẩn hoá
        if np.max(dist_transform) > 0:
            dist_transform = dist_transform / np.max(dist_transform)
            
        current_mask = dist_transform
        
        # Tính vùng chồng lấp (Intersection)
        intersection = (canvas_mask > 0) & (current_mask > 0)
        
        if np.any(intersection):
            # (6.2) XỬ LÝ "MỜ NGHIÊM TRỌNG" (GHOSTING)
            # Lỗi mờ trước đó là do dùng Alpha Blending nhạt qua toàn bộ 100% phần giao nhau.
            # Giải pháp Classical MVP: Tìm đường phân chia (Voronoi Seam) và chỉ làm mềm quanh đường đó.
            
            # Ai có khoảng cách tới viền an toàn hơn (lớn hơn) thì chiếm픽sel đó
            seam_mask = np.zeros(canvas_shape[:2], dtype=np.float32)
            seam_mask[current_mask > canvas_mask] = 1.0
            
            # Làm mềm quanh ranh giới mí ảnh để không bị cắt gắt (Narrow Band Feathering)
            # Với ảnh lớn, dải mờ kích thước 31x31 pixel (hoặc lớn hơn) giúp mắt không nhận ra đường gập ngang
            seam_smooth = cv2.GaussianBlur(seam_mask, (51, 51), 25)
            
            alpha_w = np.repeat(seam_smooth[:, :, np.newaxis], 3, axis=2)
            alpha_c = 1.0 - alpha_w
            
            # Alpha Blending thu hẹp xung quanh mạch ghép (Seam)
            blended = (warped.astype(np.float32) * alpha_w) + (canvas.astype(np.float32) * alpha_c)
            
            # Gán vùng chồng lấp
            canvas[intersection] = np.clip(blended[intersection], 0, 255).astype(np.uint8)
            
            # Những vùng đất chỉ thuộc riêng về ảnh mới
            only_new = (current_mask > 0) & (~intersection)
            canvas[only_new] = warped[only_new]
            
            canvas_mask = np.maximum(canvas_mask, current_mask)
        else:
            # Ghi trực tiếp nếu không bị đụng hàng với ảnh khác
            mask_bool = current_mask > 0
            canvas[mask_bool] = warped[mask_bool]
            canvas_mask = np.maximum(canvas_mask, current_mask)
            
    # ------ (6.3) TỰ ĐỘNG CẮT TỈA (AUTO-CROP) VIỀN ĐEN ------
    print("[*] Tự động cắt tỉa viền đen...")
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(thresh)
    
    t_end = time.time()
    pano_logger.log_blending(t_end - t_start)
    
    return canvas[y:y+h, x:x+w]