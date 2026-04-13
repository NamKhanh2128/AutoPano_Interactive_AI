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
        
        # Ghi trực tiếp ảnh đè lên nhau, bỏ qua khử biên (Feathering/Blending)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        mask_bool = thresh > 0
        canvas[mask_bool] = warped[mask_bool]
            
    t_end = time.time()
    pano_logger.log_blending(t_end - t_start)
    
    return canvas