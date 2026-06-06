import cv2
import numpy as np

def preprocess_images(images):
    print("[*] Đang thực hiện chế độ Lai ghép AI (Hybrid AI Core)")
    
    display_images = []
    
    for idx, img in enumerate(images):
        if img is None or img.size == 0:
            display_images.append(None)
            continue
            
        # ---------- (1.1) TIỀN XỬ LÝ NHÁNH HIỂN THỊ (ĐỒ HOẠ) ---------- 
        # CLAHE (Contrast Limited Adaptive Histogram Equalization) nguyên bản
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        display_images.append(enhanced_img)
        print(f"  [+] Ảnh {idx}: Đã nâng cấp tương phản mảng Đồ họa thành công.")
        
    # Nhánh AI sẽ được thu nhỏ (Resize) trực tiếp bên trong matching.py theo nhu cầu Batch
    return display_images