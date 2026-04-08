import cv2
import numpy as np
import time
from logger import pano_logger

def extract_features(images):
    print("[*] Đang trích xuất đặc trưng (Chuẩn Classical OpenCV)...")
    
    # Khởi tạo SIFT (Chính xác cao) và ORB (Dự phòng)
    # nfeatures=2000 giới hạn để chạy nhanh vừa đủ
    sift = cv2.SIFT_create(nfeatures=2000)
    orb = cv2.ORB_create(nfeatures=2000)
    
    all_kps = []
    all_des = []
    
    for idx, img in enumerate(images):
        if img is None or img.size == 0:
            all_kps.append(None)
            all_des.append(None)
            continue
            
        t_start = time.time()
        
        # ---------- (1) TIỀN XỬ LÝ: HÚT SÁNG ---------- 
        # Chuyển ảnh sang không gian LAB, tách riêng kênh sáng L
        # Dùng CLAHE (Contrast Limited Adaptive Histogram Equalization) để cân bằng tốt hơn
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Gộp kênh sáng đã xử lý vào để tăng nét cho Keypoints
        limg = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
        # ----------------------------------------------
        
        # ---------- (2) TRÍCH XUẤT ĐẶC TRƯNG ----------
        method = "SIFT"
        # Luôn chạy SIFT đầu tiên (vì ổn định và đặc trưng góc quay/tỷ lệ tốt hơn)
        kp, des = sift.detectAndCompute(gray, None)
        
        # Fallback lại bằng ORB nếu SIFT bị hỏng góc nhìn
        if kp is None or len(kp) < 100:
            method = "ORB"
            print(f"[!] SIFT thu được ít điểm (kp={len(kp) if kp else 0}) đối với ảnh {idx}. Chuyển sang ORB.")
            kp, des = orb.detectAndCompute(gray, None)
        # ----------------------------------------------
        
        t_end = time.time()
        
        all_kps.append(kp)
        all_des.append(des)
        
        # Log lại dữ liệu vào report
        pano_logger.log_feature_extraction(idx, method, len(kp) if kp else 0, t_end - t_start)
        print(f"  [+] Ảnh {idx} | {method} | KQ: {len(kp) if kp else 0} điểm | Shape: {des.shape if des is not None else 'None'}")
        
    return all_kps, all_des