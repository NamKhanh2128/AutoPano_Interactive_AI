import cv2
import numpy as np

def extract_features(images):
    print("[*] Đang trích xuất đặc trưng (SIFT) cho từng ảnh...")
    sift = cv2.SIFT_create()
    orb = cv2.ORB_create()  # Fallback nếu SIFT thất bại
    all_kps = []
    all_des = []
    
    for idx, img in enumerate(images):
        if img is None or img.size == 0:
            print(f"[!] Ảnh {idx} rỗng, bỏ qua.")
            all_kps.append(None)
            all_des.append(None)
            continue
        
        # Resize nếu ảnh quá lớn (tăng tốc mà giữ chất lượng)
        h, w = img.shape[:2]
        if max(h, w) > 2000:
            scale = 0.5
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
            print(f"[+] Resize ảnh {idx} xuống {img.shape[:2]}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Thử SIFT trước
        kp, des = sift.detectAndCompute(gray, None)
        
        # Nếu SIFT thất bại (ít kp), dùng ORB
        if kp is None or len(kp) < 50:
            print(f"[!] SIFT thất bại cho ảnh {idx} (kp={len(kp) if kp else 0}), chuyển sang ORB...")
            kp, des = orb.detectAndCompute(gray, None)
        
        all_kps.append(kp)
        all_des.append(des)
        
        print(f"[+] Ảnh {idx}: {len(kp) if kp else 0} keypoints, descriptors shape={des.shape if des is not None else 'None'}")
        
    return all_kps, all_des