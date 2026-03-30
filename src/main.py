import cv2
import os
import numpy as np  # Thêm import numpy

# Import các hàm từ các file trong thư mục src
from features import extract_features
from warping import cylindrical_warp
from matching import identify_anchor_image
from blending import stitch_to_anchor, crop_black_borders

# Xây dựng đường dẫn tuyệt đối, chống lỗi khi chạy từ các thư mục khác nhau
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FOLDER = os.path.join(BASE_DIR, "data", "input")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "output", "panorama_result.jpg")

def load_and_preprocess_images(folder_path):
    print(f"[*] Đang load ảnh từ: {folder_path}")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return []
        
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images = []
    
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            # Bẻ cong trụ ngay từ lúc load ảnh vào
            img_warped = cylindrical_warp(img)
            images.append(img_warped)
            
    print(f"[+] Đã load và bẻ cong trụ {len(images)} ảnh.")
    return images

if __name__ == "__main__":
    print("=== KHỞI ĐỘNG AUTOPANO AI ===")
    
    # 1. Load ảnh
    images = load_and_preprocess_images(INPUT_FOLDER)
    if len(images) < 2:
        print("[!] Vui lòng chép ít nhất 2 bức ảnh có chồng lấp vào thư mục 'data/input' và chạy lại.")
        exit()

    # 2. Tìm điểm đặc trưng
    all_kps, all_des = extract_features(images)

    # 3. AI tìm ảnh trung tâm
    anchor_idx, match_matrix = identify_anchor_image(images, all_des, all_kps)  # Thêm all_kps
    if np.sum(match_matrix) < 50:  # Kiểm tra tổng matches
        print("[!] Không đủ matches tốt. Kiểm tra lại ảnh input (cần chồng lấp rõ ràng).")
        exit()

    # 4. Ghép nối hình ảnh
    stitched_canvas = stitch_to_anchor(images, anchor_idx, all_kps, all_des)

    # 5. Cắt viền đen
    final_img = crop_black_borders(stitched_canvas)

    # 6. Lưu kết quả
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    cv2.imwrite(OUTPUT_FILE, final_img)
    
    print("="*30)
    print(f"[THÀNH CÔNG] Đã lưu kết quả tại: {OUTPUT_FILE}")
    print("[HƯỚNG DẪN XEM 360 ĐỘ]:")
    print("1. Mở Terminal mới, gõ: cd web_viewer")
    print("2. Gõ tiếp lệnh: python -m http.server 8000")
    print("3. Mở trình duyệt truy cập: http://localhost:8000")