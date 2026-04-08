import cv2
import os
import numpy as np

# Import các components từ pipeline Truyền thống Classical mới
from features import extract_features
from matching import identify_anchor_image
from warping import calculate_canvas_size
from blending import stitch_images
from logger import pano_logger

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FOLDER = os.path.join(BASE_DIR, "data", "input")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "output", "panorama_result.jpg")

def load_images(folder_path):
    print(f"[*] Đang load ảnh từ: {folder_path}")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return []
    
    # Ở phương pháp thuần tuý, ta không bẻ cong ảnh bằng cấu trúc tuỳ chỉnh ở khâu load
    # vì ta sẽ dùng Projective Warping trực tiếp (cv2.warpPerspective)
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images = [cv2.imread(p) for p in image_paths]
    images = [i for i in images if i is not None]
    
    print(f"[+] Đã load thành công {len(images)} ảnh đầu vào.")
    return images

def main():
    print("=== AUTO PANO AI: CHẾ ĐỘ TRUYỀN THỐNG (CLASSICAL CV) ===")
    pano_logger.start_timer("total_runtime")
    
    # Bước 1: Load ảnh
    images = load_images(INPUT_FOLDER)
    if len(images) < 2:
        print("[!] Lỗi: Cần tối thiểu 2 bức ảnh có điểm giao nhau để ghép móc. Thoát chương trình.")
        return

    # Bước 2 & 3: Tiền xử lý ánh sáng (CLAHE) + Trích xuất đặc trưng (SIFT/ORB)
    all_kps, all_des = extract_features(images)

    # Bước 4: Ghép cặp (FLANN + Lowe's Ratio) & Lọc nhiễu RANSAC Homography
    anchor_idx, match_matrix = identify_anchor_image(images, all_des, all_kps)
    
    if np.sum(match_matrix) == 0:
         print("[!] Lỗi: Các bức ảnh không có điểm chồng lấp đủ lớn hoặc bị nhiễu. Vui lòng thử bộ ảnh khác.")
         return

    # Bước 5: Chuyển đổi Perspective Warping & Cân chỉnh Canvas (Ma trận)
    canvas_shape, homographies, T = calculate_canvas_size(images, anchor_idx, all_kps, all_des)

    # Bước 6: Thành phẩm (Multi-band Laplacian/Feathering Blending & auto tỉa viền đen)
    result_img = stitch_images(images, canvas_shape, homographies)

    # Bước 6.5: Tăng cường độ nét (Sharpening / Unsharp Mask)
    print("[*] Đang áp dụng bộ lọc Tăng cường độ nét (Unsharp Mask) cho bản in...")
    gaussian = cv2.GaussianBlur(result_img, (0, 0), 2.0)
    # Lấy 1.3 lần ảnh gốc trừ đi 0.3 lần ảnh mờ để tạo hiệu ứng nổi khối (nét gai)
    result_img = cv2.addWeighted(result_img, 1.3, gaussian, -0.3, 0)

    # Lưu kết quả
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    cv2.imwrite(OUTPUT_FILE, result_img)
    
    pano_logger.stop_timer("total_runtime")
    
    # Bước 7: Xuất dữ liệu đo lường phục vụ Report
    pano_logger.save_report()

    print("="*40)
    print(f"[THÀNH CÔNG] Ảnh panorama kết quả đã được lưu tại: {OUTPUT_FILE}")
    print(f"[BÁO CÁO] Thông tin chạy pipeline Truyền thống đã xuất tại thư mục: data/logs/report.txt")
    print("        Mở file này lên và copy các thông số Inliers, Time, Keypoints vào thẳng Word/Report.")
    print("="*40)
    print("[HƯỚNG DẪN XEM 360 ĐỘ TRỰC QUAN]:")
    print("1. Mở Terminal mới, gõ lệnh chuyển thư mục: cd web_viewer")
    print("2. Gõ lệnh bật server: python -m http.server 8000")
    print("3. Vào trình duyệt truy cập: http://localhost:8000")

if __name__ == "__main__":
    main()