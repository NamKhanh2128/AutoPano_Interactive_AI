import cv2
import os
import numpy as np

print("[*] Đang khởi động hệ thống và nạp thư viện Trí Tuệ Nhân Tạo (PyTorch)...")
print("    (Quá trình này có thể mất 10-20 giây trong lần chạy đầu tiên. Vui lòng không tắt chương trình!)")

# Import các components từ pipeline Lai AI (Hybrid AI) mới
from features import preprocess_images
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
    
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images = [cv2.imread(p) for p in image_paths]
    images = [i for i in images if i is not None]
    
    print(f"[+] Đã load thành công {len(images)} ảnh đầu vào.")
    return images

def main():
    print("=== AUTO PANO AI: CHẾ ĐỘ LAI GHÉP AI (HYBRID AI) ===")
    pano_logger.start_timer("total_runtime")
    
    # Bước 1: Load ảnh
    images = load_images(INPUT_FOLDER)
    if len(images) < 2:
        print("[!] Lỗi: Cần tối thiểu 2 bức ảnh có điểm giao nhau để ghép móc. Thoát chương trình.")
        return

    # Bước 2: Tiền xử lý (CLAHE) phục vụ nhánh Đồ họa (Để in ảnh sắc nét)
    display_images = preprocess_images(images)

    # Bước 3 & 4: Dùng mạng NEURAL NETWORK (LoFTR) làm Feature Matching & Lọc nhiễu MAGSAC++
    # Trực tiếp đút ảnh thô vào Model để lấy toạ độ điểm khâu và ma trận
    anchor_idx, match_matrix, H_matrix = identify_anchor_image(images)
    
    if np.sum(match_matrix) == 0:
         print("[!] Lỗi: AI không thể tìm ra điểm giao nhau hợp lý. Vui lòng thử bộ ảnh khác.")
         return

    # Bước 5: Chuyển đổi Perspective Warping & Cân chỉnh Canvas (Ma trận chuỗi BFS)
    canvas_shape, homographies, T = calculate_canvas_size(display_images, anchor_idx, match_matrix, H_matrix)

    # Bước 6: Thành phẩm (Multi-band Laplacian/Feathering Blending & auto tỉa viền đen) trên bộ ảnh đã tăng cường màu
    result_img = stitch_images(display_images, canvas_shape, homographies)

    # Bước 6.5: Tăng cường độ nét (Sharpening / Unsharp Mask)
    print("[*] Đang áp dụng bộ lọc Tăng cường độ nét (Unsharp Mask) cho bản in...")
    gaussian = cv2.GaussianBlur(result_img, (0, 0), 2.0)
    result_img = cv2.addWeighted(result_img, 1.3, gaussian, -0.3, 0)

    # Lưu kết quả
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    cv2.imwrite(OUTPUT_FILE, result_img)
    
    pano_logger.stop_timer("total_runtime")
    
    # Bước 7: Xuất dữ liệu đo lường phục vụ Report
    pano_logger.save_report()

    print("="*40)
    print(f"[THÀNH CÔNG] Ảnh panorama kết quả dòng HYBRID AI đã lưu tại: {OUTPUT_FILE}")
    print(f"[BÁO CÁO] Thông tin chạy pipeline đã xuất tại thư mục: data/logs/report.txt")
    print("="*40)

if __name__ == "__main__":
    main()