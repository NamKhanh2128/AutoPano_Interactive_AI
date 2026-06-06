import cv2
import os
import numpy as np

print("[*] Đang khởi động hệ thống và nạp thư viện Trí Tuệ Nhân Tạo (PyTorch)...")
print("    (Quá trình này có thể mất 10-20 giây trong lần chạy đầu tiên. Vui lòng không tắt chương trình!)")

from features import preprocess_images
from matching import identify_anchor_image
from warping import calculate_canvas_size, auto_crop_canvas
from blending import stitch_images
from calibration import undistort_batch
from logger import pano_logger

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FOLDER  = os.path.join(BASE_DIR, "data", "input")
OUTPUT_FILE   = os.path.join(BASE_DIR, "data", "output", "panorama_result.jpg")
CALIB_FILE    = os.path.join(BASE_DIR, "data", "calib", "camera_params.npz")


def load_images(folder_path):
    print(f"[*] Đang load ảnh từ: {folder_path}")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return [], []

    image_paths = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    images = [cv2.imread(p) for p in image_paths]
    images = [i for i in images if i is not None]

    print(f"[+] Đã load thành công {len(images)} ảnh đầu vào.")
    return images


def main():
    print("=== AUTO PANO AI: CHẾ ĐỘ LAI GHÉP AI (HYBRID AI + CALIBRATION) ===")
    pano_logger.start_timer("total_runtime")

    # ─── BƯỚC 1: LOAD ẢNH ────────────────────────────────────────────────────
    images = load_images(INPUT_FOLDER)
    if len(images) < 2:
        print("[!] Lỗi: Cần tối thiểu 2 bức ảnh có điểm giao nhau. Thoát.")
        return

    # ─── BƯỚC 1.5: UNDISTORT (KHỬ MÉO ỐNG KÍNH) ─────────────────────────────
    # Dùng thông số từ camera_params.npz (xuất từ Calibration-ZhangZhengyou-Method)
    # Nếu không có file → tự động bỏ qua, pipeline vẫn tiếp tục bình thường
    print("\n[*] Bước 1.5: Khử méo ống kính (Camera Undistortion)...")
    images, K_new = undistort_batch(images, CALIB_FILE, alpha=0.5)

    if K_new is not None:
        print(f"[+] Undistort thành công! K_new đã được cập nhật cho pipeline.")
    else:
        print(f"[~] Không có calibration file → Bỏ qua undistort (pipeline vẫn hoạt động).")
        print(f"    Để bật undistort: copy camera_params.npz vào {CALIB_FILE}")

    # ─── BƯỚC 2: TIỀN XỬ LÝ CLAHE ────────────────────────────────────────────
    display_images = preprocess_images(images)

    # ─── BƯỚC 3 & 4: AI FEATURE MATCHING (LoFTR + MAGSAC++) ──────────────────
    anchor_idx, match_matrix, H_matrix = identify_anchor_image(images)

    if np.sum(match_matrix) == 0:
        print("[!] Lỗi: AI không tìm được điểm ghép. Thử bộ ảnh khác.")
        return

    # ─── BƯỚC 5: PROJECTIVE WARPING (BFS CHAIN) ──────────────────────────────
    canvas_shape, homographies, T = calculate_canvas_size(
        display_images, anchor_idx, match_matrix, H_matrix
    )

    # ─── BƯỚC 6: BLENDING (GAUSSIAN FEATHERING) ──────────────────────────────
    result_img = stitch_images(display_images, canvas_shape, homographies)

    # ─── BƯỚC 6.5: AUTO CROP VIỀN ĐEN ────────────────────────────────────────
    print("[*] Đang cắt bỏ viền đen (Auto Crop)...")
    result_img = auto_crop_canvas(result_img)

    # ─── BƯỚC 6.7: TĂNG CƯỜNG ĐỘ NÉT (UNSHARP MASK) ─────────────────────────
    print("[*] Đang áp dụng Unsharp Mask tăng độ nét...")
    gaussian = cv2.GaussianBlur(result_img, (0, 0), 2.0)
    result_img = cv2.addWeighted(result_img, 1.3, gaussian, -0.3, 0)

    # ─── BƯỚC 7: LƯU KẾT QUẢ ────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    cv2.imwrite(OUTPUT_FILE, result_img, [cv2.IMWRITE_JPEG_QUALITY, 97])

    pano_logger.stop_timer("total_runtime")
    pano_logger.save_report()

    print("=" * 60)
    print(f"[THÀNH CÔNG] Ảnh panorama đã lưu tại: {OUTPUT_FILE}")
    print(f"             Kích thước: {result_img.shape[1]}×{result_img.shape[0]} px")
    print(f"[BÁO CÁO]   Report tại: data/logs/report.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()