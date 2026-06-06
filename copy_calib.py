"""
copy_calib.py — Script helper: Copy camera_params.npz từ Calibration project sang AutoPano

Chạy từ thư mục AutoPano_Interactive_AI:
    python copy_calib.py

Hoặc chỉ định đường dẫn tùy chỉnh:
    python copy_calib.py --src "C:/path/to/camera_params.npz"
"""

import os
import sys
import shutil
import argparse


def main():
    parser = argparse.ArgumentParser(description='Copy calibration params to AutoPano')
    parser.add_argument(
        '--src',
        default=None,
        help='Đường dẫn đến file camera_params.npz (mặc định: tự tìm ở thư mục Calibration bên cạnh)'
    )
    args = parser.parse_args()

    # Thư mục gốc AutoPano
    auto_pano_dir = os.path.dirname(os.path.abspath(__file__))
    dst_file = os.path.join(auto_pano_dir, "data", "calib", "camera_params.npz")

    # Tìm file nguồn
    if args.src:
        src_file = args.src
    else:
        # Tự tìm ở thư mục Calibration-ZhangZhengyou-Method bên cạnh
        parent_dir = os.path.dirname(auto_pano_dir)
        calib_dir  = os.path.join(parent_dir, "Calibration-ZhangZhengyou-Method")
        src_file   = os.path.join(calib_dir, "camera_params.npz")

    print(f"[copy_calib] Nguồn : {src_file}")
    print(f"[copy_calib] Đích  : {dst_file}")

    if not os.path.exists(src_file):
        print(f"\n[!] Không tìm thấy file nguồn: {src_file}")
        print(f"[!] Hãy chạy Calibration-ZhangZhengyou-Method/calibrate.py trước!")
        sys.exit(1)

    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    shutil.copy2(src_file, dst_file)

    # Copy kèm metadata JSON nếu có
    meta_src = os.path.splitext(src_file)[0] + '_meta.json'
    if os.path.exists(meta_src):
        meta_dst = os.path.splitext(dst_file)[0] + '_meta.json'
        shutil.copy2(meta_src, meta_dst)
        print(f"[copy_calib] ✔ Đã copy metadata JSON kèm theo")

    print(f"\n[copy_calib] ✔ Thành công!")
    print(f"[copy_calib]   AutoPano sẽ tự động dùng thông số này ở lần chạy tiếp theo.")

    # Hiển thị thông tin nhanh
    try:
        import numpy as np
        data = np.load(dst_file, allow_pickle=True)
        K = data['K']
        rms = float(data['rms'][0])
        print(f"\n[copy_calib] Camera info:")
        print(f"   RMS = {rms:.4f} px")
        print(f"   fx  = {K[0,0]:.1f}  fy={K[1,1]:.1f}")
        print(f"   cx  = {K[0,2]:.1f}  cy={K[1,2]:.1f}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
