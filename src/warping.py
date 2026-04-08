import cv2
import numpy as np

# ---------- (5) PROJECTIVE WARPING ----------
def warp_to_canvas(img, H_canvas, canvas_shape):
    """
    Biến đổi ảnh (Warping) vào không gian bạt trống (Canvas) 
    dựa trên Ma trận chiếu (Homography).
    """
    h_canvas, w_canvas = canvas_shape[:2]
    # Dùng nội suy CUBIC thay vì LINEAR để giữ viền nét căng, chống mờ vỡ hạt.
    warped_img = cv2.warpPerspective(img, H_canvas, (w_canvas, h_canvas), flags=cv2.INTER_CUBIC)
    
    return warped_img

def calculate_canvas_size(images, anchor_idx, all_kps, all_des):
    """
    Tính toán kích thước bạt (Canvas) để chứa được tất cả các ảnh khi bẻ cong.
    """
    from matching import get_matches_for_pair
    
    anchor_img = images[anchor_idx]
    h, w = anchor_img.shape[:2]
    
    # Ở phiên bản MVP thuần tuý, ta khởi tạo bạt trống to gấp 3 lần ảnh gốc.
    # Ảnh anchor được đặt tại ví trí trung tâm.
    canvas_w = w * 3
    canvas_h = h * 3
    
    # Ma trận dời gốc toạ độ ra giữa bạt (T)
    offset_x = w
    offset_y = h
    T = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)

    # Lưu trữ ma trận Homography từ ảnh đang xét (i) -> Canvas
    homographies = {anchor_idx: T} 

    # So sánh và tính toán ma trận của tất cả các ảnh khác đối với tâm
    print("[*] Đang căn chỉnh Projective Warping cho các mảng...")
    for i in range(len(images)):
        if i == anchor_idx: continue
        
        # Mối quan hệ từ ảnh i tới tâm anchor
        H, _, _ = get_matches_for_pair(i, anchor_idx, all_kps, all_des)
        if H is not None:
            # H_to_canvas = Biến đổi offset tâm (T) x Ma trận góc nhìn (H)
            H_to_canvas = np.dot(T, H)
            homographies[i] = H_to_canvas

    return (canvas_h, canvas_w), homographies, T