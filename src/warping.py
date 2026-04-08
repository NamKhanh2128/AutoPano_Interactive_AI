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

def calculate_canvas_size(images, anchor_idx, match_matrix, H_matrix):
    """
    Sử dụng thuật toán BFS (Breadth-First Search) để nhân dồn ma trận của các bức ảnh liền kề,
    cho phép ghép n ảnh (ví dụ 4-5 ảnh trở lên) theo dây chuyền thay vì ép tất cả kết nối thẳng với Ảnh Tâm.
    """
    anchor_img = images[anchor_idx]
    h, w = anchor_img.shape[:2]
    
    # Dự báo kích thước bạt rộng lên dựa trên số lượng ảnh
    n = len(images)
    canvas_w = w * n
    canvas_h = h * 3
    
    # Ma trận dời gốc toạ độ (T) để nhét cụm ảnh vào giữa bạt ngang
    offset_x = (canvas_w - w) // 2
    offset_y = (canvas_h - h) // 2
    T = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)

    # Dictionary lưu trữ H tuyệt đối của từng mảng (từ mảng đó chiếu thẳng lên nền bạt)
    # Ảnh trung tâm chỉ cần dời toạ độ
    homographies = {anchor_idx: T} 
    
    # --- THUẬT TOÁN BFS LAN TRUYỀN MA TRẬN ---
    visited = {anchor_idx}
    queue = [anchor_idx]
    
    print("[*] Đang tính toán Projective Warping nối dây chuyền không gian...")
    
    while queue:
        curr = queue.pop(0)
        
        # Duyệt xem những bản thu nào xung quanh có độ trùng khớp với bản thu hiện tại
        for neighbor in range(n):
            if neighbor not in visited and match_matrix[curr, neighbor] > 0:
                # Trích xuất ma trận quy đổi góc nhìn từ neighbor -> curr
                H_neighbor_to_curr = H_matrix[(neighbor, curr)]
                
                # Nối chuỗi dây chuyền: Chiếu từ Neighbor qua Curr, rồi từ Curr qua Bạt
                H_global = homographies[curr] @ H_neighbor_to_curr
                homographies[neighbor] = H_global
                
                visited.add(neighbor)
                queue.append(neighbor)
                print(f"   [+] Truyền toạ độ ghép dây chuyền: {neighbor} -> {curr} -> Canvas")

    return (canvas_h, canvas_w), homographies, T