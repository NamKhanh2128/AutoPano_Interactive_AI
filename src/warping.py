import cv2
import numpy as np

def cylindrical_warp(img, focal_length=None):
    """Bẻ cong ảnh phẳng thành hình trụ."""
    if focal_length is None:
        focal_length = img.shape[1]  # Mặc định tiêu cự bằng chiều rộng ảnh
        
    h, w = img.shape[:2]
    cylinder = np.zeros_like(img)
    xc, yc = w // 2, h // 2
    
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Toán học chiếu hình trụ
    theta = (x - xc) / focal_length
    h_cyl = (y - yc) / focal_length
    
    X = np.sin(theta)
    Y = h_cyl
    Z = np.cos(theta)
    
    x_img = (focal_length * X / Z) + xc
    y_img = (focal_length * Y / Z) + yc
    
    valid_coords = (x_img >= 0) & (x_img < w) & (y_img >= 0) & (y_img < h) & (Z > 0)
    
    cylinder[y[valid_coords], x[valid_coords]] = img[y_img[valid_coords].astype(int), x_img[valid_coords].astype(int)]
    
    # Cắt bỏ viền đen do bẻ cong
    gray = cv2.cvtColor(cylinder, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(thresh)
    
    return cylinder[y_rect:y_rect+h_rect, x_rect:x_rect+w_rect]