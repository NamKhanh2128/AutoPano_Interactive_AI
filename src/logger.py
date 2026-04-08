import time
import os
import json
from datetime import datetime

class PanoLogger:
    def __init__(self, log_dir="data/logs"):
        # Lấy thư mục gốc bên ngoài src
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.log_dir = os.path.join(base_dir, log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.metrics = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "features": [],
            "matching": [],
            "homography": [],
            "blending": {}
        }
        self.timers = {}

    def start_timer(self, name):
        self.timers[name] = time.time()

    def stop_timer(self, name):
        if name in self.timers:
            elapsed = time.time() - self.timers[name]
            return elapsed
        return 0

    def log_feature_extraction(self, img_idx, method, kp_count, time_taken):
        self.metrics["features"].append({
            "image_index": img_idx,
            "method": method,
            "keypoints": kp_count,
            "time_ms": round(time_taken * 1000, 2)
        })

    def log_matching(self, pair, total_matches, good_matches, time_taken):
        self.metrics["matching"].append({
            "pair": pair,
            "total_matches": total_matches,
            "good_matches": good_matches,
            "time_ms": round(time_taken * 1000, 2)
        })

    def log_homography(self, pair, inliers, inlier_ratio):
        self.metrics["homography"].append({
            "pair": pair,
            "inliers": inliers,
            "inlier_ratio": round(inlier_ratio, 4)
        })

    def log_blending(self, time_taken):
        self.metrics["blending"]["time_ms"] = round(time_taken * 1000, 2)

    def save_report(self):
        log_file = os.path.join(self.log_dir, "report.json")
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=4, ensure_ascii=False)
        
        txt_file = os.path.join(self.log_dir, "report.txt")
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(f"=== BÁO CÁO KẾT QUẢ PANORAMA (CLASSICAL TRUYỀN THỐNG) ===\n")
            f.write(f"Thời gian làm báo cáo: {self.metrics['timestamp']}\n")
            f.write("Quy trình bao gồm: Tiền xử lý ánh sáng -> Trích xuất features SIFT/ORB -> Nối FLANN + Lọc Lowe's -> Tính RANSAC Homography -> Projective Warping -> Laplacian Blending.\n")
            f.write("--------------------------------------------------\n\n")
            
            f.write("1. Trích xuất đặc trưng (Features SIFT/ORB):\n")
            for f_data in self.metrics["features"]:
                f.write(f"  - Ảnh {f_data['image_index']}: Sử dụng phương pháp {f_data['method']} -> Tìm được {f_data['keypoints']} điểm keypoints ({f_data['time_ms']} ms)\n")
                
            f.write("\n2. Ghép cặp (Matching - FLANN K-Nearest Neighbors + Lowe's Ratio Test):\n")
            for m_data in self.metrics["matching"]:
                f.write(f"  - Cặp ảnh {m_data['pair']}: Tổng số matches (k=2): {m_data['total_matches']} -> Độ tin cậy (Good matches): {m_data['good_matches']} ({m_data['time_ms']} ms)\n")

            f.write("\n3. Lọc nhiễu & Ma trận chuyển đổi (RANSAC & Homography):\n")
            for h_data in self.metrics["homography"]:
                f.write(f"  - Cặp ảnh {h_data['pair']}: Điểm Inliers hợp lệ: {h_data['inliers']} (Tỷ lệ phân tán: {h_data['inlier_ratio']*100:.2f}%)\n")
                
            if "time_ms" in self.metrics["blending"]:
                f.write(f"\n4. Projective Warping & Laplacian Blending (Thành phẩm):\n")
                f.write(f"  - Hoàn tất thời gian hoà trộn: {self.metrics['blending']['time_ms']} ms\n")
                
        print(f"[*] Đã xuất file báo cáo (report.txt) đo lường tại {txt_file} thành công.")

pano_logger = PanoLogger()
