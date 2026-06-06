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
            f.write(f"=== BÁO CÁO KẾT QUẢ PANORAMA (CHẾ ĐỘ LAI GHÉP - HYBRID AI) ===\n")
            f.write(f"Thời gian trích xuất báo cáo: {self.metrics['timestamp']}\n")
            f.write("Quy trình công nghệ: Tiền xử lý (CLAHE) -> Đọc điểm ngữ cảnh qua Neural Network (LoFTR) -> Lược bỏ nhiễu biên (MAGSAC++) -> Lan truyền ma trận tự động (BFS Projective Warping) -> Ghép đè trực tiếp ảnh lên Canvas.\n")
            f.write("--------------------------------------------------\n\n")
            
            f.write("1. Trích xuất điểm tương đồng (Deep Learning - LoFTR):\n")
            for m_data in self.metrics["matching"]:
                f.write(f"  - Cặp ảnh {m_data['pair']}: Số điểm AI tìm thấy theo ngữ cảnh: {m_data['total_matches']} -> Siêu lọc độ tự tin (Confidence > 60%): {m_data['good_matches']} ({m_data['time_ms']} ms)\n")

            f.write("\n2. Rút trích Ma trận & Tính toán Inliers (Thuật toán MAGSAC++):\n")
            for h_data in self.metrics["homography"]:
                f.write(f"  - Cặp ảnh {h_data['pair']}: Số lượng móc neo bảo toàn: {h_data['inliers']} (Tỷ lệ bám dính Inliers / Điểm AI: {h_data['inlier_ratio']*100:.2f}%)\n")
                
            if "time_ms" in self.metrics["blending"]:
                f.write(f"\n3. Phân luồng Cây (BFS Projective Warping) & Ghép đè ảnh:\n")
                f.write(f"  - Hoàn tất thời gian vẽ bạt lưới và bù trừ khối lượng: {self.metrics['blending']['time_ms']} ms\n")
                
        print(f"[*] Đã xuất file báo cáo (report.txt) đo lường tại {txt_file} thành công.")

pano_logger = PanoLogger()
