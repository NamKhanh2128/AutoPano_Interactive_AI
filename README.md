***AutoPano Interactive AI***
Giới thiệu
AutoPano Interactive AI là một hệ thống ghép ảnh panorama tự động dựa trên trí tuệ nhân tạo, được phát triển bằng Python. Dự án này cho phép người dùng ghép nhiều ảnh chồng lấn thành một bức ảnh panorama liền mạch, với khả năng xem tương tác 360° thông qua giao diện web.

***Mô tả Dự án***
Dự án AutoPano Interactive AI là một pipeline xử lý ảnh hoàn chỉnh bao gồm các module chính: trích xuất đặc trưng, ghép đôi ảnh, chọn ảnh neo, biến dạng hình trụ, ghép ảnh và làm mịn biên. Hệ thống tự động hóa toàn bộ quy trình từ việc tải ảnh đầu vào đến xuất ra ảnh panorama cuối cùng, kèm theo một trình xem web tương tác để trải nghiệm kết quả.

***Bài Toán Áp Dụng***
Bài toán chính mà dự án giải quyết là tái tạo ảnh panorama tự động từ một loạt ảnh chồng lấn (thường được chụp bằng cách xoay máy ảnh liên tiếp). Các thách thức bao gồm:

Phát hiện và ghép các đặc trưng giữa các ảnh
Tính toán biến đổi hình học (homography) để căn chỉnh ảnh
Chọn ảnh neo phù hợp để giảm thiểu tích lũy lỗi
Ghép ảnh liền mạch để ẩn các đường nối
Dự án áp dụng các kỹ thuật thị giác máy tính tiên tiến như SIFT/ORB cho phát hiện đặc trưng, RANSAC cho ước lượng homography, và kỹ thuật làm mịn biên (feathering) để tạo ra ảnh panorama chất lượng cao.

***Logic***
Tiền xử lý: Histogram Equalization (cân bằng sáng).
Trích xuất đặc trưng (Features): SIFT (chính xác) hoặc ORB (nhẹ và nhanh).
Ghép cặp (Matching): FLANN dựa trên KNN (K-Nearest Neighbors) kết hợp với thuật toán bù trừ Lowe's Ratio Test.
Lọc nhiễu & Ma trận: Thuật toán RANSAC để loại bỏ các điểm match sai và tính Ma trận cường độ (Homography).
Warping: Chuyển đổi perspective cơ bản (Projective Warping).
Thành phẩm (Blending): Tìm đường nối bằng Graph-cut (Seam Finding) và hòa trộn mềm bằng Multi-band Blending (Laplacian Pyramid).
Đánh giá: Code truyền thống bằng OpenCV, tuy dễ nhưng sẽ gặp lỗi rất nặng nếu đưa vào ảnh có ít chi tiết (bức tường trắng, bầu trời) hoặc ảnh chụp bị thay đổi góc nhìn quá lớn.

Ngưỡng Quan Trọng
Tỷ lệ khớp: 0.6
Số khớp tối thiểu: 30
Tỷ lệ inlier tối thiểu: 0.5 (50%)
Điểm khớp tổng: ≥50

---Các thư viện chính:

opencv-python: Thư viện thị giác máy tính
numpy: Tính toán số học
---Sử Dụng
Đặt 2+ ảnh chồng lấn vào thư mục input (định dạng JPG/PNG).
Chạy script chính:
Kết quả sẽ được lưu tại panorama_result.jpg.
Để xem kết quả tương tác:
Sau đó truy cập: http://localhost:8000
