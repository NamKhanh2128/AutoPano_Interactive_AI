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
Luồng Xử Lý Chính
Tải và Tiền Xử Lý Ảnh: Đọc các ảnh từ thư mục input, áp dụng biến dạng hình trụ để sửa chữa biến dạng xuyên tâm.
Trích Xuất Đặc Trưng: Sử dụng SIFT (hoặc ORB nếu SIFT thất bại) để phát hiện điểm đặc trưng và mô tả chúng.
Ghép Đôi Ảnh: Sử dụng BFMatcher với kiểm tra tỷ lệ Lowe để tìm các khớp đặc trưng giữa các cặp ảnh.
Chọn Ảnh Neo: Tính điểm tin cậy dựa trên số lượng khớp tốt và tỷ lệ inlier từ RANSAC, chọn ảnh có điểm cao nhất làm neo.
Ghép Ảnh: Biến dạng các ảnh còn lại theo ảnh neo trên một canvas lớn, sử dụng kỹ thuật làm mịn biên (feathering) với Gaussian blur.
Cắt Biên Đen: Áp dụng các phép toán hình thái học để loại bỏ biên đen tự động.
Thuật Toán Chính
Phát Hiện Đặc Trưng: SIFT ưu tiên cho độ chính xác, ORB làm dự phòng.
Ghép Đặc Trưng: BFMatcher + kiểm tra tỷ lệ Lowe (ngưỡng 0.6).
Loại Bỏ Ngoại Lai: RANSAC với homography (ngưỡng 5.0 pixel).
Biến Dạng Hình Trụ: Chuyển đổi tọa độ 2D → 3D → 2D để sửa chữa biến dạng.
Làm Mịn Biên: Làm mịn tuyến tính với mặt nạ gradient trong vùng chồng lấp.
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
