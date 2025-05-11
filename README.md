Chess AlphaZero 
1. Giới thiệu: Chess AlphaZero là một dự án AI chơi cờ vua dựa trên thuật toán AlphaZero của DeepMind, sử dụng kết hợp giữa Monte Carlo Tree Search (MCTS) và mạng neural sâu để tự học cách chơi cờ vua từ đầu mà không cần dữ liệu ván cờ con người. Dự án này mô phỏng lại hệ thống AlphaZero cho cờ vua, bao gồm: Môi trường cờ vua (ChessEnv) Thuật toán MCTS kết hợp với mạng neural Quá trình tự chơi (self-play), huấn luyện, và đánh giá mô hình Mục tiêu là xây dựng một AI có thể tự học và đạt trình độ cao mà không cần dữ liệu ván cờ của con người. Tính năng Tự động sinh dữ liệu tự chơi (self-play) Huấn luyện mạng neural chính sách và giá trị Tìm kiếm nước đi bằng MCTS song song đa luồng Điều chỉnh tham số động theo tiến trình trận đấu (số mô phỏng, virtual loss, nhiệt độ, ...) Hỗ trợ kiểm tra, đánh giá mô hình và so sánh các phiên bản 
2.	Cấu trúc chính
a. Neural Network Kiến trúc: Mạng residual sâu (deep residual network) với nhiều khối residual, mô phỏng theo AlphaZero. Đầu vào: Tensor (18, 8,  biểu diễn bàn cờ (12 mặt phẳng quân cờ, 4 mặt phẳng nhập thành, 1 mặt phẳng luật 50 nước, 1 mặt phẳng en passant). Đầu ra: Policy head: Xác suất cho tất cả các nước đi hợp lệ (softmax)(vector gồm 1968 phần tử, encode từ các nước đi có thể trong 1 bàn cờ) Value head: Giá trị đánh giá vị thế hiện tại (tanh, [-1, 1]). Huấn luyện: Sử dụng dữ liệu từ tự chơi, warm-up thông qua supervised learning, tối ưu loss gồm cross-entropy cho policy và MSE cho value. b. Thuật toán MCTS (Monte Carlo Tree Search) Selection: Duyệt cây theo công thức PUCT, cân bằng giữa khai thác và khám phá. Expansion: Mở rộng nút mới, dùng neural network để dự đoán policy và value. Simulation: Không mô phỏng ngẫu nhiên, dùng trực tiếp value từ mạng. Backpropagation: Lan truyền ngược giá trị value lên các nút cha. Kỹ thuật nâng cao: Virtual loss: Tối ưu hóa tìm kiếm song song. Dirichlet noise: Thêm nhiễu vào policy ở nút gốc để tăng khám phá. Temperature sampling: Điều chỉnh mức ngẫu nhiên khi chọn nước đi cuối cùng. c. Môi trường cờ vua (ChessEnv) Quản lý trạng thái bàn cờ, luật chơi, kiểm tra hợp lệ nước đi, xác định kết thúc ván đấu (chiếu hết, hòa, lặp lại, luật 50 nước...). Chuyển đổi trạng thái bàn cờ thành tensor đầu vào cho neural network. 
3.	Quản lý tự chơi, huấn luyện, đánh giá
worker/self_play.py: Tự động sinh dữ liệu tự chơi. worker/sl.py: Huấn luyện mạng neural từ dữ liệu tự chơi. worker/evaluate.py: Đánh giá mô hình mới so với mô hình cũ. manager.py, run.py: Quản lý pipeline tổng thể. 
4.	Quy trình tổng thể:
a. Warm-up: Sử dụng dữ liệu các ván cờ thực dưới định dạng file PGN, thông qua sl.py chuyển thành data dạng (state, policy_vector, value) trong đó state được biểu diễn dưới dạng string dạng FEN, policy_vector là vector xác suất của các nước đi hợp lệ trong 1 ván đấu 
b. Self-play: Tạo nhiều tiến trình bằng self_play.py với mỗi tiến trình sử dụng một pipeline tự đánh giữa model hiện tại với hướng dẫn từ MCTS. Sau đó lưu lại dữ liệu ván đấu dưới dạng (s, p, v) để cho quá trình training. 
c. Training: Sử dụng worker/optimize.py để huấn luyện lại mạng neural từ dữ liệu tự chơi. 
d. Evaluation So sánh mô hình mới với mô hình cũ bằng nhiều trận. Nếu tỉ lệ thắng lớn hơn 55%, cập nhật làm model mới. 
e. Lặp lại Self-play -> Training -> Evaluate.
5. Giao diện chơi cờ (Frontend)
- Chế độ người chơi vs AI: Người dùng có thể thi đấu trực tiếp với mô hình AI.
- Chế độ AI vs AI: Hai mô hình AI tự động thi đấu với nhau.
- Giao diện web: Được xây dựng bằng React, hiển thị bàn cờ, nước đi, và kết quả trận đấu.
6. Hệ thống phía sau (Backend)
- FastAPI: Sử dụng để xây dựng API xử lý và phục vụ các nước đi từ mô hình AI.
- Tích hợp AI model: Backend kết nối với mô hình để trả về nước đi tối ưu tại mỗi lượt.
