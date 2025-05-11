# ♟️ Chess AlphaZero

**Chess AlphaZero** là một dự án trí tuệ nhân tạo chơi cờ vua, mô phỏng hệ thống AlphaZero nổi tiếng của DeepMind. Dự án kết hợp giữa Monte Carlo Tree Search (MCTS) và mạng neural sâu để tự học cách chơi cờ từ đầu, không sử dụng dữ liệu từ con người. Dự án này mô phỏng lại hệ thống AlphaZero cho cờ vua, bao gồm: Môi trường cờ vua (ChessEnv), thuật toán MCTS kết hợp với mạng neural Quá trình tự chơi (self-play), huấn luyện, và đánh giá mô hình. 

---

## 🚀 Mục Tiêu

Xây dựng một AI cờ vua có thể:
- Tự học chơi cờ mà không cần dữ liệu ván đấu từ con người.
- Phát triển trình độ thông qua quá trình tự chơi và huấn luyện.
- Đạt trình độ cao bằng chiến lược tìm kiếm nước đi thông minh và mạng neural mạnh mẽ.
- Điều chỉnh tham số động theo tiến trình trận đấu (số mô phỏng, virtual loss, nhiệt độ, ...).
- Hỗ trợ kiểm tra, đánh giá mô hình và so sánh các phiên bản.
---

## 🧠 Cấu Trúc Chính

### a. Mạng Neural Network

- **Kiến trúc**:
  - Mạng residual sâu theo phong cách AlphaZero.
- **Đầu vào**:
  - Tensor kích thước `(18, 8, 8)`:
    - 12 mặt phẳng quân cờ
    - 4 mặt phẳng nhập thành
    - 1 mặt phẳng luật 50 nước
    - 1 mặt phẳng en passant
- **Đầu ra**:
  - **Policy Head**:
    - Softmax trên vector 1968 chiều (tất cả nước đi hợp lệ được mã hóa)
  - **Value Head**:
    - Một giá trị đánh giá thế cờ hiện tại, chuẩn hóa về [-1, 1] (tanh)
- **Huấn luyện**:
  - Loss = Cross-entropy (policy) + MSE (value)
  - Dữ liệu huấn luyện được sinh ra từ quá trình self-play.

### b. Thuật Toán MCTS (Monte Carlo Tree Search)

- **Selection**:
  - Duyệt cây bằng công thức PUCT, cân bằng khám phá & khai thác.
- **Expansion**:
  - Mở rộng nút mới bằng đầu ra policy từ mạng neural.
- **Simulation**:
  - Không mô phỏng ngẫu nhiên, lấy trực tiếp giá trị từ mạng neural.
- **Backpropagation**:
  - Lan truyền giá trị đánh giá ngược lên cây tìm kiếm.
- **Kỹ thuật nâng cao**:
  - `Virtual loss`: Hỗ trợ song song hóa.
  - `Dirichlet noise`: Thêm nhiễu để tăng tính khám phá.
  - `Temperature`: Kiểm soát độ ngẫu nhiên khi chọn nước đi cuối.

### c. Môi Trường Cờ Vua (ChessEnv)

- Quản lý trạng thái bàn cờ và luật chơi:
  - Kiểm tra nước đi hợp lệ.
  - Xác định kết thúc trận (chiếu hết, hòa, lặp lại, 50 nước...).
- Chuyển đổi trạng thái bàn cờ sang tensor đầu vào cho mạng neural.

---

## 🔁 Quy Trình Tổng Thể

### 1. Warm-Up (Huấn luyện sơ bộ)
- Sử dụng ván cờ thực (PGN) để tạo dữ liệu dạng `(state, policy_vector, value)`.
- `state`: Biểu diễn FEN
- `policy_vector`: Xác suất các nước đi
- `value`: Kết quả ván đấu

### 2. Self-Play (Tự chơi)
- Nhiều tiến trình tự đánh với MCTS + model hiện tại
- Sinh dữ liệu dạng `(s, p, v)` lưu vào buffer

### 3. Training
- Dùng `optimize.py` để huấn luyện lại model từ dữ liệu tự chơi.

### 4. Evaluation
- So sánh model mới với model cũ qua các ván đấu
- Nếu model mới thắng >55%, sẽ được cập nhật

### 5. Lặp lại quy trình:

Self-Play → Training → Evaluation → (loop)

## 🖥️ Giao Diện Người Dùng
### 🎮 Chế độ chơi
- Người vs AI: Thi đấu trực tiếp với AI
- AI vs AI: Hai model AI tự động thi đấu

### 🌐 Giao diện web
- Sử dụng React
- Hiển thị bàn cờ, nước đi, trạng thái trận đấu.

## 🔧 Backend API
- FastAPI: Sử dụng để xây dựng API xử lý và phục vụ các nước đi từ mô hình AI.
- Tích hợp AI model: Backend kết nối với mô hình để trả về nước đi tối ưu tại mỗi lượt.
