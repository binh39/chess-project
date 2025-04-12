async function getValidMovesFromAPI(square) {
    try {
      const response = await fetch("http://localhost:8000/valid_move", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ square }), // Gửi thông tin ô cần kiểm tra
      });
  
      // Kiểm tra nếu có lỗi từ server
      if (!response.ok) {
        throw new Error("Failed to fetch valid moves from API");
      }
  
      // Parse response từ API
      const data = await response.json();
  
      // Trả về danh sách các valid moves
      return data.moves;
    } catch (error) {
      console.error("Error fetching valid moves:", error);
      return []; // Trả về mảng rỗng trong trường hợp có lỗi
    }
  }
  