### Mô hình huấn luyện cho việc nhận dạng khuôn mặt ở điều kiện thiếu sáng có thể xem ở nhánh khtk

# Hệ Thống Nhận Diện Khuôn Mặt Chính Xác

Đây là một ứng dụng web được xây dựng bằng Python (Flask) với mục tiêu nhận diện khuôn mặt một cách chính xác, đặc biệt là khả năng phân biệt người dùng đã đăng ký và "người lạ".

## Tính năng chính

-   **Giao diện Web hiện đại**: Dễ dàng tương tác và quản lý.
-   **Thu thập dữ liệu thông minh**: Cho phép người dùng đăng ký bằng cách cung cấp nhiều ảnh khuôn mặt thông qua webcam, giúp tăng độ chính xác của mô hình.
-   **Tùy chọn Upload**: Người dùng cũng có thể tải lên các bộ ảnh đã có sẵn.
-   **Nhận diện Real-time**: Sử dụng webcam để phân tích và đưa ra kết quả nhận diện ngay lập tức.
-   **Ngưỡng nhận diện tùy chỉnh**: Dễ dàng tinh chỉnh độ nhạy của mô hình để phân biệt "người lạ".
-   **Quản lý người dùng**: Xem danh sách và xóa người dùng khỏi hệ thống.

## Hướng dẫn cài đặt

1.  **Tạo và kích hoạt môi trường ảo:**

    *   **Trên Windows:**
        ```powershell
        python -m venv venv
        .\venv\Scripts\Activate.ps1
        ```
    *   **Trên Linux/macOS:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

2.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Chạy ứng dụng:**
    ```bash
    python app.py
    ```

4.  **Truy cập ứng dụng:** Mở trình duyệt và đi đến `http://127.0.0.1:5000`.

## Cấu trúc dự án

```
.
├── app.py                  # Backend chính của Flask
├── requirements.txt        # Các thư viện Python
├── face_data/              # Thư mục chứa dữ liệu đã xử lý
│   ├── user_images/        # Ảnh khuôn mặt đã được cắt của từng người
│   ├── users.json          # Thông tin người dùng
│   ├── encodings.npy       # Vector mã hóa khuôn mặt
│   └── labels.json         # Nhãn tương ứng
├── static/
│   └── styles.css
├── templates/
│   └── index.html
└── README.md
``` 
