# 🏡 Dự báo giá nhà bằng thuật toán Hồi quy tuyến tính (Linear Regression)

Dự án này áp dụng thuật toán học máy cơ bản **Linear Regression** để dự đoán giá bán của các căn hộ dựa trên tập dữ liệu thuộc cuộc thi **House Prices - Advanced Regression Techniques** từ Kaggle.

## 📌 Tổng quan giải pháp
1. **Tiền xử lý dữ liệu (Data Preprocessing):**
   * Xử lý các giá trị bị khuyết (`Missing Values`) bằng phương pháp điền khuyết Trung vị đối với biến số và hằng số đối với biến phân loại.
   * Chuẩn hóa (`StandardScaler`) đặc trưng dạng số để hỗ trợ thuật toán Hồi quy tuyến tính hội tụ tốt hơn.
   * Mã hóa One-Hot (`OneHotEncoder`) các đặc trưng định danh (dạng chữ) sang dạng số.
2. **Biến đổi mục tiêu:** Sử dụng kỹ thuật Log-transform (`np.log1p`) đối với biến mục tiêu `SalePrice` nhằm giảm thiểu tác động của hiện tượng lệch phân phối dữ liệu (skewness).
3. **Mô hình hóa:** Áp dụng mô hình toán học `Linear Regression` từ thư viện Scikit-learn.

## 📁 Cấu trúc dự án
```text
├── data/
│   ├── train.csv                # Dữ liệu huấn luyện từ Kaggle (được bỏ qua qua .gitignore)
│   ├── test.csv                 # Dữ liệu kiểm thử từ Kaggle (được bỏ qua qua .gitignore)
│   └── linear_submission.csv    # File kết quả đầu ra dùng để nộp Kaggle
├── src/
│   └── linear_regression.py     # Mã nguồn Python xử lý và huấn luyện mô hình
├── .gitignore                   # Chặn các file dữ liệu nặng lên GitHub
├── requirements.txt             # Danh sách thư viện phụ thuộc
└── README.md                    # Tài liệu hướng dẫn dự án
```

## 🛠️ Cài đặt và Chạy dự án

1. **Cài đặt thư viện phụ thuộc:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Thu thập dữ liệu:**
   Tải tệp dữ liệu từ cuộc thi Kaggle và đặt vào bên trong thư mục `data/`.

3. **Thực thi mã nguồn:**
   Chạy lệnh sau tại thư mục gốc dự án để huấn luyện và xuất kết quả:
   ```bash
   python src/linear_regression.py
   ```
