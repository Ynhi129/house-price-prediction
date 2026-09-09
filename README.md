\# 🏡 Dự án Dự báo giá nhà (Kaggle House Prices)



Dự án này sử dụng Học máy (Machine Learning) để dự báo giá nhà dựa trên các đặc trưng về diện tích, vị trí, vật liệu cấu trúc của căn nhà. Dữ liệu được lấy từ cuộc thi nổi tiếng trên Kaggle: \*\*House Prices - Advanced Regression Techniques\*\*.



\## 🛠️ Công nghệ sử dụng

\* \*\*Ngôn ngữ:\*\* Python 3.x

\* \*\*Thư viện chính:\*\* Pandas, Numpy, Scikit-learn, XGBoost



\## 📁 Cấu trúc thư mục

\* `data/`: Chứa file dữ liệu gốc (`train.csv`, `test.csv`) và file kết quả dự báo (`submission.csv`).

\* `src/`: Chứa mã nguồn xử lý dữ liệu và huấn luyện mô hình (`main.py`).



\## 🚀 Hướng dẫn cài đặt \& Sử dụng



\### 1. Cài đặt thư viện cần thiết

Mở Terminal/Command Prompt tại thư mục dự án và chạy lệnh:

```bash

pip install -r requirements.txt

```



\### 2. Chuẩn bị dữ liệu

Tải tệp `train.csv` và `test.csv` từ Kaggle, sau đó copy vào thư mục `data/`.



\### 3. Chạy mô hình dự báo

Khởi chạy file code chính để huấn luyện mô hình và xuất ra kết quả:

```bash

python src/main.py

```

Sau khi chạy xong, file kết quả `submission.csv` sẽ tự động được tạo ra trong thư mục `data/` để bạn lấy đem nộp lên Kaggle.



