import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Đảm bảo thư mục đầu ra tồn tại
os.makedirs('data', exist_ok=True)

print("--- 1. Đọc dữ liệu từ thư mục data/ ---")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Tách đặc trưng (X) và nhãn mục tiêu (y)
X = train_df.drop(columns=['Id', 'SalePrice'])
# Sử dụng log1p (log(1+x)) để chuẩn hóa phân phối biến giá nhà (tránh bị lệch - skewness)
y = np.log1p(train_df['SalePrice']) 
X_test = test_df.drop(columns=['Id'])

print("--- 2. Phân loại đặc trưng Số và Chữ ---")
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

print("--- 3. Xây dựng Pipeline tiền xử lý dữ liệu ---")
# Đối với dữ liệu Số: Điền giá trị khuyết bằng số Trung vị, sau đó Chuẩn hóa dữ liệu (StandardScaler)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Đối với dữ liệu Chữ: Điền khuyết bằng chữ 'Missing', sau đó mã hóa sang dạng số (One-Hot)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Gộp các luồng xử lý lại theo cột tương ứng
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

print("--- 4. Tích hợp thuật toán Linear Regression vào Pipeline ---")
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=1.0))
])

print("--- 5. Tách tập Train/Validation cục bộ để đánh giá thử ---")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- 6. Huấn luyện mô hình và tính Độ lỗi cục bộ ---")
model_pipeline.fit(X_train, y_train)

# Dự đoán thử trên tập Validation để xem mô hình chạy tốt không
val_predictions = model_pipeline.predict(X_val)
# Tính RMSE (Root Mean Squared Error) trên thang đo Log
rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
print(f"-> Độ lỗi cục bộ (Validation RMSE): {rmse:.4f}")

print("--- 7. Dự báo trên tập Test của Kaggle và xuất file nộp bài ---")
# Huấn luyện lại trên 100% dữ liệu gốc để mô hình đạt độ chính xác cao nhất
model_pipeline.fit(X, y)
test_predictions = model_pipeline.predict(X_test)

# Chuyển đổi ngược từ Log về giá tiền gốc (đơn vị USD) bằng hàm expm1 (exp(x)-1)
final_predictions = np.expm1(test_predictions)

# Tạo dataframe đúng định dạng cấu trúc nộp bài của cuộc thi Kaggle
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'SalePrice': final_predictions
})

submission.to_csv('data/linear_submission.csv', index=False)
print("-> Đã xuất file dự báo thành công tại: data/linear_submission.csv")
