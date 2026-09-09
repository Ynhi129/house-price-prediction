import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

print("--- 1. Đọc dữ liệu từ thư mục data/ ---")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Tách đặc trưng (Features) và nhãn (Target)
X = train_df.drop(columns=['Id', 'SalePrice'])
y = np.log1p(train_df['SalePrice']) # Log-transform để giảm độ lệch (skewness) của giá nhà
X_test = test_df.drop(columns=['Id'])

print("--- 2. Phân loại đặc trưng Số và Chữ ---")
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

print("--- 3. Xây dựng Pipeline tiền xử lý dữ liệu ---")
# Xử lý dữ liệu số: Điền khuyết bằng giá trị trung vị (median)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# Xử lý dữ liệu chữ: Điền khuyết bằng chữ 'Missing' và mã hóa One-Hot
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Gộp 2 luồng xử lý lại
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

print("--- 4. Tích hợp Mô hình Random Forest vào Pipeline ---")
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

print("--- 5. Chia tập train/val để đánh giá cục bộ ---")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- 6. Huấn luyện mô hình ---")
model_pipeline.fit(X_train, y_train)

# Đánh giá bằng chỉ số RMSE (Root Mean Squared Error)
val_predictions = model_pipeline.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
print(f"Độ lỗi cục bộ (Validation RMSE): {rmse:.4f}")

print("--- 7. Dự báo trên tập Test của Kaggle và xuất file ---")
# Huấn luyện lại trên toàn bộ tập Train trước khi dự báo tập Test
model_pipeline.fit(X, y)
test_predictions = model_pipeline.predict(X_test)
# Chuyển ngược từ dạng log về dạng giá tiền gốc
final_predictions = np.expm1(test_predictions)

# Tạo file submission đúng định dạng Kaggle yêu cầu
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'SalePrice': final_predictions
})
submission.to_csv('data/submission.csv', index=False)
print("Đã tạo thành công file dự báo: data/submission.csv")
