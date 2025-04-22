# ecommerce_return_prediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Create a mock dataset
data = {
    'product_price': np.random.uniform(10, 500, 1000),
    'category': np.random.choice(['clothing', 'electronics', 'home', 'beauty'], 1000),
    'user_rating': np.random.choice([1, 2, 3, 4, 5], 1000),
    'delivery_time': np.random.randint(1, 10, 1000),
    'return_reason_code': np.random.choice(['fit_issue', 'damaged', 'no_reason', 'wrong_item'], 1000),
    'is_returned': np.random.choice([0, 1], 1000, p=[0.7, 0.3])  # 30% return rate
}
df = pd.DataFrame(data)

# 2. Features and Target
X = df.drop('is_returned', axis=1)
y = df['is_returned']

# 3. Preprocessing pipeline
categorical_cols = ['category', 'return_reason_code']
numerical_cols = ['product_price', 'user_rating', 'delivery_time']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# 4. Full pipeline with classifier
clf_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 5. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train the model
clf_pipeline.fit(X_train, y_train)

# 7. Evaluate the model
y_pred = clf_pipeline.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Make a prediction on a new order
sample_order = pd.DataFrame([{
    'product_price': 120,
    'category': 'clothing',
    'user_rating': 4,
    'delivery_time': 3,
    'return_reason_code': 'fit_issue'
}])

prediction = clf_pipeline.predict(sample_order)
print("\nPrediction (1=Will Return, 0=Will Not Return):", prediction[0])
