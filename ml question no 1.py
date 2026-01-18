# ==============================
# 1. IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==============================
# 2. LOAD DATA
# ==============================
file_path = "Section-A-Q1-USA-Real-Estate-Dataset-realtor-data-Rev1.csv"
df = pd.read_csv(file_path)
print("Raw Shape:", df.shape)

# ==============================
# 3. DATA CLEANING
# ==============================
# Drop high-cardinality or unnecessary columns
df.drop(columns=['street', 'brokered_by', 'prev_sold_date'], inplace=True)

# Drop rows with missing target
df = df.dropna(subset=['price'])

# ==============================
# 4. FEATURE ENGINEERING
# ==============================
df['lot_sqft'] = df['acre_lot'] * 43560
df['bed_bath_ratio'] = df['bed'] / (df['bath'] + 0.5)

# ==============================
# 5. OUTLIER REMOVAL
# ==============================
for col in ['price','house_size','acre_lot']:
    low = df[col].quantile(0.01)
    high = df[col].quantile(0.99)
    df = df[(df[col] >= low) & (df[col] <= high)]

print("Shape after cleaning:", df.shape)
print(df.head())

# ==============================
# 6. EDA – SAVE PLOTS
# ==============================
plots = []

# Price Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['price'], bins=100)
plt.title("Price Distribution")
plt.xlim(0, 1_000_000)
plt.savefig("price_distribution.png")
plt.close()

# Log Price Distribution
plt.figure(figsize=(8,5))
sns.histplot(np.log1p(df['price']), bins=100)
plt.title("Log(Price) Distribution")
plt.savefig("log_price_distribution.png")
plt.close()

# Correlation Heatmap
num_cols = ['price','bed','bath','house_size','acre_lot']
plt.figure(figsize=(7,5))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

# Price by State
plt.figure(figsize=(8,5))
df.groupby('state')['price'].median().sort_values().plot(kind='barh')
plt.title("Median House Price by State")
plt.xlabel("Price")
plt.savefig("median_price_by_state.png")
plt.close()

# Bedrooms vs Price
plt.figure(figsize=(8,5))
sns.boxplot(x='bed', y='price', data=df)
plt.ylim(0, 1_000_000)
plt.title("Price by Bedrooms")
plt.savefig("price_by_bed.png")
plt.close()

# Bathrooms vs Price
plt.figure(figsize=(8,5))
sns.boxplot(x='bath', y='price', data=df)
plt.ylim(0, 1_000_000)
plt.title("Price by Bathrooms")
plt.savefig("price_by_bath.png")
plt.close()

# Lot Size vs Price
plt.figure(figsize=(8,5))
sns.scatterplot(x='lot_sqft', y='price', data=df, alpha=0.3)
plt.title("Price vs Lot Size")
plt.xlim(0, 500_000)
plt.ylim(0, 1_000_000)
plt.savefig("price_vs_lot.png")
plt.close()

# ==============================
# 7. TARGET & FEATURES
# ==============================
y = np.log1p(df['price'])
X = df.drop(columns=['price'])

num_features = ['bed', 'bath', 'house_size', 'acre_lot', 'lot_sqft', 'bed_bath_ratio']
cat_features = ['city', 'state', 'status']

# ==============================
# 8. PREPROCESSING PIPELINE
# ==============================
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_features),
    ('cat', categorical_transformer, cat_features)
])

# ==============================
# 9. MODELS
# ==============================
models = {
    "Ridge": Ridge(alpha=1.0),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
    "RandomForest": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
}
# ==============================
# 10. TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.sample(frac=0.2, random_state=42) 
y_train = y_train.loc[X_train.index]
# ==============================
# 11. TRAIN & EVALUATE
# ==============================
results = {}
for name, model in models.items():
    pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    results[name] = {
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds)
    }

results_df = pd.DataFrame(results).T
print("\nMODEL COMPARISON")
print(results_df)

# ==============================
# 12. FINAL PREDICTION EXAMPLE
# ==============================
best_model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=150, max_depth=20, random_state=42, n_jobs=-1))
])
best_model.fit(X_train, y_train)

sample = X_test.iloc[[0]]
predicted_price = np.expm1(best_model.predict(sample))
print("\nExample Predicted Price:", predicted_price[0])
# ==============================
# 13. FEATURE IMPORTANCE
# ==============================
importances = best_model.named_steps['model'].feature_importances_
feature_names = best_model.named_steps['preprocessor'].transformers_[0][2] + \
                best_model.named_steps['preprocessor'].named_transformers_['cat']['onehot'].get_feature_names_out(cat_features).tolist()
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(20)
plt.figure(figsize=(8,6))
feat_imp.plot(kind='barh')
plt.title("Top 20 Feature Importances")
plt.savefig("feature_importance.png")
plt.close()
