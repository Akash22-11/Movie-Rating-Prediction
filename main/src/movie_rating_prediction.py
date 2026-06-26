import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 55)
print("  TASK 2 — MOVIE RATING PREDICTION")
print("=" * 55)
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "models")


os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "movies.csv")

df = pd.read_csv(DATA_PATH)
print(f"\n[1] Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"    Target  : vote_average (ratings 0-10)")
print(f"    Rating range : {df['vote_average'].min():.1f} - {df['vote_average'].max():.1f}")
print(f"    Mean rating  : {df['vote_average'].mean():.2f}\n")

print("[2] Exploratory Data Analysis")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Movie Rating Prediction - EDA', fontsize=16, fontweight='bold')

axes[0,0].hist(df['vote_average'], bins=30, color='#3498db', edgecolor='white')
axes[0,0].set_title('Rating Distribution')
axes[0,0].set_xlabel('Rating (vote_average)')
axes[0,0].set_ylabel('Count')
axes[0,0].axvline(df['vote_average'].mean(), color='red',
                  linestyle='--', label=f"Mean: {df['vote_average'].mean():.2f}")
axes[0,0].legend()

axes[0,1].scatter(df['vote_count'], df['vote_average'], alpha=0.3,
                  color='#9b59b6', s=10)
axes[0,1].set_title('Vote Count vs Rating')
axes[0,1].set_xlabel('Vote Count')
axes[0,1].set_ylabel('Rating')

df_b = df[df['budget'] > 0]
axes[0,2].scatter(df_b['budget']/1e6, df_b['vote_average'], alpha=0.3,
                  color='#e67e22', s=10)
axes[0,2].set_title('Budget (M$) vs Rating')
axes[0,2].set_xlabel('Budget (Millions $)')
axes[0,2].set_ylabel('Rating')

df_rt = df[df['runtime'] > 0]
axes[1,0].scatter(df_rt['runtime'], df_rt['vote_average'], alpha=0.3,
                  color='#2ecc71', s=10)
axes[1,0].set_title('Runtime vs Rating')
axes[1,0].set_xlabel('Runtime (minutes)')
axes[1,0].set_ylabel('Rating')

axes[1,1].scatter(df['popularity'], df['vote_average'], alpha=0.3,
                  color='#e74c3c', s=10)
axes[1,1].set_title('Popularity vs Rating')
axes[1,1].set_xlabel('Popularity Score')
axes[1,1].set_ylabel('Rating')

lang_ratings = (df.groupby('original_language')['vote_average']
                  .mean().sort_values(ascending=False).head(10))
lang_ratings.plot(kind='bar', ax=axes[1,2], color='#1abc9c', edgecolor='white')
axes[1,2].set_title('Avg Rating by Language (Top 10)')
axes[1,2].set_xlabel('Language')
axes[1,2].set_ylabel('Avg Rating')
axes[1,2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "movie_evaluation.png"),
    dpi=150,
    bbox_inches="tight"
)
plt.savefig(os.path.join(OUTPUT_DIR, "movie_evaluation.png"),
            dpi=150,
            bbox_inches="tight")            
plt.close()
print("    EDA plots saved.\n")

print("[3] Feature Engineering")

data = df.copy()

GENRE_LABELS = ['Action','Adventure','Animation','Comedy','Crime',
                'Documentary','Drama','Family','Fantasy','History',
                'Horror','Music','Mystery','Romance','Science',
                'Thriller','War','Western','Foreign']

data['genres'] = data['genres'].fillna('')
for g in GENRE_LABELS:
    data[f'genre_{g}'] = data['genres'].str.contains(g, case=False).astype(int)

genre_cols = [f'genre_{g}' for g in GENRE_LABELS]
print(f"    Genres one-hot encoded : {len(GENRE_LABELS)} genre columns")

dir_avg = data.groupby('director')['vote_average'].mean()
data['director_avg_rating'] = (data['director']
                                .map(dir_avg)
                                .fillna(data['vote_average'].mean()))
print(f"    Director avg rating feature : {data['director'].nunique()} unique directors")

def count_cast(val):
    try:
        items = __import__('ast').literal_eval(val)
        return len(items) if isinstance(items, list) else 0
    except:
        return 0

data['cast_count'] = data['cast'].apply(count_cast)

# --- Year & decade ---
data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')
data['release_year'] = data['release_date'].dt.year.fillna(0).astype(int)
data['decade']       = (data['release_year'] // 10 * 10).astype(int)

# --- Log-transform skewed numerics ---
data['log_budget']     = np.log1p(data['budget'])
data['log_revenue']    = np.log1p(data['revenue'])
data['log_popularity'] = np.log1p(data['popularity'])
data['log_vote_count'] = np.log1p(data['vote_count'])

# --- Language flag ---
data['is_english'] = (data['original_language'] == 'en').astype(int)

# --- Runtime fill ---
data['runtime'] = data['runtime'].fillna(data['runtime'].median())

print(f"    Log transforms : budget, revenue, popularity, vote_count")
print(f"    Release years  : {data[data['release_year']>0]['release_year'].min()} "
      f"- {data['release_year'].max()}\n")

# ─────────────────────────────────────────────
# 4. FEATURE SELECTION
# ─────────────────────────────────────────────
print("[4] Selecting Features")

FEATURES = (
    ['log_budget','log_revenue','log_popularity','log_vote_count',
     'runtime','release_year','is_english','director_avg_rating','cast_count']
    + genre_cols
)
TARGET = 'vote_average'

data = data[data[TARGET] > 0].reset_index(drop=True)
X = data[FEATURES].fillna(0)
y = data[TARGET]

print(f"    Total features : {len(FEATURES)}")
print(f"    Usable samples : {len(X)}\n")

# ─────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT + SCALING
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_test_sc   = scaler.transform(X_test)

print(f"[5] Split -> Train: {len(X_train)} | Test: {len(X_test)}\n")

# ─────────────────────────────────────────────
# 6. TRAIN MODELS
# ─────────────────────────────────────────────
print("[6] Training Models")
print("-" * 50)

models = {
    'Linear Regression' : LinearRegression(),
    'Ridge Regression'  : Ridge(alpha=1.0),
    'Random Forest'     : RandomForestRegressor(n_estimators=200, max_depth=8,
                                                 random_state=42, n_jobs=-1),
    'Gradient Boosting' : GradientBoostingRegressor(n_estimators=200,
                                                     learning_rate=0.05,
                                                     max_depth=5,
                                                     random_state=42),
}

results = {}
for name, model in models.items():
    use_sc = name in ['Linear Regression', 'Ridge Regression']
    Xtr = X_train_sc if use_sc else X_train
    Xte = X_test_sc  if use_sc else X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    results[name] = {'model': model, 'y_pred': y_pred,
                     'mae': mae, 'rmse': rmse, 'r2': r2}

    print(f"  {name}")
    print(f"    MAE  : {mae:.4f}")
    print(f"    RMSE : {rmse:.4f}")
    print(f"    R2   : {r2:.4f}")
    print()

# ─────────────────────────────────────────────
# 7. BEST MODEL
# ─────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['r2'])
best = results[best_name]
print(f"[7] Best Model : {best_name}")
print(f"    MAE  : {best['mae']:.4f}")
print(f"    RMSE : {best['rmse']:.4f}")
print(f"    R2   : {best['r2']:.4f}\n")

# ─────────────────────────────────────────────
# 8. EVALUATION PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'Model Evaluation - {best_name}', fontsize=14, fontweight='bold')

# Actual vs Predicted
axes[0,0].scatter(y_test, best['y_pred'], alpha=0.4, color='#3498db', s=20)
mn, mx = y_test.min(), y_test.max()
axes[0,0].plot([mn,mx],[mn,mx],'r--', linewidth=2, label='Perfect Prediction')
axes[0,0].set_title('Actual vs Predicted Rating')
axes[0,0].set_xlabel('Actual Rating')
axes[0,0].set_ylabel('Predicted Rating')
axes[0,0].legend()

# Residuals
residuals = y_test.values - best['y_pred']
axes[0,1].hist(residuals, bins=40, color='#e74c3c', edgecolor='white')
axes[0,1].axvline(0, color='black', linestyle='--')
axes[0,1].set_title('Residual Distribution')
axes[0,1].set_xlabel('Residual (Actual - Predicted)')
axes[0,1].set_ylabel('Count')

# Model comparison R2
names  = list(results.keys())
r2s    = [results[m]['r2'] for m in names]
colors = ['#2ecc71' if m == best_name else '#95a5a6' for m in names]
bars   = axes[1,0].bar(names, r2s, color=colors, edgecolor='white')
axes[1,0].set_title('R2 Score - All Models')
axes[1,0].set_ylabel('R2 Score')
axes[1,0].set_ylim(0, 1)
axes[1,0].tick_params(axis='x', rotation=15)
for bar, val in zip(bars, r2s):
    axes[1,0].text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

# Feature importance (Random Forest)
rf   = results['Random Forest']['model']
fimp = pd.Series(rf.feature_importances_, index=FEATURES)
fimp.sort_values(ascending=False).head(15).sort_values().plot(
    kind='barh', ax=axes[1,1], color='#9b59b6', edgecolor='white')
axes[1,1].set_title('Top 15 Feature Importances (Random Forest)')
axes[1,1].set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "movie_evaluation.png"),
            dpi=150,
            bbox_inches="tight")
plt.close()
print("    Evaluation plots saved.\n")

# ─────────────────────────────────────────────
# 9. SAMPLE PREDICTIONS
# ─────────────────────────────────────────────
print("[8] Sample Predictions (10 movies)")
print("-" * 60)
sample_idx = X_test.head(10).index
s = data.loc[sample_idx, ['title','release_year','director']].copy()
s['Actual']    = y_test.head(10).values
s['Predicted'] = np.round(best['y_pred'][:10], 2)
s['Error']     = np.abs(s['Actual'] - s['Predicted']).round(2)
print(s[['title','release_year','Actual','Predicted','Error']].to_string(index=False))

print("\n" + "=" * 55)
print("  DONE - All plots saved.")
print("=" * 55)
