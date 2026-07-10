import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("D:\\Github-Contributer-predictor\\data\\raw\\pandas_prs.csv")
df.columns

df.describe(include='all')


df.info(memory_usage='deep')

def audit_dataframe(df):
    audit_df = pd.DataFrame({
        'Data Type': df.dtypes,
        'Missing Values': df.isnull().sum(),
        '% Missing': (df.isnull().mean()) * 100,
        'Unique Values': df.nunique()
    })
    
    return audit_df.sort_values(by='% Missing', ascending=False).round(2)

print(audit_dataframe(df))
print("columns before drop:",df.shape[1])
df=df.dropna(axis=1,how='all')
print("columns after:",df.shape[1])
constant_cols = df.columns[df.nunique(dropna=False) == 1]

print(f"Constant columns: {len(constant_cols)}")
print(constant_cols)
url_cols = [col for col in df.columns if "url" in col.lower()]
df = df.drop(columns=url_cols)
df.shape[1]
constant_cols = df.columns[df.nunique(dropna=False) == 1]
print(len(constant_cols))
df = df.drop(columns=constant_cols)
df.shape[1]
df["user.id"]
def audit_dataframe(df):
    audit_df = pd.DataFrame({
        'Data Type': df.dtypes,
        'Missing Values': df.isnull().sum(),
        '% Missing': (df.isnull().mean()) * 100,
        'Unique Values': df.nunique()
    })
    return audit_df.sort_values(by='% Missing', ascending=False).round(2)

print(audit_dataframe(df))

df['body'] = df['body'].fillna("no_description")

df['title'] = df['title'].fillna("no_title")
df.shape[0]
date_cols=[col for col in df.columns if "_at" in col.lower()]
print(date_cols)
for col in date_cols:
    df[col]=pd.to_datetime(df[col])
print(df[date_cols].dtypes)
df = df.sort_values(by=['user.login', 'created_at'])
df = df.reset_index(drop=True)
df.shape[0]
missing_percent = df.isna().mean() * 100

high_missing_cols = missing_percent[missing_percent > 95].index

print(f"Dropping {len(high_missing_cols)} columns")
print(high_missing_cols.tolist())

df = df.drop(columns=high_missing_cols)

[col for col in df.columns if "login" in col.lower()]
contributers=df["user.login"]
user=contributers[0]
user_df=df[df["user.login"]==user]
print(user_df[["user.login", "created_at", "merged_at", "title"]])
df["author_association"].value_counts()
df.shape

def build_dataset_for_N(df, N, W_days=180):
    
    dataset_max_date = df['created_at'].max()
    
    df = df.sort_values(['user.id', 'created_at']).copy()
    df['pr_rank'] = df.groupby('user.id').cumcount() + 1
    
    pr_counts = df.groupby('user.id').size()
    eligible_users = pr_counts[pr_counts >= N].index
    df_eligible = df[df['user.id'].isin(eligible_users)].copy()
    
    obs = df_eligible[df_eligible['pr_rank'] <= N]
    future = df_eligible[df_eligible['pr_rank'] > N]
    
    cutoffs = obs[obs['pr_rank'] == N][['user.id', 'created_at']].rename(
        columns={'created_at': 'cutoff_date'}
    )
    
    W = pd.Timedelta(days=W_days)
    cutoffs['outcome_end'] = cutoffs['cutoff_date'] + W
    cutoffs['is_censored'] = cutoffs['outcome_end'] > dataset_max_date
    
    labels_df = cutoffs[~cutoffs['is_censored']].copy()
    
    future_tracked = future.merge(labels_df, on='user.id', how='inner')
    
   
    active_in_window = future_tracked[
        (future_tracked['created_at'] > future_tracked['cutoff_date']) & 
        (future_tracked['created_at'] <= future_tracked['outcome_end'])
    ]
    
    retained_users = active_in_window['user.id'].unique()
    labels_df['label'] = labels_df['user.id'].isin(retained_users).astype(int)
    
    labels_df = labels_df[['user.id', 'label']]
    obs["days_since_prev_pr"] = (
    obs.groupby("user.id")["created_at"]
       .diff()
       .dt.days
)
    features = obs.groupby("user.id").agg(
    n_obs_prs=("pr_rank", "count"),
    frac_merged=("merged_at", lambda x: x.notna().mean()),
    first_pr_date=("created_at", "min"),
    last_obs_pr_date=("created_at", "max"),
    author_association=("author_association", "last"),
    avg_days_between_prs=("days_since_prev_pr", "mean")
).reset_index()
    
    features['days_to_reach_Nth_pr'] = (
        features['last_obs_pr_date'] - features['first_pr_date']
    ).dt.days
    
    features = features.drop(columns=['first_pr_date', 'last_obs_pr_date'])
    
    final_df = features.merge(labels_df, on='user.id', how='inner')
    return final_df
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

candidate_N = [2, 3,4,5,10]

results = []

for N in candidate_N:

    # Build dataset
    dataset = build_dataset_for_N(df, N)
    


    X = dataset.drop(columns=["user.id", "label"])
    y = dataset["label"]

    # Train (80%) and Temp (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Validation (10%) and Test (10%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )

    # Train model
    categorical_features = ["author_association"]

    numerical_features = [
        col for col in X_train.columns
        if col not in categorical_features
]
    preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            "passthrough",
            numerical_features
        )
    ]
)
    pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(class_weight="balanced"))
])

    pipeline.fit(X_train, y_train)

    y_val_pred = pipeline.predict(X_val)
    val_score = f1_score(y_val, y_val_pred)

    print(f"N = {N}, Validation F1 = {val_score:.4f}")
    print(f"Eligible contributors: {len(dataset)}")
    print(dataset["label"].value_counts())

    results.append({
        "N": N,
        "validation_score": val_score
    })

results = pd.DataFrame(results)



best_N = 3
final_df=build_dataset_for_N(df,best_N)
from pathlib import Path

# Create directory if it doesn't exist
processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

# Save dataset
final_df.to_csv(processed_dir / "github_contributor_dataset_N3.csv", index=False)

print("Dataset saved successfully!")
best_N = 3
final_df=build_dataset_for_N(df,best_N)
X = final_df.drop(columns=["user.id", "label"])
y = final_df["label"]

# Train/Validation/Test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)


# Combine train and validation
X_train_final = pd.concat([X_train, X_val], axis=0)
y_train_final = pd.concat([y_train, y_val], axis=0)



categorical_features = ["author_association"]

numerical_features = [
    col for col in X_train_final.columns
    if col not in categorical_features
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            "passthrough",
            numerical_features
        )
    ]
)

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(class_weight="balanced"))
])

# Train final model
final_pipeline.fit(X_train_final, y_train_final)

# Predict on the untouched test set
y_test_pred = final_pipeline.predict(X_test)

# Evaluate
test_f1 = f1_score(y_test, y_test_pred)

print(f"Final Test F1 Score: {test_f1:.4f}")

