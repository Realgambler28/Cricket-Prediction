
import pandas as pd

df = pd.read_csv("training_data.csv")

print("Dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

print("\nWin label distribution:")
print(df["won"].value_counts())

print("\nFeature summary:")
print(df[
    [
        "score",
        "wickets_lost",
        "balls_remaining",
        "target",
        "runs_remaining",
        "current_run_rate",
        "required_run_rate"
    ]
].describe())

print("\nSample match states:")
print(df.head(10).to_string(index=False))
