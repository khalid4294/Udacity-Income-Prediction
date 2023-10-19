import pandas as pd

df = pd.read_csv("data/census.csv")

# remove the prefixed spacing
df.columns = [x.split()[0] for x in df.columns]

# create label column
for idx, col in df.iterrows():
    if col["salary"] == " >50K":
        # above 50K
        df.loc[idx, "labels"] = 1
    else:
        df.loc[idx, "labels"] = 0

df.drop(columns=["salary"])

# change from float to int
df["labels"] = df["labels"].astype(int)


# remove spaces in values under categorical cols
cat_columns = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

for col in cat_columns:
    df[col] = df[col].apply(lambda x: x.split()[0])

df.to_csv("./data/census_cleaned.csv")
