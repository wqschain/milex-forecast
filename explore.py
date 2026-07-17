import pandas as pd

df = pd.read_excel("sipri_data.xlsx", sheet_name="Share of GDP", header=5)

countries_wanted = ["United States of America", "Canada", "Mexico", "Brazil", "Argentina", "Colombia",
                     "United Kingdom", "Germany", "France", "Italy", "Poland", "Russia", "Ukraine",
                     "Israel", "Saudi Arabia", "Türkiye", "Iran", "China", "India", "Japan",
                     "Korea, South", "Korea, North", "Pakistan",
                     "Australia", "Indonesia", "Viet Nam", "South Africa", "Nigeria", "Egypt", "Algeria",
                     "Sweden", "Norway"]

df_filtered = df[df["Country"].isin(countries_wanted)]

year_columns = df_filtered.columns[2:]

for year in year_columns:
    df_filtered[year] = pd.to_numeric(df_filtered[year], errors="coerce")

df_long = df_filtered.melt(
    id_vars=["Country"],
    value_vars=year_columns,
    var_name="Year",
    value_name="Spending"
)

df_long["Year"] = pd.to_numeric(df_long["Year"])
df_long = df_long[df_long["Year"] >= 1980]



df_long = df_long.sort_values(["Country", "Year"])  
df_long["Spending"] = df_long.groupby("Country")["Spending"].transform(lambda group: group.interpolate())

df_long.to_csv("cleaned_data.csv", index=False)