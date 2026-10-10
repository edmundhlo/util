"""Generate sample datasets for trying the app."""
import numpy as np, pandas as pd
rng = np.random.default_rng(0)
n = 1500
df = pd.DataFrame({
    "customer_id": [f"C{i:05d}" for i in range(n)],
    "age": rng.integers(18, 80, n),
    "income": rng.lognormal(10.5, 0.5, n).round(2),
    "region": rng.choice(["North", "South", "East", "West"], n),
    "plan": rng.choice(["basic", "plus", "premium"], n, p=[.5, .3, .2]),
    "signup_date": pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.integers(0, 1500, n), "D"),
    "support_calls": rng.poisson(2, n),
})
df.loc[rng.random(n) < 0.05, "income"] = np.nan
logit = (-2 + 0.6 * df.support_calls - 0.00002 * df.income.fillna(40000)
         + (df.plan == "basic") * 1.0 + (df.region == "South") * 0.5 + 0.01 * (df.age - 50))
df["churned"] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "yes", "no")
df["monthly_spend"] = (20 + 0.0004 * df.income.fillna(40000) + df.plan.map({"basic": 0, "plus": 15, "premium": 40})
                       + rng.normal(0, 5, n)).round(2)
df.to_csv("sample_data/customers.csv", index=False)
df.drop(columns="churned").to_excel("sample_data/customers.xlsx", sheet_name="data", index=False)
print(df.head())
