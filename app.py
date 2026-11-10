from pathlib import Path


import pandas as pd




ROOT = Path(__file__).resolve().parent


SEASON = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Autumn"}




def main():
   day = pd.read_csv(ROOT / "day.csv", parse_dates=["dteday"])
   hour = pd.read_csv(ROOT / "hour.csv", parse_dates=["dteday"])


   print("Data checks")
   for name, frame in (("day.csv", day), ("hour.csv", hour)):
       missing = int(frame.isna().sum().sum())
       dupes = int(frame.duplicated().sum())
       print(f"{name}: {len(frame)} rows, {frame.shape[1]} columns, {missing} missing, {dupes} duplicate rows")


   rows_per_date = hour.groupby("dteday").size()
   hourly_total = hour.groupby("dteday")["cnt"].sum()
   n_short = int((rows_per_date != 24).sum())
   n_differ = int(day.set_index("dteday")["cnt"].ne(hourly_total).sum())
   print(f"{n_short} dates have fewer than 24 hours")
   if n_differ:
       print(f"day.csv cnt differs from the hourly sum on {n_differ} dates")
   else:
       print("day.csv cnt equals the hourly sum on every date")


   print(" ")
   print("Task 1: rentals by season")
   print("1 Winter, 2 Spring, 3 Summer, 4 Autumn")
   summary = day.groupby("season")["cnt"].agg(["count", "sum", "mean", "median", "min", "max"])
   summary.index = summary.index.map(SEASON)
   print(summary.round(1))
   means = summary["mean"]
   print(f"highest mean: {means.idxmax()} ({means.max():.1f})")
   print(f"lowest mean: {means.idxmin()} ({means.min():.1f})")
   for season, row_index in day.groupby("season")["cnt"].idxmin().items():
       row = day.loc[row_index]
       recorded = int(rows_per_date.get(row["dteday"], 0))
       if recorded < 24:
           when = row["dteday"].date()
           print(f"{SEASON[season]} minimum is {int(row['cnt'])} on {when} ({recorded} of 24 hours recorded)")


   print("")
   print("Task 2: weekday versus weekend")
   print("weekday = codes 1-5, weekend = 0 and 6")
   hour["day_type"] = "weekend"
   hour.loc[hour["weekday"].between(1, 5), "day_type"] = "weekday"
   hourly_means = hour.groupby(["hr", "day_type"])["cnt"].mean().unstack()
   print(hourly_means.round(1))


   for kind in ("weekday", "weekend"):
       column = hourly_means[kind]
       above = []
       for hr, value in column.items():
           left = column.get((hr - 1) % 24)
           right = column.get((hr + 1) % 24)
           if pd.notna(left) and pd.notna(right) and value > left and value > right:
               above.append(hr)
       above.sort(key=lambda hr: column[hr], reverse=True)
       low = column.idxmin()
       highs = ", ".join(f"{hr} ({column[hr]:.1f})" for hr in above)
       print(f"{kind}: above both neighbors {highs}, lowest hour {int(low)} ({column[low]:.1f})")


   daily = hour.groupby(["dteday", "day_type"])["cnt"].sum()
   daily_means = daily.groupby("day_type").mean()
   print("Mean of summed hourly counts per date:")
   for kind in ("weekday", "weekend"):
       print(f"  {kind}: {daily_means[kind]:.1f}")


   print("")
   print("Task 3: weather")
   print(f"{'code':>4} {'days':>6} {'mean':>8}")
   for code in (1, 2, 3, 4):
       group = day.loc[day["weathersit"] == code, "cnt"]
       if group.empty:
           print(f"{code:>4} {0:>6}")
       else:
           print(f"{code:>4} {len(group):>6} {group.mean():>8.1f}")
   print("Pearson correlation with cnt:")
   corr = day[["temp", "atemp", "hum", "windspeed"]].corrwith(day["cnt"])
   for col, value in corr.items():
       print(f"  {col}: {value:.3f}")




if __name__ == "__main__":
   main()
