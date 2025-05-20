import os
import json
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from itertools import combinations
import fastparquet
import matplotlib.pyplot as plt

# ✅ 设置字体为 WenQuanYi Zen Hei
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

# --------------------
# 配置路径
# --------------------
data_folder = "/work/share/acf6pa03fy/liyanjie/DATA/30g_data_new"
catalog_path = "/work/home/liyanjie/zzz/product2_catalog.json"
output_folder = "/work/home/liyanjie/zzz/time1_series_analysis"
os.makedirs(output_folder, exist_ok=True)

# --------------------
# 加载商品目录：构建 ID ➝ 大类 映射
# --------------------
with open(catalog_path, 'r') as f:
    catalog = json.load(f)

id_to_category = {
    int(p['id']): p.get('major_category')
    for p in catalog['products']
    if p.get('major_category')
}

# --------------------
# 提取所有订单中的时间 & 类别信息
# --------------------
records = []
user_order_seq = defaultdict(list)  # 用户 ID ➝ [（时间，类别列表）]

parquet_files = [f for f in os.listdir(data_folder) if f.endswith('.parquet')]

print("📂 正在提取用户订单信息...")
for file in tqdm(parquet_files):
    try:
        df = fastparquet.ParquetFile(os.path.join(data_folder, file)).to_pandas(columns=["id", "purchase_history"])
    except Exception as e:
        print(f"❌ 跳过文件 {file}: {e}")
        continue

    for row in df.itertuples():
        user_id = row.id
        try:
            record = json.loads(row.purchase_history)
            purchase_date = pd.to_datetime(record.get("purchase_date"))
            items = record.get("items", [])

            category_set = set()
            for item in items:
                pid = item.get("id")
                cat = id_to_category.get(pid)
                if cat:
                    category_set.add(cat)

            if category_set and purchase_date:
                records.append({
                    "user_id": user_id,
                    "purchase_date": purchase_date,
                    "categories": list(category_set)
                })
                user_order_seq[user_id].append((purchase_date, list(category_set)))
        except Exception:
            continue

df_orders = pd.DataFrame(records)
df_orders["year"] = df_orders["purchase_date"].dt.year
df_orders["month"] = df_orders["purchase_date"].dt.month
df_orders["quarter"] = df_orders["purchase_date"].dt.to_period("Q").astype(str)
df_orders["weekday"] = df_orders["purchase_date"].dt.day_name()

# --------------------
# 可视化 ① 总体购物行为（按月/季度/星期）
# --------------------
plt.figure(figsize=(10, 5))
df_orders["month"].value_counts().sort_index().plot(kind="bar", color="skyblue")
plt.title("📅 月度购物次数")
plt.xlabel("月份")
plt.ylabel("订单数")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "monthly_orders.png"))
plt.close()

plt.figure(figsize=(10, 5))
df_orders["quarter"].value_counts().sort_index().plot(kind="bar", color="orange")
plt.title("📅 季度购物次数")
plt.xlabel("季度")
plt.ylabel("订单数")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "quarterly_orders.png"))
plt.close()

plt.figure(figsize=(10, 5))
df_orders["weekday"].value_counts()[["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]].plot(kind="bar", color="lightgreen")
plt.title("📅 每周购物分布")
plt.xlabel("星期")
plt.ylabel("订单数")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "weekly_orders.png"))
plt.close()

# --------------------
# 可视化 ② 不同类别的时间趋势（按月）
# --------------------
cat_months = []

for _, row in df_orders.iterrows():
    month = row["month"]
    for cat in row["categories"]:
        cat_months.append((cat, month))

df_cm = pd.DataFrame(cat_months, columns=["category", "month"])
pivot = df_cm.groupby(["month", "category"]).size().unstack(fill_value=0)

pivot.plot(figsize=(12, 6), marker='o')
plt.title("📊 不同商品大类的月度购买趋势")
plt.xlabel("月份")
plt.ylabel("购买次数")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "category_month_trends.png"))
plt.close()

# --------------------
# ③ 时序模式分析：“先购买 A，后购买 B”
# --------------------
sequence_pairs = Counter()

for user_id, orders in user_order_seq.items():
    sorted_orders = sorted(orders, key=lambda x: x[0])  # 按时间排序
    seen = set()
    for _, cats in sorted_orders:
        for a in seen:
            for b in cats:
                if a != b:
                    sequence_pairs[(a, b)] += 1
        seen.update(cats)

df_seq = pd.DataFrame([
    {"from": a, "to": b, "count": c}
    for (a, b), c in sequence_pairs.items() if c >= 20
]).sort_values("count", ascending=False)

df_seq.to_csv(os.path.join(output_folder, "sequential_category_pairs.csv"), index=False)

# 可视化：前 10 个“先A后B”关系
plt.figure(figsize=(12, 6))
top_seq = df_seq.head(10)
labels = [f"{row['from']} → {row['to']}" for _, row in top_seq.iterrows()]
plt.barh(labels, top_seq["count"], color="plum")
plt.xlabel("频次")
plt.title("⏱️ Top 10 时序购买关系：先A后B")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "sequential_category_pairs.png"))
plt.close()

print("✅ 时间序列模式分析完成，结果保存在：", output_folder)
