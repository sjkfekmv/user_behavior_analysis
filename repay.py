import os
import json
import pandas as pd
from tqdm import tqdm
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import fastparquet
import matplotlib.pyplot as plt

# ✅ 设置字体为 WenQuanYi Zen Hei
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

# --------------------
# 路径设置
# --------------------
data_folder = "/work/share/acf6pa03fy/liyanjie/DATA/30g_data_new"
catalog_path = "/work/home/liyanjie/zzz/product2_catalog.json"
output_folder = "/work/home/liyanjie/zzz/refund_analysis"
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
# 构建事务：商品类别 + 退款状态
# --------------------
transactions = []
unknown_ids = set()

parquet_files = [f for f in os.listdir(data_folder) if f.endswith('.parquet')]

print("📦 正在解析退款相关订单...")
for file in tqdm(parquet_files):
    try:
        df = fastparquet.ParquetFile(os.path.join(data_folder, file)).to_pandas(columns=['purchase_history'])
    except Exception as e:
        print(f"❌ 跳过文件 {file}: {e}")
        continue

    for entry in df['purchase_history'].dropna():
        try:
            record = json.loads(entry)
            items = record.get("items", [])
            payment_status = record.get("payment_status")
            if payment_status not in ["已退款", "部分退款"]:
                continue

            categories = set()
            for item in items:
                pid = item.get("id")
                cat = id_to_category.get(pid)
                if cat:
                    categories.add(cat)
                else:
                    unknown_ids.add(pid)

            if categories:
                basket = list(categories) + ["退款"]
                transactions.append(basket)
        except Exception:
            continue

print(f"✅ 收集到 {len(transactions)} 条退款事务")
if not transactions:
    raise ValueError("❌ 没有有效退款订单，分析无法继续。")

# --------------------
# One-hot 编码 + 频繁项集挖掘
# --------------------
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_trans = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df_trans, min_support=0.005, use_colnames=True)
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.4)

# --------------------
# 只保留 以“退款”为结果的规则
# --------------------
refund_rules = rules[rules['consequents'].apply(lambda x: '退款' in x)]
refund_rules.to_csv(os.path.join(output_folder, "refund_category_rules.csv"), index=False)

# --------------------
# 可视化：Top 10 提升度最大的规则
# --------------------
top_lift = refund_rules.sort_values("lift", ascending=False).head(10)
if not top_lift.empty:
    plt.figure(figsize=(12, 6))
    labels = [f"{', '.join(a)} => {', '.join(c)}"
              for a, c in zip(top_lift['antecedents'], top_lift['consequents'])]
    plt.barh(labels, top_lift['lift'], color='salmon')
    plt.xlabel("Lift")
    plt.title("Top 10 商品组合 ➝ 退款 的规则")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "top10_lift_rules_refund.png"))
    plt
