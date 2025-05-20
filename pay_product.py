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
# 设置路径
# --------------------
data_folder = "/work/share/acf6pa03fy/liyanjie/DATA/30g_data_new"
catalog_path = "/work/home/liyanjie/zzz/product2_catalog.json"
output_folder = "/work/home/liyanjie/zzz/payment_category_analysis"
os.makedirs(output_folder, exist_ok=True)

# --------------------
# 加载商品目录：构建商品 ID ➝ 大类、价格 映射
# --------------------
with open(catalog_path, 'r', encoding='utf-8') as f:
    product_catalog = json.load(f)

id_to_category = {}
id_to_price = {}

for p in product_catalog['products']:
    if 'major_category' in p and p['major_category']:
        id_to_category[int(p['id'])] = p['major_category']
    if 'price' in p:
        id_to_price[int(p['id'])] = float(p['price'])

# --------------------
# 购物篮构建：商品类别 + 支付方式联合事务
# --------------------
transactions = []
high_value_payment_methods = []
unknown_ids = set()
parquet_files = [f for f in os.listdir(data_folder) if f.endswith('.parquet')]

print("📂 正在解析数据...")
for file in tqdm(parquet_files):
    file_path = os.path.join(data_folder, file)
    try:
        df = fastparquet.ParquetFile(file_path).to_pandas(columns=['purchase_history'])
    except Exception as e:
        print(f"❌ 跳过文件 {file}: {e}")
        continue

    for entry in df['purchase_history'].dropna():
        try:
            record = json.loads(entry)
            items = record.get('items', [])
            payment_method = record.get('payment_method')

            categories = set()
            is_high_value = False

            for item in items:
                pid = item.get('id')
                category = id_to_category.get(pid)
                price = id_to_price.get(pid)
                if category:
                    categories.add(category)
                else:
                    unknown_ids.add(pid)
                if price and price > 5000:
                    is_high_value = True

            if payment_method and categories:
                transaction = list(categories) + [f"支付:{payment_method}"]
                transactions.append(transaction)

                if is_high_value:
                    high_value_payment_methods.append(payment_method)
        except Exception:
            continue

print(f"✅ 共构建 {len(transactions)} 条支付-商品类别事务")

# --------------------
# 频繁项集与关联规则分析
# --------------------
print("🔍 正在挖掘规则（使用 Apriori）...")
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_trans = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df_trans, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

rules.to_csv(os.path.join(output_folder, "payment_category_rules.csv"), index=False)

# --------------------
# 可视化：前10条 Lift 最大的规则
# --------------------
top_lift = rules.sort_values("lift", ascending=False).head(10)
if not top_lift.empty:
    plt.figure(figsize=(12, 6))
    labels = [f"{', '.join(a)} => {', '.join(c)}" for a, c in zip(top_lift['antecedents'], top_lift['consequents'])]
    plt.barh(labels, top_lift['lift'], color='coral')
    plt.xlabel("Lift")
    plt.title("Top 10 Rules: 商品类别 ➝ 支付方式")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "top10_lift_rules_payment.png"))
    plt.close()
else:
    print("⚠️ 没有可视化的规则。")

# --------------------
# 高价值商品偏好支付方式分析
# --------------------
print("📊 正在分析高价值商品的支付方式偏好...")
high_value_df = pd.Series(high_value_payment_methods).value_counts().reset_index()
high_value_df.columns = ["payment_method", "count"]
high_value_df.to_csv(os.path.join(output_folder, "high_value_payment_methods.csv"), index=False)

plt.figure(figsize=(8, 5))
plt.bar(high_value_df["payment_method"], high_value_df["count"], color="green")
plt.title("高价值商品（价格>5000）首选支付方式")
plt.xlabel("支付方式")
plt.ylabel("购买次数")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "high_value_payment_methods.png"))
plt.close()

print("✅ 分析完成！结果保存在：", output_folder)

