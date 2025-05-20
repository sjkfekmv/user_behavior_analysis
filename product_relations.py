import os
import json
import pandas as pd
from tqdm import tqdm
from mlxtend.frequent_patterns import fpgrowth, association_rules
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
catalog_path = "/work/home/liyanjie/zzz/product2_catalog.json"  # 商品目录，包含 major_category 字段
output_folder = "/work/home/liyanjie/zzz/product_relations"
os.makedirs(output_folder, exist_ok=True)

# --------------------
# 加载商品目录并构建 ID ➝ 大类 映射
# --------------------
with open(catalog_path, 'r', encoding='utf-8') as f:
    product_catalog = json.load(f)

product_to_category = {
    int(p['id']): p.get('major_category')
    for p in product_catalog['products']
    if p.get('major_category') is not None
}
print(f"📦 商品目录中大类映射的商品数: {len(product_to_category)}")
print(f"🎯 示例商品 ID 映射（前5个）: {list(product_to_category.items())[:5]}")

# --------------------
# 构建购物篮
# --------------------
category_baskets = []
unknown_ids = set()
parquet_files = [f for f in os.listdir(data_folder) if f.endswith('.parquet')]

print("📂 正在读取 parquet 数据并构建购物篮...")
for file in tqdm(parquet_files):
    file_path = os.path.join(data_folder, file)
    try:
        pf = fastparquet.ParquetFile(file_path)
        df = pf.to_pandas(columns=['purchase_history'])
    except Exception as e:
        print(f"❌ 跳过文件 {file}: {e}")
        continue

    for entry in df['purchase_history'].dropna():
        try:
            record = json.loads(entry)
            items = record.get('items', [])
            category_list = []
            for item in items:
                product_id = item.get('id')
                category = product_to_category.get(product_id)
                if category:
                    category_list.append(category)
                else:
                    unknown_ids.add(product_id)

            if not category_list:
                print(f"⚠️ 无法识别商品 ID: {[item.get('id') for item in items]}")

            if len(set(category_list)) >= 2:
                category_baskets.append(sorted(set(category_list)))
        except Exception:
            continue

print(f"✅ 共构建 {len(category_baskets)} 个有效购物篮")
if unknown_ids:
    print("⚠️ 找不到类别的商品 ID（最多展示20个）:", list(unknown_ids)[:20])

if not category_baskets:
    raise ValueError("❌ 没有可用购物篮，无法进行关联规则挖掘。")

# --------------------
# One-hot 编码 + 挖掘频繁项集
# --------------------
print("🔍 开始进行频繁项集挖掘...")
te = TransactionEncoder()
te_ary = te.fit(category_baskets).transform(category_baskets, sparse=True)
basket_df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)

frequent_itemsets = fpgrowth(basket_df, min_support=0.02, use_colnames=True)
if frequent_itemsets.empty:
    raise ValueError("❌ 未挖掘到任何频繁项集，请调低支持度或检查数据量。")

rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)
rules.to_csv(os.path.join(output_folder, 'all_rules.csv'), index=False)

# --------------------
# 关注包含“电子产品”的规则
# --------------------
electronics_rules = rules[rules['antecedents'].apply(lambda x: any('电子' in i for i in x)) |
                          rules['consequents'].apply(lambda x: any('电子' in i for i in x))]
electronics_rules.to_csv(os.path.join(output_folder, 'rules_with_electronics.csv'), index=False)

# --------------------
# 可视化前 10 条提升度最高规则
# --------------------
print("📊 正在生成可视化图表...")
top_lift = rules.sort_values('lift', ascending=False).head(10)
if not top_lift.empty:
    plt.figure(figsize=(12, 6))
    labels = [f"{', '.join(a)} => {', '.join(c)}"
              for a, c in zip(top_lift['antecedents'], top_lift['consequents'])]
    plt.barh(labels, top_lift['lift'], color='skyblue')
    plt.xlabel('Lift')
    plt.title('Top 10 Association Rules by Lift')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'top10_lift_rules.png'))
    plt.close()
else:
    print("⚠️ 没有满足可视化条件的规则。")

print("✅ 任务完成！已保存到：", output_folder)
