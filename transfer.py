import json

# 小类 ➝ 大类映射（请根据你之前规则复制完整）
subcategory_to_major = {
     # 电子产品
    "智能手机": "电子产品", "笔记本电脑": "电子产品", "平板电脑": "电子产品",
    "智能手表": "电子产品", "耳机": "电子产品", "音响": "电子产品",
    "相机": "电子产品", "摄像机": "电子产品", "游戏机": "电子产品",

    # 服装
    "上衣": "服装", "裤子": "服装", "裙子": "服装", "内衣": "服装",
    "鞋子": "服装", "帽子": "服装", "手套": "服装", "围巾": "服装", "外套": "服装",

    # 食品
    "零食": "食品", "饮料": "食品", "调味品": "食品", "米面": "食品",
    "水产": "食品", "肉类": "食品", "蛋奶": "食品", "水果": "食品", "蔬菜": "食品",

    # 家居
    "家具": "家居", "床上用品": "家居", "厨具": "家居", "卫浴用品": "家居",

    # 办公
    "文具": "办公", "办公用品": "办公",

    # 运动户外
    "健身器材": "运动户外", "户外装备": "运动户外",

    # 玩具
    "玩具": "玩具", "模型": "玩具", "益智玩具": "玩具",

    # 母婴
    "婴儿用品": "母婴", "儿童课外读物": "母婴",

    # 汽车用品
    "车载电子": "汽车用品", "汽车装饰": "汽车用品"
}

# 原始目录路径
old_path = "/path/to/old_product_catalog.json"
new_path = "/path/to/product_catalog_with_major.json"

with open(old_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 添加 major_category 字段
for p in data["products"]:
    p["major_category"] = subcategory_to_major.get(p["category"], "其他")

# 保存新文件
with open(new_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ 新文件已生成：", new_path)
