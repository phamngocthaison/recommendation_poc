import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Đọc dữ liệu từ file CSV với encoding phù hợp
df = pd.read_csv('Online Retail.csv', encoding='utf-8-sig')

# Loại bỏ các dòng có giá trị Quantity âm (trả hàng)
df = df[df['Quantity'] > 0]
# Có thể lọc theo Country nếu cần, ví dụ:
df = df[df['Country'] == 'United Kingdom']

print(df.columns)
print(df.head())

# Tạo bảng pivot: hàng là các InvoiceNo, cột là Description, giá trị là Quantity
basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)


# Chuyển đổi số lượng sang giá trị nhị phân (1: có mua, 0: không mua)
def encode_units(x):
    return 1 if x >= 1 else 0


basket_encoded = basket.apply(lambda col: col.map(encode_units))
print(basket_encoded.head())

basket_encoded_bool = basket_encoded.astype(bool)

# Giảm ngưỡng min_support để có nhiều tập mục phổ biến hơn
frequent_itemsets = apriori(basket_encoded_bool, min_support=0.01, use_colnames=True)
print(frequent_itemsets.head())

# Khai thác luật kết hợp với ngưỡng confidence thấp hơn nếu cần (ví dụ: 0.2)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
rules = rules.sort_values(by='confidence', ascending=False)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())


# Hàm gợi ý: chuẩn hóa tên sản phẩm bằng cách loại bỏ khoảng trắng thừa và chuyển thành chữ thường
def recommend_products(product, rules_df, top_n=5):
    product_norm = product.strip().lower()
    # Lọc các luật mà antecedents chứa sản phẩm (chuẩn hóa tên)
    filtered_rules = rules_df[rules_df['antecedents'].apply(
        lambda x: product_norm in [p.strip().lower() for p in x]
    )]

    # Nếu không có luật nào phù hợp, trả về danh sách rỗng
    if filtered_rules.empty:
        return []

    # Kiểm tra nếu cột 'confidence' có tồn tại, nếu không, sắp xếp theo 'lift' (nếu có)
    if 'confidence' in filtered_rules.columns:
        filtered_rules = filtered_rules.sort_values(by='confidence', ascending=False)
    elif 'lift' in filtered_rules.columns:
        filtered_rules = filtered_rules.sort_values(by='lift', ascending=False)

    recommendations = set()
    for _, row in filtered_rules.iterrows():
        recs = [p.strip() for p in row['consequents']]
        recommendations = recommendations.union(recs)

    # Loại bỏ sản phẩm gốc nếu có
    recommendations.discard(product)
    return list(recommendations)[:top_n]


# Ví dụ gợi ý cho sản phẩm 'WHITE HANGING HEART T-LIGHT HOLDER'
product_input = 'WHITE HANGING HEART T-LIGHT HOLDER'
recommended = recommend_products(product_input, rules)
print("Các sản phẩm gợi ý cho", product_input, "là:", recommended)