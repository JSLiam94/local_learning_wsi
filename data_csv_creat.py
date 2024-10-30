import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split

# 指定目录路径
base_dir = '/home/zhengjingyuan/Baseline-NB/data/selected_patches'

# 初始化类别字典
category_dict = {}
next_category_id = 0

# 初始化 CSV 数据
csv_data = []

# 遍历目录
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.jpg'):
            # 获取文件路径
            file_path = os.path.join(root, file)
            # 获取子文件夹名称
            subfolder = os.path.basename(root)
            # 解析类别名称
            category_name = subfolder.split('_')[0]
            # 获取或生成类别 ID
            if category_name not in category_dict:
                category_dict[category_name] = next_category_id
                next_category_id += 1
            category_id = category_dict[category_name]
            # 添加到 CSV 数据，暂时都标记为'tmp'标识符
            csv_data.append([subfolder+'/'+file, category_id, 'tmp'])

# 将数据转换为 DataFrame 方便处理
df = pd.DataFrame(csv_data, columns=['slide_id', 'label', 'type'])

# 使用 sklearn 的 train_test_split 进行数据集分割
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.3333, random_state=42)

# 更新 type 列
train_df['type'] = 'train'
val_df['type'] = 'valid'
test_df['type'] = 'test'

# 合并三个 DataFrame 并排序
final_df = pd.concat([train_df, val_df, test_df]).sort_values(by='slide_id')

# 写入 CSV 文件
csv_file_path = '/home/zhushenghao/data/JS/local_learning_wsi/alldata-2.csv'
final_df.to_csv(csv_file_path, index=False)

print(f"CSV file created at {csv_file_path}")