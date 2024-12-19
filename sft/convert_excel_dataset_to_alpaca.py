import os
from datetime import datetime

import pandas as pd
import json


def convert_excel_to_alpaca_format(excel_path, output_path):
    """
    将Excel文件转换为Alpaca数据集格式的JSON文件。

    参数:
    excel_path (str): 输入的Excel文件路径。
    output_path (str): 输出的JSON文件路径。
    """
    # 读取Excel文件
    df = pd.read_excel(excel_path)

    # 初始化一个空列表来存储转换后的数据
    alpaca_data = []

    # 遍历每一行，将其转换为所需的格式
    for index, row in df.iterrows():
        entry = {
            "instruction": row['instruction'],
            "input": "",  # 如果没有用户输入，可以留空
            "output": row['output'],
            "system": "",  # 如果没有系统提示词，可以留空
            "history": []  # 如果没有历史记录，可以留空
        }
        alpaca_data.append(entry)

    # 将数据转换为JSON格式并保存到文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=4)

    print(f" ✅ 转换完成，数据已保存到{output_path}")


# 把 Excel 数据集转换成 Alpaca 格式
if __name__ == '__main__':
    # 数据输入文件
    input_dataset = "/Users/zhangyuehua/Desktop/tke_sft_dataset_202412.xlsx"
    # 数据集输出目录
    output_directory = "/Users/zhangyuehua/Desktop/"

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_directory, f"alpaca_data_{current_time}.json")
    convert_excel_to_alpaca_format(excel_path=input_dataset, output_path=output_path)
