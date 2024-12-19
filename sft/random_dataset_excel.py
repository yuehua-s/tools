import pandas as pd
import random


def shuffle_excel_rows(input_file, output_file):
    try:
        # 读取Excel文件
        df = pd.read_excel(input_file)

        # 将DataFrame按行打乱
        df_shuffled = df.sample(frac=1, random_state=random.randint(0, 10000)).reset_index(drop=True)

        # 将打乱后的DataFrame写入新的Excel文件
        df_shuffled.to_excel(output_file, index=False)

        print(f"成功将 {input_file} 的内容按行随机打乱，并写入到 {output_file} 中。")
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    # 在这里定义输入文件和输出文件路径
    input_file = '/Users/zhangyuehua/Desktop/tke_doc_sft_dataset_clean_20241219_expand.xlsx'
    output_file = '/Users/zhangyuehua/Desktop/tke_doc_sft_dataset_clean_20241219_expand_finally.xlsx'

    shuffle_excel_rows(input_file, output_file)
    print("Done!")
