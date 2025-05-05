import os
import sys
import pandas as pd
import pyarrow.parquet as pq
import json


def parquet_to_jsonl_pandas(parquet_path, jsonl_path):
    df = pd.read_parquet(parquet_path, engine='pyarrow')
    with open(jsonl_path, 'w', encoding='utf-8') as f_out:
        for record in df.to_dict(orient='records'):
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')


def parquet_to_jsonl_pyarrow(parquet_path, jsonl_path, batch_size=10000):
    parquet_file = pq.ParquetFile(parquet_path)
    with open(jsonl_path, 'w', encoding='utf-8') as f_out:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            df = batch.to_pandas()
            for record in df.to_dict(orient='records'):
                f_out.write(json.dumps(record, ensure_ascii=False) + '\n')


def main(parquet_path, jsonl_path, threshold_mb=500):
    file_size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
    if file_size_mb < threshold_mb:
        print(f"文件较小（{file_size_mb:.2f} MB），使用 pandas 处理。")
        parquet_to_jsonl_pandas(parquet_path, jsonl_path)
    else:
        print(f"文件较大（{file_size_mb:.2f} MB），使用 pyarrow 分批处理。")
        parquet_to_jsonl_pyarrow(parquet_path, jsonl_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        """
        python eval/parquet2jsonl.py ~/Desktop/数据集/benckmark/aime2024/train-00000-of-00001.parquet ~/Desktop/数据集/benckmark/aime2024/aime.jsonl
        """
        print("用法: python parquet2jsonl.py 输入.parquet 输出.jsonl")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
