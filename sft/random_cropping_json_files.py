import json
import random


def is_alpaca_format(data):
    # 检查数据是否为列表
    if not isinstance(data, list):
        return False

    # 检查每个字典对象是否包含特定的键
    required_keys = {'instruction', 'output'}
    for entry in data:
        if not isinstance(entry, dict):
            return False
        if not required_keys.issubset(entry.keys()):
            return False

    return True


def is_sharegpt_format(data):
    # 检查数据是否为列表
    if not isinstance(data, list):
        return False

    # 检查每个字典对象是否包含特定的键
    for entry in data:
        if not isinstance(entry, dict):
            return False
        if 'conversations' not in entry or not isinstance(entry['conversations'], list):
            return False

    return True


def select_random_entries(input_file, output_file, N, format_type):
    try:
        # 读取原始JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 根据用户指定的格式进行校验
        if format_type == 'alpaca':
            if not is_alpaca_format(data):
                raise ValueError("输入数据格式不正确，必须为alpaca格式")
        elif format_type == 'sharegpt':
            if not is_sharegpt_format(data):
                raise ValueError("输入数据格式不正确，必须为sharegpt格式")
        else:
            raise ValueError("未知的格式类型，必须为'alpaca'或'sharegpt'")

        # 确保N不超过数据的总长度
        if N > len(data):
            raise ValueError("N不能超过数据的总长度")

        # 随机选择N个字典对象
        selected_entries = random.sample(data, N)

        # 将选择的字典对象写入新的JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(selected_entries, f, ensure_ascii=False, indent=4)

        print(f"成功从 {input_file} 中随机选择了 {N} 个字典对象，并写入到 {output_file} 中。数据格式为 {format_type}。")
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    # 在这里定义输入文件、输出文件、选择的数量和数据格式
    # input_file = '/Users/zhangyuehua/Desktop/alpaca_gpt4_data_zh.json'
    # output_file = '/Users/zhangyuehua/Desktop/alpaca_gpt4_data_zh_1w.json'
    # N = 8000
    # format_type = 'alpaca'

    input_file = '/Users/zhangyuehua/Desktop/glaive_toolcall.json'
    output_file = '/Users/zhangyuehua/Desktop/glaive_toolcall_en_4k.json'
    N = 4000
    format_type = 'sharegpt'

    select_random_entries(input_file, output_file, N, format_type)
    print("Done!")
