import os
import pytz
import pandas as pd

from tqdm import tqdm
from typing import List
from datetime import datetime
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 定义问答对模型
class QaPair(BaseModel):
    instruction: str = Field(description='问题的内容')
    output: str = Field(description='问题的回答')


class QaPairs(BaseModel):
    qas: List[QaPair] = Field(description='问答对列表')


# 拆分文档的函数
def split_document(filepath, chunk_size: int, chunk_overlap: int) -> list[Document]:
    """
        Splits the document at the given filepath into chunks.

        Args:
            filepath (str): The path to the document file.
            chunk_size (int): The size of each chunk.
            chunk_overlap (int): The overlap between chunks.

        Returns:
            List[Document]: A list of document chunks.
        """
    loader = UnstructuredFileLoader(filepath)
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    documents = loader.load_and_split(text_spliter)
    return documents


# 创建链的函数
def create_chain(model_name: str, temperature: float, max_tokens: int, openai_api_base: str, openai_api_key: str):

    QA_PAIRS_SYSTEM_PROMPT = """
    <Context></Context> 标记中的内容是腾讯云`容器服务TKE`和`容器服务EKS（TKE-Serverless）`和`容器镜像服务TCR`的官网文档（包含但不限于产品介绍、购买指南、用户指南、常见问题等内容），学习和分析它，并按要求整理结果并返回。
    要求如下：
    1. 学习文本内容，提取内容中的问题和每个问题的答案。
    2. 答案需详细完整，尽可能保留原文描述。
    3. 答案可以包含普通文字、链接、代码、表格、公示、媒体链接等 Markdown 元素。
    4. 最多不提出三十个问题。
    """

    QA_PAIRS_HUMAN_PROMPT = """
    <Context>
    {text}
    </Context>

    请按以下格式整理问题和答案，并确保返回有效的 JSON 格式：
    [
        {{"instruction": "问题1","output":"答案1"}},
        {{"instruction": "问题2","output":"答案2"}}
    ]
    """

    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", QA_PAIRS_SYSTEM_PROMPT),
        ("human", QA_PAIRS_HUMAN_PROMPT)
    ])

    llm = ChatOpenAI(model=model_name,
                     temperature=temperature,
                     max_tokens=max_tokens,
                     base_url=openai_api_base,
                     api_key=openai_api_key)

    chain = prompt | llm.with_structured_output(QaPairs)

    return chain


# 将数据保存到 Excel 文件
def save_to_excel_with_filename(data, output_directory, filename):
    try:
        df = pd.DataFrame(data)
        output_file = os.path.join(output_directory, f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)

        # 获取当前北京时间并打印日志
        beijing_tz = pytz.timezone('Asia/Shanghai')
        current_time = datetime.now(beijing_tz).strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n ✅ Data successfully written to {output_file} at {current_time} (Beijing Time)")

    except Exception as e:
        print(f"\n ❌ Error saving to Excel: {e}")


# 将所有 Excel 文件的内容汇总到一个 Excel 文件中
def consolidate_excels_to_excel(output_directory, final_output_dataset):
    try:
        all_data = []
        for file in os.listdir(output_directory):
            if file.endswith('.xlsx') and not file.startswith('sft_dataset_'):
                file_path = os.path.join(output_directory, file)
                df = pd.read_excel(file_path)
                all_data.append(df)

        final_df = pd.concat(all_data, ignore_index=True)
        with pd.ExcelWriter(final_output_dataset, engine='openpyxl') as writer:
            final_df.to_excel(writer, index=False)

        print(f"\n ✅ Data successfully consolidated to {final_output_dataset}")
    except Exception as e:
        print(f"\n ❌ Error consolidating to Excel: {e}")


# 处理输入目录中的文件
def process_files(input_directory, chain, chunk_size: int, chunk_overlap: int, output_directory: str):
    data = []
    for root, dirs, files in os.walk(input_directory):
        for filename in files:
            if filename.startswith('.'):
                continue  # 跳过以点开头的文件
            file_path = os.path.join(root, filename)
            print(f"\n 🚀 Start Process file: {file_path}")
            try:
                documents = split_document(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            except Exception as e:
                print(f"\nError processing file {file_path}: {e}")
                continue  # 跳过到下一个文件

            if not documents:
                print(f"\nNo documents found in {file_path}")
                continue  # 如果没有文档，跳过

            bar = tqdm(total=len(documents))
            for doc in documents:
                bar.update(1)
                try:
                    out = chain.invoke({'text': doc.page_content})
                    if isinstance(out, QaPairs):  # 检查输出类型
                        for qa_pair in out.qas:
                            data.append({
                                'instruction': qa_pair.instruction,
                                'output': qa_pair.output,
                            })
                    else:
                        print(f"\nUnexpected output format for document: {doc.page_content}")
                except Exception as e:
                    print(f"\nError processing output for document: {doc.page_content}, error: {e}")
            save_to_excel_with_filename(data, output_directory, filename)
            data = []  # 清空数据列表
            bar.close()
    return data


# 用 LLM 读取文档，并输出 Excel 数据集
if __name__ == '__main__':
    # 数据输入目录
    input_directory = "/Users/zhangyuehua/Desktop/TKE文档"
    # 数据集输出目录
    output_directory = "/Users/zhangyuehua/Desktop/dataset"
    # LLM 模型名称
    model_name = "Qwen2.5-72B-Instruct"
    # LLM temperature 参数
    temperature = 0.7
    max_tokens = 32768
    # LLM 地址（这里使用 Xinference 作为推理）
    openai_api_base = "http://101.33.197.124:9997/v1"
    # LLM API 密钥
    openai_api_key = "EMPTY"
    # 数据文件切片大小
    chunk_size = 4096
    # 数据文件切片重叠大小
    chunk_overlap = 768

    # 获取当前时间并格式化为字符串
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_output_dataset = os.path.join(output_directory, f"sft_dataset_{current_time}.xlsx")

    chain = create_chain(model_name=model_name,
                         temperature=temperature,
                         max_tokens=max_tokens,
                         openai_api_base=openai_api_base,
                         openai_api_key=openai_api_key)

    # 处理文件并获取数据
    data = process_files(input_directory, chain, chunk_size=chunk_size, chunk_overlap=chunk_overlap, output_directory=output_directory)

    # 将所有 Excel 文件的内容汇总到一个最终的 Excel 文件中
    consolidate_excels_to_excel(output_directory, final_output_dataset)

    print("Done!")
