import pandas as pd

from tqdm import tqdm
from typing import List
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class QuestionsList(BaseModel):
    questions: List[str] = Field(description='问题列表')


# 创建链的函数
def create_chain(model_name: str, temperature: float, max_tokens: int, openai_api_base: str, openai_api_key: str):
    prompt_template = """
    你是一个资深的语言专家和同义词转换器。下面有一个腾讯云和云原生行业相关的问答样本，你需要为给定的"问题"生成额外的两种不同问法，这些问法应与原问题在语义上等价或近似，但在措辞、结构、句式上有所不同。

    原问题：
    {instruction}
    
    答案：
    {output}

    请为该问题生成两个变体问法，每个问法应满足：
    - 保留问题的中心语义和领域背景。
    - 尽量不改变问题的原始答案所对应的知识点。
    - 使用不同的句式和措辞，尽可能多样化。

    请按以下格式整理问题，并确保返回有效的问题列表：
    ["原问题", "变体问题1", "变体问题2"]
    """

    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template)
    ])

    llm = ChatOpenAI(model=model_name,
                     temperature=temperature,
                     max_tokens=max_tokens,
                     base_url=openai_api_base,
                     api_key=openai_api_key)

    chain = prompt | llm.with_structured_output(QuestionsList)

    return chain


# 处理Excel文件并生成变体问法
def process_excel(input_excel_path: str, output_excel_path: str, chain):
    try:
        df = pd.read_excel(input_excel_path)
        if 'instruction' not in df.columns or 'output' not in df.columns:
            raise ValueError(" ❌ Input Excel must contain 'instruction' and 'output' columns")

        data = []
        bar = tqdm(total=len(df))
        for index, row in df.iterrows():
            bar.update(1)
            instruction = row['instruction']
            output = row['output']
            prompt = {
                'instruction': instruction,
                'output': output
            }
            try:
                response = chain.invoke(prompt)
                print(f"\n Response: {response}")
                if isinstance(response, QuestionsList):  # 检查输出类型
                    for question in response.questions:
                        data.append({
                            'instruction': question,
                            'output': output,
                        })
                else:
                    print(f"\n ❌ Unexpected output format for instruction: {instruction}")
                    data.append({
                        'instruction': instruction,
                        'output': output,
                    })
            except Exception as e:
                print(f"\n ❌ Error processing instruction: {instruction}, error: {e}")
        bar.close()

        # 保存结果到Excel
        result_df = pd.DataFrame(data)
        with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
            result_df.to_excel(writer, index=False)

        print(f"\n ✅ Data successfully written to {output_excel_path}")

    except Exception as e:
        print(f"\n ❌ Error processing Excel: {e}")


# 根据要求扩写指令
if __name__ == '__main__':
    # 输入Excel文件路径
    input_excel_path = "/Users/zhangyuehua/Desktop/tke_doc_sft_dataset_clean_20241219.xlsx"

    # 输出Excel文件路径
    output_excel_path = "/Users/zhangyuehua/Desktop/tke_doc_sft_dataset_clean_20241219_expand.xlsx"
    # LLM 模型名称
    model_name = "Qwen2.5-72B-Instruct"
    # LLM temperature 参数
    temperature = 0.7
    max_tokens = 32768
    # LLM 地址（这里使用 Xinference 作为推理）
    openai_api_base = "http://101.33.197.124:9997/v1"
    # LLM API 密钥
    openai_api_key = "EMPTY"

    chain = create_chain(model_name=model_name,
                         temperature=temperature,
                         max_tokens=max_tokens,
                         openai_api_base=openai_api_base,
                         openai_api_key=openai_api_key)

    # 处理Excel文件并生成变体问法
    process_excel(input_excel_path=input_excel_path,
                  output_excel_path=output_excel_path,
                  chain=chain)

    print("Done!")
