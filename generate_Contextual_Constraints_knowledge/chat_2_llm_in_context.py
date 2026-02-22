from openai import OpenAI
from preprocessing_datasets import source_data
from tqdm import tqdm
#一次输入的最大长度
N = 200000
START = "[START]"
END = "[END]"
max_retries =3
def long_prompt_chunker(input_str):
    idx = 0
    length = len(input_str)
    yield START
    while idx < length:
        yield input_str[idx:idx + N]
        idx += N
    yield END

def do_context_chat(system_promot,all_user_input):
    messages = [
        {"role": "system", "content": system_promot},
    ]
    client = OpenAI(api_key=source_data.get_api_key(),
            base_url="https://api.hunyuan.cloud.tencent.com/v1",)
    prompt_generator = long_prompt_chunker(all_user_input)
    tqdm.write(f"共{int(len(all_user_input)/N)+3}轮对话")
    ans = ""
    for prompt in prompt_generator:
        messages.append({"role": "user", "content": prompt})
        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(model="hunyuan-large-longcontext", messages=messages)
                answer = completion.choices[0].message.content
                if (prompt == END):
                    ans = answer
                messages.append({"role": "assistant", "content": answer})
                break
            except Exception as e:
                tqdm.write(f"Attempt {attempt + 1} failed: {e} prompt: {prompt}")
                if attempt == max_retries - 1:
                    tqdm.write(f"All attempts failed, prompt: {prompt}")


    return ans

def do_context_chat_(all_user_input:list):
    messages = [
        {"role": "system", "content": "You are a good assistant to reading, understanding and summarizing.."},
    ]
    client = OpenAI(
            # api_key=source_data.get_api_key(),
            api_key="sk-9540d9dc36c8414b9996f7ff373bfda2",
            # base_url="https://api.hunyuan.cloud.tencent.com/v1"
            base_url="https://api.deepseek.com"
            ,)
    ans = ""
    for prompt in all_user_input:
        messages.append({"role": "user", "content": prompt})
        for attempt in range(max_retries):
            try:

                # completion = client.chat.completions.create(model="hunyuan-turbos-latest", messages=messages)
                completion = client.chat.completions.create(model="deepseek-chat", messages=messages)
                answer = completion.choices[0].message.content
                if (prompt == all_user_input[-1]):
                    ans = answer
                messages.append({"role": "assistant", "content": answer})
                break
            except Exception as e:
                tqdm.write(f"Attempt {attempt + 1} failed: {e} prompt: {prompt}")
                if attempt == max_retries - 1:
                    raise ValueError(f"llm request error:{e}")
    return ans