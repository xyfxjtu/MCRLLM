import sys
sys.path.append("..")
import source_data
from openai import OpenAI
# dataset_name="WN18RR"
dataset_name="WN18RR_IMG"
'''
关系文本对齐处理wn9数据集
方法见：Wei Y, Huang Q, Zhang Y, et al. KICGPT: Large Language Model with Knowledge in Context for Knowledge Graph Completion[C]
//The 2023 Conference on Empirical Methods in Natural Language Processing.
fb15k-237已处理，见 https://github.com/WEIYanbin1999/KICGPT
'''
#这里需要确定对数据集进行关系文本对齐用的是哪一个大模型的apikey
client = OpenAI(
       api_key=source_data.get_api_key(),
       base_url="https://api.hunyuan.cloud.tencent.com/v1",
    #    base_url="https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
    )
prompt ="You are a good assistant to reading, understanding and summarizing.\n {}\n" \
        "In above examples, What do you think \"{}\" mean? Summarize and descript its meaning using the format:" \
        "If the example shows something [H] {} something [T], it means [H] is [mask] of [T]." \
        "Fill the mask and the statement should be Clear, complete, and logical."
config = source_data.get_config_by_dataset_name(dataset_name)

train_path = config["train"]
entityId_2_name = source_data.get_map_of_ids_to_attributes("ent", dataset_name, "name", "utf-8")
relId_2_name = source_data.get_map_of_ids_to_attributes("rel", dataset_name, "name", "utf-8")
train_set = source_data.get_triples(dataset_name, "train")
rel_ids = source_data.get_id_list(dataset_name, "rel")
target_path = source_data.get_config_by_dataset_name(dataset_name)["rel_desc"]
for rel_id in rel_ids:
    example = ""
    for triple in train_set:
        if triple[2]==rel_id:
            head = entityId_2_name[triple[0]]
            rel = relId_2_name[rel_id]
            tail = entityId_2_name[triple[1]]
            tip = head+" "+rel +" "+tail+"\n"
            example+=tip
    cur_prompt =prompt.format(example,relId_2_name[rel_id],relId_2_name[rel_id])
    print(cur_prompt)
    completion = client.chat.completions.create(
        model="hunyuan-large",
        messages=[
            {"role": "system",
             "content": [{"type": "text",
                          "text": "You are a good assistant to reading, understanding and summarizing."}]},
            {"role": "user",
             "content": [
                 {"type": "text", "text": cur_prompt}]}],
        modalities=["text"],
        stream=True
    )
    rel_desc = ""
    for chunk in completion:
        if chunk.choices:
            if chunk.choices[0].delta.content is not None:
                rel_desc += chunk.choices[0].delta.content
    with open(target_path, 'a', encoding="utf-8") as file:
        # 用 tab 分隔 id 和 desc，最后添加换行符
        file.write(f"{rel_id}\t{source_data.remove_newlines(rel_desc)}\n")


