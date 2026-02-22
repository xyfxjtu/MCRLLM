import sys
sys.path.append(".")
from preprocessing_datasets import source_data
from tqdm import tqdm

from openai import OpenAI

client = OpenAI(api_key=source_data.get_api_key(),
                base_url="https://api.hunyuan.cloud.tencent.com/v1", )
dataset_name="WN18RR_IMG"
with open("./generate_Contextual_Constraints_knowledge/knowledges/prompts/graph_context_constraints_prompt.txt",'r', encoding='utf-8') as f:
    prompt =f.read()
config = source_data.get_config_by_dataset_name(dataset_name)
train_path = config["train"]
#下面这个entityid_2_name变量里面取index报错了,应该更换ent_id是这个报错了
entityId_2_name = source_data.get_map_of_ids_to_attributes("ent", dataset_name, "name", "utf-8")

relId_2_name = source_data.get_map_of_ids_to_attributes("rel", dataset_name, "name", "utf-8")
train_set = source_data.get_triples(dataset_name, "train")
#下面的ent_id_list需要看下什么情况，应该存储的是数字
ent_id_list = source_data.get_id_list(dataset_name, "ent")

target_path  = f"./generate_Contextual_Constraints_knowledge/knowledges/{dataset_name}/graph_context_constraints.txt"
with tqdm(total=len(ent_id_list)) as pbar:
    for ent_id in ent_id_list:
        pbar.set_description(f"processing id:{ent_id},rel_name:{entityId_2_name[ent_id]}")
        as_tail="As tail:\n"
        as_head="As head:\n"
        for triple in train_set:
            if triple[0] == ent_id:
                head = entityId_2_name[triple[0]]
                tail = entityId_2_name[triple[2]]
                rel = relId_2_name[triple[1]]
                as_head+=f"{head},{rel},{tail}\n"
            if triple[2] == ent_id:
                head = entityId_2_name[triple[0]]
                tail = entityId_2_name[triple[2]]
                rel = relId_2_name[triple[1]]
                as_tail+=f"{head},{rel},{tail}\n"
        cur_prompt =prompt+as_head+as_tail
        try:
            completion = client.chat.completions.create(model="hunyuan-standard-256K",messages=[
                    {"role": "system",
                     "content": [{"type": "text",
                                  "text": "You are a good assistant to reading, understanding and summarizing.."}]},
                    {"role": "user",
                     "content": [
                         {"type": "text", "text": cur_prompt}]}],)

            answer = completion.choices[0].message.content
            with open(target_path, 'a', encoding="utf-8") as file:
                file.write(
                    f"{ent_id}\t{source_data.remove_newlines(answer)}\n")
            pbar.update(1)
        except Exception as e:
            tqdm.write(f"failed: {e} prompt_len: {len(prompt)} entity_name={entityId_2_name[ent_id]}")





