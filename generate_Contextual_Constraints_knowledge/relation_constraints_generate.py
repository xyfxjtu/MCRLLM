import sys
sys.path.append('.')
# sys.path.append("..")
from preprocessing_datasets import source_data
import chat_2_llm_in_context
from tqdm import tqdm
import sys
dataset_name="WN18RR_IMG"

with open("./generate_Contextual_Constraints_knowledge/knowledges/prompts/relation_constraints_prompt.txt",'r', encoding='utf-8') as f:
    prompt =f.read()

config = source_data.get_config_by_dataset_name(dataset_name)

train_path = config["train"]
entityId_2_name = source_data.get_map_of_ids_to_attributes("ent", dataset_name, "name", "utf-8")


relId_2_name = source_data.get_map_of_ids_to_attributes("rel", dataset_name, "name", "utf-8")
train_set = source_data.get_triples(dataset_name, "train")
rel_ids = source_data.get_id_list(dataset_name, "rel")

rel_desc = source_data.get_map_of_ids_to_attributes("rel", dataset_name, "desc", "utf-8")


target_path  = f"./generate_Contextual_Constraints_knowledge/knowledges/{dataset_name}/relation_constraints.txt"

with tqdm(total=len(rel_ids)) as pbar:
    for rel_id in rel_ids:
        pbar.set_description(f"processing id:{rel_id},rel_name:{relId_2_name[rel_id]}")
        example = "\n"+f"Relationship Name: {relId_2_name[rel_id]}\n"+f"Relationship Description:{rel_desc[rel_id]}\n"+"Subject and Object:\n"
        for triple in train_set:
            if triple[1]==rel_id:
                head = entityId_2_name[triple[0]]
                tail = entityId_2_name[triple[2]]


                example+=f"{{Subject Entity [H]:{head}\n Object Entity [T]:{tail}}}\n,"


        rel_constrain = chat_2_llm_in_context.do_context_chat(prompt,example)

        with open(target_path, 'a', encoding="utf-8") as file:
            # 用 tab 分隔 id 和 desc，最后添加换行符
            file.write(f"{rel_id}\t{relId_2_name[rel_id]}\t{rel_desc[rel_id]}\t{source_data.remove_newlines(rel_constrain)}\n")
        pbar.update(1)