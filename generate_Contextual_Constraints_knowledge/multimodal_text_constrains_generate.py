import sys
sys.path.append(".")
from preprocessing_datasets import source_data
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key=source_data.get_api_key(),
                base_url="https://api.hunyuan.cloud.tencent.com/v1", )
dataset_name="WN18RR_IMG"
with open("./generate_Contextual_Constraints_knowledge/knowledges/prompts/multimodal_text_constrains_prompt.txt",'r', encoding='utf-8') as f:
    prompt =f.read()


entityId_2_name = source_data.get_map_of_ids_to_attributes("ent", dataset_name, "name", "utf-8")
ent_desc = source_data.get_map_of_ids_to_attributes("ent", dataset_name, "desc", "utf-8")
ent_img_desc = source_data.get_map_of_ids_to_attributes("ent", dataset_name, "img_desc", "utf-8")
rel_2_desc = source_data.get_map_of_ids_to_attributes("rel", dataset_name, "desc", "utf-8")

relId_2_name = source_data.get_map_of_ids_to_attributes("rel", dataset_name, "name", "utf-8")
test_set = source_data.get_triples(dataset_name, "test")


target_path  = f"./generate_Contextual_Constraints_knowledge/knowledges/{dataset_name}/multimodal_text_constrains_2.txt"
with tqdm(total=len(test_set)) as pbar:
    for triple in test_set:
        head = f"[{entityId_2_name[triple[0]]}]"
        if triple[0] in ent_desc:
            head_desc = ent_desc[triple[0]]
        else:
            head_desc = head
        hed_img_desc = ent_img_desc[triple[0]]
        if dataset_name=="WN18RR":
            rel_id = triple[2]
        else:
            rel_id = triple[1]
        rel = f"[{relId_2_name[rel_id]}]"
        rel_desc = rel_2_desc[rel_id]
        constrain_id = triple[0]+":"+rel_id
        intput=f"\nTriple: ({head},{rel},?)\n Head Entity Text Description:{head_desc}\n Head Entity Image Description:{hed_img_desc} \n " \
               f"Relationship Description:{rel_desc} "
        cur_prompt =prompt+intput
        try:
            completion = client.chat.completions.create(model="hunyuan-turbos-latest",messages=[
                    {"role": "system",
                     "content": [{"type": "text",
                                  "text": "You are a good assistant to reading, understanding and summarizing.."}]},
                    {"role": "user",
                     "content": [
                         {"type": "text", "text": cur_prompt}]}],)
            answer = completion.choices[0].message.content
            with open(target_path, 'a', encoding="utf-8") as file:
                file.write(
                    f"{constrain_id}\t{source_data.remove_newlines(answer)}\n")
            pbar.update(1)
        except Exception as e:
            tqdm.write(f"failed: {e} prompt_len: {len(prompt)} entity_name={entityId_2_name[triple[0]]},rel={relId_2_name[rel_id]}")





