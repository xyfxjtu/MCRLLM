import sys
sys.path.append("..")
import source_data
from openai import OpenAI
import os
import time
import random
from tqdm import tqdm

def fill_null(id,name,desc,target_path):
    full_prompt = "Suppose there is a set of images related to the entity [{}],the entity come from WordNet and the entity's description is[{}]. These images might come from a search engine. " \
                  "Please provide just one detailed description of what the most of these images might look like, in no more than 200 words." \
                  "You should not output any unnecessary characters other than words and spaces.And please answer in English".format(name, desc)
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="hunyuan-vision",
                messages=[
                    {"role": "system",
                     "content": [{"type": "text",
                                  "text": "You are an image and text recognition assistant."}]},
                    {"role": "user",
                     "content": [
                         {"type": "text", "text": full_prompt}]}],
                modalities=["text"],
                stream=True
            )
            ent_imgdesc = ""
            for chunk in completion:
                if chunk.choices:
                    if chunk.choices[0].delta.content is not None:
                        ent_imgdesc += chunk.choices[0].delta.content
            with open(target_path, 'a',encoding="utf-8") as file:
                # 用 tab 分隔 id 和 desc，最后添加换行符
                file.write(f"{id}\t{source_data.remove_newlines(ent_imgdesc)}\n")
            time.sleep(1)
            break
        except Exception as e:
            tqdm.write(f"Attempt {attempt + 1} failed: {e} entity id: {id} entity name: {name}")
            if attempt == max_retries - 1:
                tqdm.write(f"All attempts failed for entity id: {id} entity name: {name}, failed:{e}")

def do_call_llm():
    for data_set_name in data_set_names:
        finished_map = source_data.get_map_of_ids_to_attributes(type="ent", dataset_name=data_set_name, attribute="img_desc", encoding="utf-8")
        config = source_data.get_config_by_dataset_name(dataset_name=data_set_name)
        images_path = config["images"]
        entid2name_map = source_data.get_map_of_ids_to_attributes(type="ent", dataset_name=data_set_name, attribute="name", encoding="utf-8")
        entid2desc_map = source_data.get_map_of_ids_to_attributes(type="ent", dataset_name=data_set_name, attribute="desc", encoding="utf-8")
        target_path = config["ent_imgdesc"]
        with tqdm(total=len(entid2name_map)) as pbar:
            for id,name in entid2name_map.items():
                pbar.set_description(f"processing id:{id},name:{name}")
                if id in finished_map:
                    pbar.update(1)
                    continue
                print("data_set_name如下：\n")
                print(data_set_name)
                target_images_path = os.path.join(images_path, source_data.id_to_img_file_name(data_set_name, "n"+id))
                print("target_images如下：\n")
                print(target_images_path)
                print("********************")
                desc = entid2desc_map.get("n"+id)
                if not os.path.exists(target_images_path):
                    tqdm.write(id+" "+ name+" 缺失了 imgs." )
                    # 利用大语言模型做缺失图片文本化处理
                    fill_null(id,name,desc,target_path)
                    pbar.update(1)
                    continue
                img_names = [f for f in os.listdir(target_images_path) if os.path.isfile(os.path.join(target_images_path, f))]
                print("img_names如下：\n")
                print(img_names)
                if len(img_names)>=max_imgs_of_entity:
                    img_names = random.sample(img_names, max_imgs_of_entity)
                image_parts = []
                for image_name in img_names:
                    image_base64 = source_data.image_to_base64(os.path.join(target_images_path, image_name))
                    print("image_base64如下：\n")
                    print(image_base64)  # 打印字符
                    image_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}})
                    prompt_cur = prompt.format(name,name)
                for attempt in range(max_retries):
                    try:
                        completion = client.chat.completions.create(
                            # model="hunyuan-lite-vision",
                            model="hunyuan-vision",
                            messages=[
                                {"role": "system",
                                 "content": [{"type": "text",
                                              "text": "You are an image recognition assistant, capable of easily identifying the relationship between text and images. Even abstract or implicit connections can be easily captured, and you can provide accurate and clear summaries."}]},
                                {"role": "user",
                                 "content": [
                                     *image_parts,
                                     {"type": "text", "text": prompt_cur}]}],
                            modalities=["text"],
                            stream=True
                        )
                        print("completion如下：\n",completion)
                        ent_imgdesc = ""
                        for chunk in completion:
                            if chunk.choices:
                                if chunk.choices[0].delta.content is not None:
                                    ent_imgdesc += chunk.choices[0].delta.content
                        if "我还未学习到如何回答这个问题的内容" in ent_imgdesc:
                            tqdm.write(f"entity id: {id} entity name: {name} 图片违规，看作图片缺失")
                            fill_null(id, name, desc, target_path)
                            pbar.update(1)
                            break
                        with open(target_path, 'a',encoding="utf-8") as file:
                            # 用 tab 分隔 id 和 desc，最后添加换行符
                            file.write(f"{id}\t{source_data.remove_newlines(ent_imgdesc)}\n")
                        pbar.update(1)
                        time.sleep(1)
                        break
                    except Exception as e:
                        tqdm.write(f"Attempt {attempt + 1} failed: {e} entity id: {id} entity name: {name}")
                        if attempt == max_retries - 1:
                            tqdm.write(f"All attempts failed for entity id: {id} entity name: {name}, failed:{e}")




'''
利用大语言模型生成多模态数据集中图片的描述文本。
输入：数据集实体的多张图片
输出：用不超过150个单词总结贴切地描述这些图片与实体的关系
'''

data_set_names = ["WN18RR_IMG"]
max_retries = 3  # 最大重试次数
max_imgs_of_entity = 5
client = OpenAI(
        api_key=source_data.get_api_key(),
        # base_url="https://open.hunyuan.tencent.com/openapi/v1/agent/chat/completions",
        base_url="https://api.hunyuan.cloud.tencent.com/v1",
        # base_url="https://open.hunyuan.tencent.com/openapi/v1/agent/chat/completions"
    )
prompt = "These are images retrieved from a search engine related to the entity [{}] " \
         "in a multimodal knowledge graph. While these images are related to the entity, " \
         "the strength of their association is random.Some images may not be related at all, " \
         "and some may even be duplicated. Please summarize the relationship between the entity [{}] " \
         "and these images in no more than 200 words. You do not need to summarize each image individually; " \
         "instead, choose the images or parts of images most closely related to the entity, " \
         "and provide a general summary of their strongest connections to the entity. " \
         "You may ignore duplicates or images you believe are unrelated to the entity.And please answer in English."

do_call_llm()