import json
import os
import base64
current_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(current_dir, 'configs.json')
with open(config_file_path, 'r') as config_file:
    configs = json.load(config_file)
def get_config_by_dataset_name(dataset_name):
    datasets = configs['datasets']
    for dataset in datasets:
        if dataset["name"]==dataset_name:
            return dataset
def get_map_of_ids_to_attributes(type,dataset_name,attribute,encoding):
    global file_path
    map = {}
    paths = get_config_by_dataset_name(dataset_name)
    if type == "ent":
        if attribute == "name":
            file_path =paths["ent_names"]
        elif attribute== "desc":
            file_path = paths["ent_desc"]
        elif attribute== "img_desc":
            file_path = paths["ent_imgdesc"]
    if type == "rel":
        if attribute == "name":
            file_path =paths["rel_names"]
        elif attribute== "desc":
            file_path = paths["rel_desc"]
    with open(file_path, 'r',encoding=encoding) as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:
                id_part = parts[0]
                desc_part = " ".join(parts[1:])
                map[id_part] = desc_part
    return map

def get_triples(dataset_name,type):
    """
    :param dataset_name: dataset_name
    :param type: "train"/"valid"/"test"
    :return: triples
    """
    triples=[]
    paths = get_config_by_dataset_name(dataset_name)
    target =paths[type]
    # print("target如下：\n")
    # print(target)
    with open(target, 'r',encoding="utf-8") as file:
        for line in file:
            triples.append(line.split())
    return triples
def get_id_list(dataset_name,type):
    """
    :param dataset_name: dataset_name
    :param type: "rel"/"ent"
    :return: list of ids
    """
    global target
    #这里获取的应该是fb15k-237的数据
    paths = get_config_by_dataset_name(dataset_name)
    ids =[]
    if type == "rel":
        target =paths["rel_ids"]
    elif type == "ent":
        target = paths["ent_ids"]
    with open(target, 'r', encoding='utf-8') as f:
        for line in f:
            ids.append(line.strip())
    return ids

def id_to_img_file_name(dataset_name,source):
    if dataset_name=="WN18RR_IMG":
        return source
    if dataset_name =="FB15K_237_IMG":
        return source.replace('/', '.')[1:]
def get_api_key():
    return configs["api_key"]
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
def remove_newlines(input_str):
    return input_str.replace('\n', '').replace('\r', '')

if __name__ == '__main__':
    print(get_map_of_ids_to_attributes("ent","WN18_IMG","name"))