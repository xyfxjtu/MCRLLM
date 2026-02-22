import sys
sys.path.append("/usr/src/MCC-GPT")

import numpy as np
import torch
from typing import Tuple, List, Dict
from base_model.models import KBCModel
import pickle
from models import sample_model


def eval_and_get_rank_lists(model:KBCModel,dataset_name:str):
    test = open(f"generate_Contextual_Constraints_knowledge/knowledges/{dataset_name}/test.pickle", 'rb')
    print("test:\n",dataset_name)
    to_skip: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load\
        (open(f"generate_Contextual_Constraints_knowledge/knowledges/{dataset_name}/to_skip.pickle", 'rb'))
    examples = torch.from_numpy(np.array(pickle.load(test)).astype('int64'))
    count_8000 = torch.sum(examples >= 8000).item() 
    print("init_data:\n",examples)
    q = examples.clone()
    rank_list = model.get_rank_list(q, to_skip["rhs"])
    
    np.save(f"generate_Contextual_Constraints_knowledge/knowledges/{dataset_name}/base_model_inference_ans.npy", rank_list)




dataset_names =["WN18RR_IMG","FB15K_237_IMG"]

for dataset_name in dataset_names:
    if dataset_name =="WN18RR":
        #是model_WN9.pth的问题
        model = torch.load("./checkpoint/model_WN9.pth",map_location=torch.device('cpu'),weights_only=False)
    else:
        model = torch.load("./checkpoint/model_FB15K237.pth",map_location=torch.device('cpu'),weights_only=False)

    eval_and_get_rank_lists(model,dataset_name)