import pickle
import re
from abc import ABC, abstractmethod
import numpy as np
from datetime import datetime
from generate_Contextual_Constraints_knowledge.shortest_path import find_shortest_path
from preprocessing_datasets import source_data
from generate_Contextual_Constraints_knowledge.chat_2_llm_in_context import do_context_chat_
from tqdm import tqdm
import utils


class Strategy(ABC):
    def __init__(self, dataset_name, strategy_name, K, mode, sampling_ratio):
        self.test_set = None
        self.res = None
        self.input_2_llm = None
        self.dataset_name = dataset_name
        self.process_id=0
        self.mode = mode
        self.id_list, self.rel_id_list = self.init_ids(dataset_name)
        self.entityId_2_name = source_data.get_map_of_ids_to_attributes("ent", dataset_name, "name", "utf-8")
        self.relId_2_name = source_data.get_map_of_ids_to_attributes("rel", dataset_name, "name", "utf-8")
        self.ent_desc = source_data.get_map_of_ids_to_attributes("ent", dataset_name, "desc", "utf-8")
        if dataset_name =="WN18RR":
            self.ent_img_desc = source_data.get_map_of_ids_to_attributes("ent", dataset_name, "img_desc", "utf-8") #gbk
        else:
            self.ent_img_desc = source_data.get_map_of_ids_to_attributes("ent", dataset_name, "img_desc", "utf-8")
        #TODO 这面这行代码可能是自动进行了去重处理
        # self.name_2_entityId = {v.split(",")[0]: k for k, v in self.entityId_2_name.items()}
        self.name_2_entityId = {value: key for key, value in self.entityId_2_name.items()}
        # self.name_2_entityId = swapped_dict
        self.strategy_name = strategy_name
        self.K = K
        self.instruction = self.get_instruction()
        self.prompt = "Please re-rank the candidate sequence based on the following information, " \
                      "ensuring that entities more likely to be the true prediction appear earlier.\n" \
                      "Prediction Item: ({})\nSupplemental Information for Prediction Item:\n{}\n" \
                      "Candidates:{}\n" \
                      "Supplemental Information for Candidates:\n{}\n" \
                      "Please provide the answer directly in the example output format without any additional analysis or explanations." \
                      "You must ensure that every entity strictly matches the candidate entities I provided (If an entity has aliases or synonyms, separate them with | and do not split them apart—they form a single entity.), " \
                      "and you must rank all candidate entities without adding new entries or omitting any existing ones." \
                      "You must ensure that every element in the re-ranked array is contained within the candidate array I provided"
        if mode == "Full":
            self.sampling_ratio = None
        elif mode == "Sampling":
            self.sampling_ratio = sampling_ratio
        self.load_inference_res()

    def eval(self):
        self.rank()
        self.get_metrics(self.res, self.test_set[:, 2])

    def rank(self):
        with tqdm(total=len(self.input_2_llm), desc=f"Process {self.process_id}", position=self.process_id, leave=True) as pbar:
            for idx, row in enumerate(self.input_2_llm):
                pbar.set_postfix(strategy=self.__class__.__name__)
                self.rerank(row, idx, pbar)
                pbar.update(1)

    def get_full_prompt(self, triple, candidates, have_relation_constraint, have_graph_context_constraint,
                        have_multimodal_text_constrain, have_path_constraint):
        head = self.entityId_2_name[triple[0]]
        rel = self.relId_2_name[triple[1]]
        prediction_item = f"[{head}],[{rel}],?"
        rel_constraint, rel_desc = self.get_relation_constrain(triple[1])
        if have_graph_context_constraint:
            graph_context_constrain = self.get_graph_context_constrain(triple[0])
        else:
            graph_context_constrain = "This entity has not relation constraint."
        if have_multimodal_text_constrain:
            multimodal_text_constrain = self.get_multimodal_text_constrain(triple[0], triple[1])
        else:
            multimodal_text_constrain = "This entity has not multimodal text constraint."
        if not have_relation_constraint:
            rel_constraint = "This triple has not relation constraint."
        supplementalprediction_item = f"Supplemental Information for Prediction Item:{{\n" \
                                      f"Entity Text Description:{self.ent_desc[triple[0]] if triple[0] in self.ent_desc.keys() else None}\n" \
                                      f"Entity Image Description:{self.ent_img_desc[triple[0]]}\n" \
                                      f"graph Context Constraints:{graph_context_constrain}\n" \
                                      f"Relation Description:{rel_desc}\n" \
                                      f"Relation Constraints:{rel_constraint}\n" \
                                      f"Multimodal Text Constraints:{multimodal_text_constrain} }}\n"
        names_of_candidates = [self.entityId_2_name[id].split(",")[0] for id in candidates]

        candidates_str = f"[{', '.join(names_of_candidates)}]"
        supplemental_candidates = "{\n"
        for id in candidates:
            if have_path_constraint:
                path_constraint = self.get_path_constrain(triple[0], id)
            else:
                path_constraint = "This entity has not path constrain."
            if have_graph_context_constraint:
                graph_context_constraint = self.get_graph_context_constrain(id)
            else:
                graph_context_constraint = "This entity has not relation constraint."
            cur_str = f"{{\n" \
                      f"Entity Text Description:{self.ent_desc[id] if id in self.ent_desc.keys() else None}\n" \
                      f"Entity Image Description:{self.ent_img_desc[id]}\n" \
                      f"Graph Context Constraints:{graph_context_constraint}\n" \
                      f"Path Constraint:{path_constraint}\n" \
                      f"}},\n"

            supplemental_candidates += cur_str
        supplemental_candidates += "}"
        full_prompt = self.prompt.format(prediction_item, supplementalprediction_item, candidates_str,
                                         supplemental_candidates)
        return full_prompt

    def parse_llm_ans(self, ans, init):
        try:
            ans = re.search(r'[\[【](.*?)[\]】]', ans)
            if not ans:
                print(ans)
                raise ValueError("llm replay format error.")
            ans = ans.group(1).strip()
            normalized = re.sub(r'[，、\n]', ',', ans)
            elements = []
            for item in normalized.split(','):
                item = item.strip()
                item = re.sub(r'^["“”‘’\']+|["“”‘’\']+$', '', item)
                if item:
                    elements.append(item)
            # 对不确定的输出进行兜底
            elements =utils.restore_and_match(init,elements)
            return elements
        except Exception as e:
            print(e)
            return init

    @abstractmethod
    def rerank(self, row, idx, pbar):
        pass

    def get_graph_context_constrain(self, id):
        path = f"./generate_Contextual_Constraints_knowledge/knowledges/{self.dataset_name}/graph_context_constraints.txt"
        if self.dataset_name =="WN18RR_IMG":
            decode = "utf-8"
        else:
            decode ="utf-8" #gbk
        with open(path, "r",encoding=decode) as f:
            for line in f:
                parts = line.strip().split()
                if (parts[0] == id):
                    return parts[1]
        return "This entity has not graph context constrain."

    def get_relation_constrain(self, id):
        path = f"./generate_Contextual_Constraints_knowledge/knowledges/{self.dataset_name}/relation_constraints.txt"
        if self.dataset_name =="WN18RR_IMG":
            decode = "utf-8"
        else:
            decode ="utf-8" #gbk
        with open(path, "r",encoding=decode) as f:
            for line in f:
                parts = line.strip().split("\t")
                if (parts[0] == id):
                    if len(parts) >= 4:
                        return parts[2], parts[3]
                    elif len(parts)==3:
                        return parts[2], None
                    else:
                        return None,None


    def get_multimodal_text_constrain(self, ent_id, rel_id):
        path = f"./generate_Contextual_Constraints_knowledge/knowledges/{self.dataset_name}/multimodal_text_constrains.txt"
        id = ent_id + ":" + rel_id
        if self.dataset_name =="WN18RR_IMG":
            decode = "utf-8"
        else:
            decode ="utf-8" #gbk
        with open(path, "r",encoding=decode) as f:
            for line in f:
                parts = line.strip().split()
                if (parts[0] == id):
                    return parts[1]
        return "The current head entity and relation has not multimodal text constrains."
    #TODO 重点debug这个方法
    def get_path_constrain(self, head_id, tail_id):
        return find_shortest_path(self.dataset_name, head_id, tail_id)

    def get_instruction(self):
        path = "/usr/src/MCC-GPT/llm_reranking/instruction.txt"
        with open(path, "r") as f:
            return f.read()

    def load_inference_res(self):
        path = f"./generate_Contextual_Constraints_knowledge/knowledges/{self.dataset_name}/base_model_inference_ans.npy"
        test_path = f"./generate_Contextual_Constraints_knowledge/knowledges/{self.dataset_name}/test.pickle"
        res = np.load(path)
        test_set = pickle.load(open(test_path, "rb"))
        if self.mode == "Full":
            self.res = res
            self.test_set = test_set
        elif self.mode == "Sampling":
            n_rows = res.shape[0]
            sample_size = int(np.round(n_rows * self.sampling_ratio))
            indices = np.random.choice(n_rows, size=sample_size, replace=False)
            print("res:\n",res[:100])
            print("type(res):",type(res))
            print("test_set:\n",test_set[:100])
            print("type(test_set):",type(test_set))
            test_set=np.array(test_set)
            self.res = res[indices]
            self.test_set = test_set[indices]
        self.input_2_llm = self.res[:, :self.K]
        print("input_2_llm:\n")
        print(self.input_2_llm)
    def init_ids(self, dataset_name):
        id_list = []
        rel_id_list = []
        with open(f"./generate_Contextual_Constraints_knowledge/knowledges/{dataset_name}/ent_id", "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    if (dataset_name == "WN18RR_IMG"):
                        id_list.append(parts[0])
                    else:
                        # id_list.append("n" + parts[0])
                        id_list.append(parts[0])
        with open(f"./generate_Contextual_Constraints_knowledge/knowledges/{dataset_name}/rel_id", "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    rel_id_list.append(parts[0])
        return id_list, rel_id_list

    def get_metrics(self, reranked_list: np.ndarray, ground_truth_labels: np.ndarray):
        metrics = {}
        matches = reranked_list == ground_truth_labels.reshape(-1, 1)
        ranks = np.argmax(matches.astype(int), axis=1) + 1
        mean_reciprocal_rank = np.mean(1. / ranks).item()
        mean_rank = np.mean(ranks).item()
        hits_at1 = np.mean((ranks <= 1)).item()
        hits_at3 = np.mean((ranks <= 3)).item()
        hits_at10 = np.mean((ranks <= 10)).item()
        metrics["MRR"] = mean_reciprocal_rank
        metrics["MR"] = mean_rank
        metrics["Hits@1"] = hits_at1
        metrics["Hits@3"] = hits_at3
        metrics["Hits@10"] = hits_at10
        print(metrics)
        with open("./record.txt", "a+") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            log_line = (
                f"************************************************************************************\n"
                f"[Dataset: {self.dataset_name}] "
                f"[Strategy: {self.strategy_name}] "
                f"[K: {self.K}] "
                f"[mode: {self.mode}] "
                f"[sampling_ratio: {self.sampling_ratio}] "
                f"[Time: {timestamp}]\n"
                f"MRR: {metrics['MRR']:.4f}, "
                f"MR: {metrics['MR']:.2f}, "
                f"Hits@1: {metrics['Hits@1']:.4f}, "
                f"Hits@3: {metrics['Hits@3']:.4f}, "
                f"Hits@10: {metrics['Hits@10']:.4f}\n"
                f"************************************************************************************\n"
            )
            f.write(log_line)
            f.flush()


class WeightedFusion(Strategy):
    def __init__(self, dataset_name, K, mode, sampling_ratio):
        super().__init__(dataset_name, "按权融合重排", K, mode, sampling_ratio)
        self.process_id = 0
    def borda_count(self, rankings):
        """
        使用 Borda Count 算法融合多个排名列表
        """
        scores = {}
        for ranking in rankings:
            m = len(ranking)  # 当前列表候选数量
            # 为每个候选对象计算当前列表得分
            for idx, candidate in enumerate(ranking):
                score = m - 1 - idx
                if candidate not in scores:
                    scores[candidate] = 0
                scores[candidate] += score
        sorted_candidates = sorted(
            scores.items(),
            key=lambda x: (-x[1], x[0])  # 先按分数降序，再按字母升序
        )
        return [candidate for candidate, _ in sorted_candidates]

    def rerank(self, row, idx, pbar):
        ranks = [None] * 4
        triple = [self.id_list[self.test_set[idx][0]], self.rel_id_list[self.test_set[idx][1]]]
        candidates = [self.id_list[id] for id in row]
        names_of_candidates = [self.entityId_2_name[id].split(",")[0] for id in candidates]
        for i in range(4):
            prompt_list = [self.get_instruction()]
            if i == 0:
                prompt_list.append(self.get_full_prompt(triple, candidates, True, False, False, False))
            elif i == 1:
                prompt_list.append(self.get_full_prompt(triple, candidates, False, True, False, False))
            elif i == 2:
                prompt_list.append(self.get_full_prompt(triple, candidates, False, False, True, False))
            else:
                prompt_list.append(self.get_full_prompt(triple, candidates, False, False, False, True))
            try:
                ans = do_context_chat_(prompt_list)
                ranks[i] = self.parse_llm_ans(ans, names_of_candidates)
            except Exception as e:
                pbar.write(f"error:{e}")
                ranks[i] = names_of_candidates
        final_rank = self.borda_count(ranks)
        try:
            # ranked_ids = [self.id_list.index(self.name_2_entityId[name]) for name in final_rank]
            ranked_ids = []
            for name in final_rank:
                #TODO 直接改成在self.entityId_2_name里找name对应的id，看看能不能解决问题
                # entity_id = self.name_2_entityId[name]      # 获取名字对应的实体ID
                entity_id = next((k for k, v in self.entityId_2_name.items() if v == name), None)
                index = self.id_list.index(entity_id)       # 获取实体ID在列表中的索引
                ranked_ids.append(index)                     # 将索引添加到结果列表                                                                                                                                                                                                  
            row[:] = ranked_ids
        except Exception as e:
            pbar.write(f"error:{e}")


class MultiLevel(Strategy):
    def __init__(self, dataset_name,K, mode, sampling_ratio):
        super().__init__(dataset_name, "多级重排", K, mode, sampling_ratio)
        self.process_id = 1

    def rerank(self, row, idx, pbar):
        cur_prompt = "Below, I will further refine the supplemental information. Please update and adjust your " \
                     "re-ranking results by integrating this newly added information with the original data and your " \
                     "current ranking outcomes."
        prompt_list = [self.get_instruction()]
        triple = [self.id_list[self.test_set[idx][0]], self.rel_id_list[self.test_set[idx][1]]]
        candidates = [self.id_list[id] for id in row]
        prompt_list.append(self.get_full_prompt(triple, candidates, True, False, False, False))
        prompt_list.append(self.get_full_prompt(triple, candidates, True, True, False, False).replace("Please re-rank "
                                                                                                      "the candidate "
                                                                                                      "sequence based "
                                                                                                      "on the "
                                                                                                      "following "
                                                                                                      "information,",
                                                                                                      cur_prompt))
        prompt_list.append(self.get_full_prompt(triple, candidates, True, True, True, False).replace("Please re-rank "
                                                                                                     "the candidate "
                                                                                                     "sequence based "
                                                                                                     "on the "
                                                                                                     "following "
                                                                                                     "information,",
                                                                                                     cur_prompt))
        prompt_list.append(self.get_full_prompt(triple, candidates, True, True, True, True).replace("Please re-rank "
                                                                                                    "the candidate "
                                                                                                    "sequence based "
                                                                                                    "on the following "
                                                                                                    "information,",
                                                                                                    cur_prompt))
        names_of_candidates = [self.entityId_2_name[id].split(",")[0] for id in candidates]
        try:
            ans = do_context_chat_(prompt_list)
            rank = self.parse_llm_ans(ans, names_of_candidates)
        except Exception as e:
            pbar.write(f"error:{e}")
            rank = names_of_candidates
        try:
            # ranked_ids = [self.id_list.index(self.name_2_entityId[name]) for name in rank]
            ranked_ids = []
            for name in rank:
                #TODO 直接改成在self.entityId_2_name里找name对应的id，看看能不能解决问题
                # entity_id = self.name_2_entityId[name]      # 获取名字对应的实体ID
                entity_id = next((k for k, v in self.entityId_2_name.items() if v == name), None)
                index = self.id_list.index(entity_id)       # 获取实体ID在列表中的索引
                ranked_ids.append(index)                     # 将索引添加到结果列表
            row[:] = ranked_ids
        except Exception as e:
            pbar.write(f"error:{e}")


class SelfGeneratedExample(Strategy):
    def __init__(self, dataset_name,K, mode, sampling_ratio):
        super().__init__(dataset_name, "正负例自生成重排", K, mode, sampling_ratio)
        self.process_id = 2
    def rerank(self, row, idx, pbar):
        triple = [self.id_list[self.test_set[idx][0]], self.rel_id_list[self.test_set[idx][1]]]
        # print("id_list:",self.id_list)
        print("len:",len(self.id_list))
        print("cur_id:",self.test_set[idx][0])
        print("id:",row)
        print("dataset:",self.dataset_name)
        candidates = [self.id_list[id] for id in row]
        names_of_candidates = [self.entityId_2_name[id].split(",")[0] for id in candidates]
        prompt1 = "Now, please generate supporting evidence and contradictions for each candidate entity according to " \
                  "your re-ranked order, based on the existing information and your re-ranking results."
        prompt2 = "Analyze and reflect on the supporting evidence and identified contradictions, then refine your " \
                  "current re-ranking results by performing an additional round of re-ranking to enhance the logical " \
                  "consistency of the final outcome."
        prompt_list = [self.get_instruction(), self.get_full_prompt(triple, candidates, True, True, True, True),prompt1,prompt2]
        try:
            ans = do_context_chat_(prompt_list)
            rank = self.parse_llm_ans(ans, names_of_candidates)
        except Exception as e:
            pbar.write(f"error111:{e}")
            rank = names_of_candidates
        try:
            #TODO 是这里报错了！！！
            print()
            # ranked_ids = [self.id_list.index(self.name_2_entityId[name]) for name in rank]
            ranked_ids = []
            for name in rank:
                #TODO 直接改成在self.entityId_2_name里找name对应的id，看看能不能解决问题
                # entity_id = self.name_2_entityId[name]      # 获取名字对应的实体ID
                entity_id = next((k for k, v in self.entityId_2_name.items() if v == name), None)
                index = self.id_list.index(entity_id)       # 获取实体ID在列表中的索引
                ranked_ids.append(index)                     # 将索引添加到结果列表
            row[:] = ranked_ids
        except Exception as e:

            pbar.write(f"error222:{e}")