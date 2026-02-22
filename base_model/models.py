from base_models import *



class KBCModel(nn.Module, ABC):
    @abstractmethod
    def candidates_score(self, chunk_begin: int, chunk_size: int, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    # 二分类式训练
    def forward_bpr(self, batch):
        pass

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        self.eval()
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]

                    scores = self.candidates_score(c_begin, chunk_size, these_queries)
                    targets = self.score(these_queries)
                    for i, query in enumerate(these_queries):
                        # 当前满足 （head,relation，tail） 的tail组成的列表
                        filter_out = filters[(query[0].item(), query[1].item())]
                        # 真实tail id 放进去
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    # 一个batch中大于等于当前预测得分的尾实体的总数（过滤掉存在的list） （+1得到排名）
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks

    def get_rank_list(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        count_8000 = torch.sum(queries <= 8000).item() 
        print("input_count_8000:\n",count_8000)
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        all_ranks = []
        self.eval()
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    # batch_size x nodes_num
                    scores = self.candidates_score(c_begin, chunk_size, these_queries)
                    print("scores:\n",scores)
                    # >>>>>>>>>>> 新增：屏蔽掉 >=8000 的候选 <<<<<<<<<<<
                    # if c_begin < 30000:
                    #     end_in_chunk = min(c_begin + chunk_size, 30000)
                    #     valid_len = end_in_chunk - c_begin
                    #     # 将 [valid_len, chunk_size) 范围内的分数设为极小值
                    #     if valid_len < scores.size(1):
                    #         scores[:, valid_len:] = -1e6
                    # else:
                    #     # 当前 chunk 完全 >=8000，全部屏蔽
                    #     scores.fill_(-1e6)
                     # <<<<<<<<< 新增结束 <<<<<<<<<
                    
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        #TODO: 除了当前ground truth外别的正确标签过滤掉
                        filter_out = [item for item in filter_out if item!=queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            # 真实分数置为-1e6
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks = torch.argsort(scores, dim=1, descending=True).to(torch.int16)
                    all_ranks.append(ranks)
                    b_begin += batch_size
                c_begin += chunk_size
        print("ranks:\n",all_ranks[:100])
        return torch.cat(all_ranks, dim=0)
class sample_model(KBCModel):
    def __init__(self, sizes, n_node, n_rel, dim, n_neighbor, context_weight,adj, img_info, desc_info, node_info=None,
                 rel_desc_info=None):
        super(sample_model, self).__init__()
        self.sizes = sizes
        self.n_node = n_node
        self.n_rel = n_rel
        self.dim = dim
        self.context_weight = context_weight
        self.adj_indices, self.adj_values = adj[0].to('cuda'), adj[1].to('cuda')
        self.n_neighbor = n_neighbor
        if img_info is not None:
            self.img_info = nn.Embedding.from_pretrained(
                torch.from_numpy(pickle.load(open(img_info, 'rb'))), freeze=False)
        else:
            self.img_info = nn.Embedding(n_node, dim, sparse=True)
            nn.init.xavier_uniform_(self.img_info.weight)
        if desc_info is not None:
            self.desc_info = nn.Embedding.from_pretrained(
                torch.from_numpy(pickle.load(open(desc_info, 'rb'))), freeze=False)
        else:
            self.desc_info = nn.Embedding(n_node, dim, sparse=True)
            nn.init.xavier_uniform_(self.desc_info.weight)
        if node_info is not None:
            self.node_info = nn.Embedding.from_pretrained(
                torch.from_numpy(pickle.load(open(node_info, 'rb'))), freeze=False)
        else:
            self.node_info = nn.Embedding(n_node, dim, sparse=True)
            nn.init.xavier_uniform_(self.node_info.weight)
        if rel_desc_info is not None:
            self.rel_info = nn.Embedding.from_pretrained(
                torch.vstack((torch.from_numpy(pickle.load(open(rel_desc_info, 'rb'))),
                              torch.from_numpy(pickle.load(open(rel_desc_info, 'rb')))))
                , freeze=False)
        else:
            self.rel_info = nn.Embedding(n_rel * 2, dim, sparse=True)
            nn.init.xavier_uniform_(self.rel_info.weight)

        self._score = ComplEx(dim//2,self.node_info)

    def forward(self, x, candidate=False, score=False, chunk_begin=-1, chunk_size=-1):
        lhs_node = self.node_info.weight[x[:, 0]]
        lhs_desc = self.desc_info.weight[x[:, 0]]
        lhs_img = self.img_info.weight[x[:, 0]]
        rhs_node = self.node_info.weight[x[:, 2]]
        rel = self.rel_info.weight[x[:, 1]]
        # fusion_info = (lhs_node+lhs_desc+lhs_img).to(torch.float32)

        # assert not torch.isnan(fusion_info).any()
        pred_node = self._score(lhs_node, rel,rhs = rhs_node,to_score=self.node_info.weight,candidate=candidate, score=score, start=chunk_begin,
                            end=chunk_begin + chunk_size, queries=x)
        pred_desc = self._score(lhs_desc, rel,rhs = rhs_node,to_score=self.desc_info.weight,candidate=candidate, score=score, start=chunk_begin,
                            end=chunk_begin + chunk_size, queries=x)
        pred_img = self._score(lhs_img, rel,rhs = rhs_node,to_score=self.img_info.weight,candidate=candidate, score=score, start=chunk_begin,
                            end=chunk_begin + chunk_size, queries=x)
        if candidate or score:
            pred = pred_node+pred_desc+pred_img
        else:
            pred = (pred_node[0]+pred_desc[0]+pred_img[0],
                    (pred_node[1][0]+pred_desc[1][0]+pred_img[1][0],
                     pred_node[1][1] + pred_desc[1][1] + pred_img[1][1],
                     pred_node[1][2] + pred_desc[1][2] + pred_img[1][2]))
        return pred





    def candidates_score(self, chunk_begin: int, chunk_size: int, queries: torch.Tensor):

        return self.forward(queries, candidate=True, chunk_begin=chunk_begin, chunk_size=chunk_size)

    def score(self, x: torch.Tensor):

        return self.forward(x, score=True)

    def forward_bpr(self, batch):
        # pos_scores = self.score(pos)
        # neg_scores = self.score(neg)
        scores = self.score(batch)
        # delta = pos_scores - neg_scores
        # fac = self.get_factor(torch.cat((pos, neg), dim=0))
        return scores




