from collections import deque
from preprocessing_datasets import source_data
#TODO 这里的方法报错
#generate path constraints
def find_shortest_path(dataset_name, start_id, end_id):
    entityId_2_name = source_data.get_map_of_ids_to_attributes("ent", dataset_name, "name", "utf-8")
    relId_2_name = source_data.get_map_of_ids_to_attributes("rel", dataset_name, "name", "utf-8")
    train_data = source_data.get_triples(dataset_name, "train")
    graph = {}
    if dataset_name=="FB15K_237_IMG":
        for head, relation, tail in train_data:
            if head not in graph:
                graph[head] = []
            graph[head].append((relation, tail))
    else:
        for head, relation, tail in train_data:
            if head not in graph:
                graph[head] = []
            # print("graph数据：头部实体:{head};尾部实体:{tail};关系:{relation}\n",head,tail,relation)
            graph[head].append((relation, tail))
    # BFS
    queue = deque([(start_id, [entityId_2_name[start_id]])])
    visited = set()
    visited.add(start_id)
    while queue:
        current_node, path = queue.popleft()
        if current_node == end_id:
            return " -> ".join(path)
        for relation, neighbor in graph.get(current_node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                #TODO relation这里有严重bug
                # print("relation_id:",relation)
                # print("\nentity_id:",neighbor)
                queue.append((neighbor, path + [relId_2_name[relation], entityId_2_name[neighbor]]))

    return f"The graph has no reachable path from the head entity {entityId_2_name[start_id]} to the tail entity {entityId_2_name[end_id]}."  # 如果没有路径，返回"没有路径"


