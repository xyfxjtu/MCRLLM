First configure your API key and dataset path in `preprocessing_datasets/configs.json`

## 1、Data processing

```bash
python ./preprocessing_datasets/imgs_to_one_txt_by_llm.py
python ./preprocessing_datasets/rel_to_desc_by_llm.py
python ./base_model/inference.py
```

## 2、generate MCC

```
python ./generate_Contextual_Constraints_knowledge/relation_constraints_generate.py
python ./generate_Contextual_Constraints_knowledge/multimodal_text_constrains_generate.py
python ./generate_Contextual_Constraints_knowledge/graph_context_constraints_generate.py
```

## 3、rerank

```
python ./llm_reranking/inference_and_eval.py --dataset FB15K_237_IMG --rerank SPN --mode Sampling --sampling_rate 0.025 --k 20
```

## Note

By default, MCC-GPT uses ComplEx as the retriever. The trained models for the two datasets are located at `./base_model/checkpoint/model_FB15K237.pth` and `./base_model/checkpoint/model_WN9.pth`. You can run `./base_model/inference.py` to save inference results. If you wish to replace the retriever, rewrite the code in the `./base_model` directory.

