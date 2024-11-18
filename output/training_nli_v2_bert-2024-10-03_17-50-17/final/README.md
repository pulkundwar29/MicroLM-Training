---
datasets:
- sentence-transformers/all-nli
language:
- en
library_name: sentence-transformers
metrics:
- pearson_cosine
- spearman_cosine
- pearson_manhattan
- spearman_manhattan
- pearson_euclidean
- spearman_euclidean
- pearson_dot
- spearman_dot
- pearson_max
- spearman_max
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:557850
- loss:MultipleNegativesRankingLoss
widget:
- source_sentence: A man dressed in yellow rescue gear walks in a field.
  sentences:
  - A person messes with some papers.
  - The man is outdoors.
  - The man is bowling.
- source_sentence: A young woman tennis player dressed in black carries many tennis
    balls on her racket.
  sentences:
  - A young woman tennis player have many tennis balls.
  - Two men are fishing.
  - A young woman never wears white dress.
- source_sentence: An older gentleman enjoys a scenic stroll through the countryside.
  sentences:
  - A pirate boards the spaceship.
  - A man walks the countryside.
  - Girls standing at a whiteboard in front of class.
- source_sentence: A kid in a red and black coat is laying on his back in the snow
    with his arm in the air and a red sled is next to him.
  sentences:
  - It is a cold day.
  - A girl with her hands in a tub.
  - The kid is on a sugar high.
- source_sentence: A young boy playing in the grass.
  sentences:
  - A woman in a restaurant.
  - The boy is in the sand.
  - There is a child in the grass.
model-index:
- name: SentenceTransformer
  results:
  - task:
      type: semantic-similarity
      name: Semantic Similarity
    dataset:
      name: sts dev
      type: sts-dev
    metrics:
    - type: pearson_cosine
      value: .nan
      name: Pearson Cosine
    - type: spearman_cosine
      value: .nan
      name: Spearman Cosine
    - type: pearson_manhattan
      value: .nan
      name: Pearson Manhattan
    - type: spearman_manhattan
      value: .nan
      name: Spearman Manhattan
    - type: pearson_euclidean
      value: .nan
      name: Pearson Euclidean
    - type: spearman_euclidean
      value: .nan
      name: Spearman Euclidean
    - type: pearson_dot
      value: .nan
      name: Pearson Dot
    - type: spearman_dot
      value: .nan
      name: Spearman Dot
    - type: pearson_max
      value: .nan
      name: Pearson Max
    - type: spearman_max
      value: .nan
      name: Spearman Max
  - task:
      type: semantic-similarity
      name: Semantic Similarity
    dataset:
      name: sts test
      type: sts-test
    metrics:
    - type: pearson_cosine
      value: .nan
      name: Pearson Cosine
    - type: spearman_cosine
      value: .nan
      name: Spearman Cosine
    - type: pearson_manhattan
      value: .nan
      name: Pearson Manhattan
    - type: spearman_manhattan
      value: .nan
      name: Spearman Manhattan
    - type: pearson_euclidean
      value: .nan
      name: Pearson Euclidean
    - type: spearman_euclidean
      value: .nan
      name: Spearman Euclidean
    - type: pearson_dot
      value: .nan
      name: Pearson Dot
    - type: spearman_dot
      value: .nan
      name: Spearman Dot
    - type: pearson_max
      value: .nan
      name: Pearson Max
    - type: spearman_max
      value: .nan
      name: Spearman Max
---

# SentenceTransformer

This is a [sentence-transformers](https://www.SBERT.net) model trained on the [all-nli](https://huggingface.co/datasets/sentence-transformers/all-nli) dataset. It maps sentences & paragraphs to a 50-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
<!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 50 tokens
- **Similarity Function:** Cosine Similarity
- **Training Dataset:**
    - [all-nli](https://huggingface.co/datasets/sentence-transformers/all-nli)
- **Language:** en
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 50, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'A young boy playing in the grass.',
    'There is a child in the grass.',
    'The boy is in the sand.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 50]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Semantic Similarity
* Dataset: `sts-dev`
* Evaluated with [<code>EmbeddingSimilarityEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator)

| Metric              | Value   |
|:--------------------|:--------|
| pearson_cosine      | nan     |
| **spearman_cosine** | **nan** |
| pearson_manhattan   | nan     |
| spearman_manhattan  | nan     |
| pearson_euclidean   | nan     |
| spearman_euclidean  | nan     |
| pearson_dot         | nan     |
| spearman_dot        | nan     |
| pearson_max         | nan     |
| spearman_max        | nan     |

#### Semantic Similarity
* Dataset: `sts-test`
* Evaluated with [<code>EmbeddingSimilarityEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator)

| Metric              | Value   |
|:--------------------|:--------|
| pearson_cosine      | nan     |
| **spearman_cosine** | **nan** |
| pearson_manhattan   | nan     |
| spearman_manhattan  | nan     |
| pearson_euclidean   | nan     |
| spearman_euclidean  | nan     |
| pearson_dot         | nan     |
| spearman_dot        | nan     |
| pearson_max         | nan     |
| spearman_max        | nan     |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### all-nli

* Dataset: [all-nli](https://huggingface.co/datasets/sentence-transformers/all-nli) at [d482672](https://huggingface.co/datasets/sentence-transformers/all-nli/tree/d482672c8e74ce18da116f430137434ba2e52fab)
* Size: 557,850 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                             | positive                                                                          | negative                                                                           |
  |:--------|:-----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                            | string                                                                             |
  | details | <ul><li>min: 10 tokens</li><li>mean: 17.73 tokens</li><li>max: 90 tokens</li></ul> | <ul><li>min: 9 tokens</li><li>mean: 23.09 tokens</li><li>max: 79 tokens</li></ul> | <ul><li>min: 10 tokens</li><li>mean: 24.49 tokens</li><li>max: 90 tokens</li></ul> |
* Samples:
  | anchor                                                                     | positive                                         | negative                                                   |
  |:---------------------------------------------------------------------------|:-------------------------------------------------|:-----------------------------------------------------------|
  | <code>A person on a horse jumps over a broken down airplane.</code>        | <code>A person is outdoors, on a horse.</code>   | <code>A person is at a diner, ordering an omelette.</code> |
  | <code>Children smiling and waving at camera</code>                         | <code>There are children present</code>          | <code>The kids are frowning</code>                         |
  | <code>A boy is jumping on skateboard in the middle of a red bridge.</code> | <code>The boy does a skateboarding trick.</code> | <code>The boy skates down the sidewalk.</code>             |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Evaluation Dataset

#### all-nli

* Dataset: [all-nli](https://huggingface.co/datasets/sentence-transformers/all-nli) at [d482672](https://huggingface.co/datasets/sentence-transformers/all-nli/tree/d482672c8e74ce18da116f430137434ba2e52fab)
* Size: 6,584 evaluation samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                              | positive                                                                          | negative                                                                          |
  |:--------|:------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                              | string                                                                            | string                                                                            |
  | details | <ul><li>min: 10 tokens</li><li>mean: 34.38 tokens</li><li>max: 134 tokens</li></ul> | <ul><li>min: 7 tokens</li><li>mean: 17.19 tokens</li><li>max: 63 tokens</li></ul> | <ul><li>min: 7 tokens</li><li>mean: 18.27 tokens</li><li>max: 56 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                         | positive                                                    | negative                                                |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------|:--------------------------------------------------------|
  | <code>Two women are embracing while holding to go packages.</code>                                                                                                             | <code>Two woman are holding packages.</code>                | <code>The men are fighting outside a deli.</code>       |
  | <code>Two young children in blue jerseys, one with the number 9 and one with the number 2 are standing on wooden steps in a bathroom and washing their hands in a sink.</code> | <code>Two kids in numbered jerseys wash their hands.</code> | <code>Two kids in jackets walk to school.</code>        |
  | <code>A man selling donuts to a customer during a world exhibition event held in the city of Angeles</code>                                                                    | <code>A man selling donuts to a customer.</code>            | <code>A woman drinks her coffee in a small cafe.</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 128
- `per_device_eval_batch_size`: 128
- `num_train_epochs`: 1
- `warmup_ratio`: 0.1
- `fp16`: True
- `batch_sampler`: no_duplicates

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 128
- `per_device_eval_batch_size`: 128
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: proportional

</details>

### Training Logs
| Epoch  | Step | loss   | sts-dev_spearman_cosine | sts-test_spearman_cosine |
|:------:|:----:|:------:|:-----------------------:|:------------------------:|
| 0      | 0    | -      | nan                     | -                        |
| 0.1266 | 10   | 5.2666 | nan                     | -                        |
| 0.2532 | 20   | 5.2666 | nan                     | -                        |
| 0.3797 | 30   | 5.2666 | nan                     | -                        |
| 0.5063 | 40   | 5.2666 | nan                     | -                        |
| 0.6329 | 50   | 5.2666 | nan                     | -                        |
| 0.7595 | 60   | 5.2666 | nan                     | -                        |
| 0.8861 | 70   | 5.2666 | nan                     | -                        |
| 1.0    | 79   | -      | -                       | nan                      |


### Framework Versions
- Python: 3.12.4
- Sentence Transformers: 3.1.1
- Transformers: 4.45.1
- PyTorch: 2.4.1
- Accelerate: 0.34.2
- Datasets: 3.0.1
- Tokenizers: 0.20.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->