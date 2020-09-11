# UNION

This repository contains the source code and data samples for the paper ”UNION: An Unreferenced Metric for Evaluating Open-ended Story Generation“.

## Prerequisites

The code is written in TensorFlow library. To use the program the following prerequisites need to be installed.

- Python 3.7.0
- tensorflow-gpu==1.14.0
- numpy ==1.18.1
- regex==2.5.76

## Computing infrastructure

We train UNION based on the platform: 

- OS: Ubuntu 16.04.3 LTS (GNU/Linux 4.4.0-98-generic x86_64)
- GPU: NVIDIA TITAN Xp



## Quick Start

#### 1. Constructing Negative Samples

Execute the following command: 

```shell
cd ./Data
python3 ./get_vocab.py your_mode
python3 ./gen_train_data.py your_mode
```

`your_mode` is `roc` for `ROCStories corpus` or  `wp` for `WritingPrompts dataset`. Then the summary of vocabulary and the corresponding frequency and pos-tagging will be found under `ROCStories/ini_data/entitiy_vocab.txt` or `WritingPrompts/ini_data/entity_vocab.txt`. Then negative samples and human-written stories will be constructed based on the original training set. The training set will be found under `ROCStories/train_data` or `WritingPrompts/train_data`.



#### 2. Training of UNION

Execute the following command: 

```shell
python3 ./run_union.py --data_dir your_data_dir \
											--output_dir ./model/union \
											--task_name train \
											--init_checkpoint ./model/uncased_L-12_H-768_A-12/bert_model.ckpt
```

`your_data_dir` is `./Data/ROCStories` or `./Data/WritingPrompts`.

The initial checkpoint of BERT can be downloaded from https://github.com/google-research/bert. We use the base version of BERT (~110M parameters). We train the model for 40 steps at most. The training process will task about 1~2 days. 



#### 3. Prediction with UNION

Execute the following command: 

```shell
python3 ./run_union.py --data_dir your_data_dir \
											--output_dir ./model/output \
											--task_name pred \
											--init_checkpoint your_model_name
```

`your_data_dir` is `./Data/ROCStories` or `./Data/WritingPrompts`, `your_model_name` is `./model/union_roc/union_roc` or `./model/union_wp/union_wp`. Then the union score of the stories under `your_data_dir/ant_data` can be found under the output_dir `./model/output`.

**Note: The checkpoint for reproduction will be provided by cloud disks after the Anonymous period due to the large size.**



#### 4. Correlation Calculation

Execute the following command: 

```shell
python3 ./correlation.py your_mode
```

Then the correlation between the human judgements under  `your_data_dir/ant_data` and the scores of metrics under `your_data_dir/metric_output` will be output.



## Data Instruction for files under `./Data`

- `negation.txt`: Manually constructed negation word vocabulary.
- `conceptnet_antonym.txt`: Triples with antonym relations extracted from ConceptNet.
- `conceptnet_entity.csv`: Entities acquired from ConceptNet.
- The full data for`ROCStories corpus` and  `WritingPrompts` can be found under `./ROCStories` and `./WritingPrompts`, respectively. The details for different directories and files are as follows:
  - `ant_data`:
    - `ant_data.txt` and `ant_data_all.txt`: Sampled stories and corresponding human annotation. `ant_data.txt` include only binary annotation for reasonable(1) or unreasonable(0). `ant_data_all.txt` include the annotation for specific error types (only available for `ROCStories corpus`): reasonable(0), repeated plots(1), bad coherence(2), conflicting logic(3), chaotic scenes(4), and others(5). The data are formatted as `Story ID ||| Story ||| Seven Annotated Scores`.
    - `reference.txt`,`reference_ipt.txt` and `reference_opt.txt`: the human-written stories with the same leading context with annotated stories.
  - `ini_data`: Original dataset for training/validation/testing. Besides,  `entity_vocab.txt` consists of all the entities and the corresponding tagged Part-Of-Speech followed by the mention frequency in the dataset.
  - `train_data`: Constructed negative samples and corresponding human-written stories for training.
  - `Metric output`: The scores of different metrics including `BLEU`/`MoverScore`/`RUBER-BERT`/`Perplexity`/`DisScore`/`UNION`/`UNION_Recon`, which can be used to replicate the correlation in Table 5 of the paper. `UNION_Recon` is the ablated model without the reconstruction task. And the sign of the result of Perplexity needs to be changed to get the result for *minus* Perplexity.

Note: currently only 10 samples of the full original data and training data are provided. **The full data will be provided by cloud disks after the Anonymous period due to the large size.**

