# BERT on MOCO

This repository contains code for [BERT on STILTs](https://arxiv.org/abs/1811.01088v2). It is a fork of the [Hugging Face implementation of BERT](https://github.com/huggingface/pytorch-pretrained-BERT).


## MOCO task

#### Data Preparation

You need to augment your data via two different ways and save them in the *'augment.csv' in the same form.

Frist way: English --> Chinese --> English

Second way: English --> German --> English

#### Model Output

Before training, you need to build the moco_model with *mkdir moco_model*

#### Train
You need to change the number of negtive samples (number of augmented data) in MOCO.py *line 84* , you can also change the epoch: *line 41*, batch size:*line 45*, learning rate:*line 50*, and temperature: *line 90*

You can train the MOCO task with:

*CUDA_VISIBLE_DEVICES=0 python MOCO.py*

#### Transform Model

After training, you can extract encoder_k from the whole model with

*python trans.py*

*num_labels=2* (The number of output labels--2 for binary classifier) You can increase this for multiple classification 

=======

## finetune Models

#### Preparation

You will need to download the GLUE data to run our tasks. See [here](https://gluebenchmark.com/tasks).

You will also need to set the two following environment variables:

* `GLUE_DIR`: This should point to the location of the GLUE data downloaded from `jiant`.
* `BERT_ALL_DIR`: Set `BERT_ALL_DIR=/PATH_TO_THIS_REPO/cache/bert_metadata` 
    * For mor general use: `BERT_ALL_DIR` should point to the location of BERT downloaded from [here](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip). Importantly, the `BERT_ALL_DIR` needs to contain the files `uncased_L-24_H-1024_A-16/bert_config.json` and `uncased_L-24_H-1024_A-16/vocab.txt`.
  
 You can also change the dataset: *line24* and the epoch: *line89*:

##### Example 1: Generating Predictions

To generate validation/test predictions, as well as validation metrics, run something like the following:

```bash
export GLUE_DIR=./data/MNLI 
export TASK=rte
export BERT_LOAD_PATH=path/to/mnli__rte.p
export OUTPUT_PATH=rte_output

python train.py \
    --task_name $TASK \
    --do_val --do_test \
    --do_lower_case \
    --bert_model bert-large-uncased \
    --bert_load_mode full_model_only \
    --bert_load_path $BERT_LOAD_PATH \
    --eval_batch_size 64 \
    --output_dir $OUTPUT_PATH
``` 

##### Example 2: Fine-tuning from vanilla BERT

We recommend training with a batch size of 16/24/32.

```bash
export GLUE_DIR=./data/MNLI                                                                                              
export BERT_ALL_DIR=./   
export TASK=mnli
export OUTPUT_PATH=mnli_output

python train.py \
    --task_name $TASK \
    --do_train --do_val --do_test --do_val_history \
    --do_save \
    --do_lower_case \
    --bert_model bert-large-uncased \
    --bert_load_mode from_pretrained \
    --bert_save_mode model_all \
    --train_batch_size 24 \
    --learning_rate 2e-5 \
    --output_dir $OUTPUT_PATH
``` 


##### Example 3: Fine-tuning from MOCO model

```bash
export GLUE_DIR=./data/RTE
export PRETRAINED_MODEL_PATH=/path/to/moco.p
export TASK=rte
export OUTPUT_PATH=rte_output

python train.py \
    --task_name $TASK \
    --do_train --do_val --do_test --do_val_history \
    --do_save \
    --do_lower_case \
    --bert_model bert-large-uncased \
    --bert_load_path $PRETRAINED_MODEL_PATH \
    --bert_load_mode model_only \
    --bert_save_mode model_all \
    --train_batch_size 24 \
    --learning_rate 2e-5 \
    --output_dir $OUTPUT_PATH
``` 
You can take [example.sh](https://github.com/ColeFang/CERT/blob/master/example.sh) as an example.

## Submission to GLUE leaderboard

We have included helper scripts for exporting submissions to the GLUE leaderboard. To prepare for submission, copy the template from `cache/submission_template` to a given new output folder:

```bash
cp -R cache/submission_template /path/to/new_submission
```

After running a fine-tuned/pretrained model on a task with the `--do_test` argument, a folder (e.g. `rte_output`) will be created containing `test_preds.csv` among other files. Run the following command to convert `test_preds.csv` to the submission format in the output folder.

```bash
python format_for_glue.py\
    --task-name rte \
    --input-base-path /path/to/rte_output \
    --output-base-path /path/to/new_submission
```

Once you have exported submission predictions for each task, you should have 11 `.tsv` files in total. If you run `wc -l *.tsv`, you should see something like the following:

```
   1105 AX.tsv
   1064 CoLA.tsv
   9848 MNLI-mm.tsv
   9797 MNLI-m.tsv
   1726 MRPC.tsv
   5464 QNLI.tsv
 390966 QQP.tsv
   3001 RTE.tsv
   1822 SST-2.tsv
   1380 STS-B.tsv
    147 WNLI.tsv
 426597 total 
```

Next run `zip -j -D submission.zip *.tsv` in the folder to generate the submission zip file. Upload the zip file to [https://gluebenchmark.com/submit](https://gluebenchmark.com/submit) to submit to the leaderboard.


