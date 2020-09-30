## Preprocess the data
```
usage: preprocess.py [-h] --corpus_path CORPUS_PATH [--vocab_path VOCAB_PATH]
                     [--spm_model_path SPM_MODEL_PATH]
                     [--dataset_path DATASET_PATH]
                     [--tokenizer {bert,char,space}]
                     [--processes_num PROCESSES_NUM]
                     [--target {bert,lm,cls,mlm,bilm,albert}]
                     [--docs_buffer_size DOCS_BUFFER_SIZE]
                     [--seq_length SEQ_LENGTH] [--dup_factor DUP_FACTOR]
                     [--short_seq_prob SHORT_SEQ_PROB] [--full_sentences]
                     [--seed SEED] [--dynamic_masking] [--span_masking]
                     [--span_geo_prob SPAN_GEO_PROB]
                     [--span_max_length SPAN_MAX_LENGTH]
```
Users have to preprocess the corpus before pre-training. <br> 
The example of pre-processing on a single machine：
```
python3 preprocess.py --corpus_path corpora/book_review_bert.txt --vocab_path models/google_zh_vocab.txt --dataset_path dataset.pt\
                      --processes_num 8 --target bert
```
If multiple machines are available, users can run *preprocess.py* on one machine and copy the *dataset.pt* to other machines. 

We need to specify model's target (*--target*) in pre-processing stage since different targets require different data formats. Currently, UER-py consists of the following target modules:
- lm: language model
- mlm: masked language model (cloze test)
- cls: classification
- bilm: bi-directional language model
- bert: masked language model + next sentence prediction
- albert: masked language model + sentence order prediction

*--processes_num n* denotes that n processes are used for pre-processing. More processes can speed up the preprocess stage but lead to more memory consumption. <br>
*--dynamic_masking* denotes that the words are masked during the pre-training stage, which is used in RoBERTa. <br>
*--full_sentences* allows a sample to include contents from multiple documents, which is used in RoBERTa. <br>
*--span_masking* denotes that masking consecutive words, which is used in SpanBERT. If dynamic masking is used, we should specify *--span_masking* in pre-training stage, otherwise we should specify *--span_masking* in pre-processing stage. <br>
*--docs_buffer_size* specifies the buffer size in memory in pre-processing stage. <br>
Sequence length is specified in pre-processing stage by *--seq_length* . The default value is 128. <br>
Vocabulary and tokenizer are also specified in pre-processing stage. More details are discussed in *Tokenization and vocabulary* section.
<br><br>

## Pretrain the model
```
usage: pretrain.py [-h] [--dataset_path DATASET_PATH]
                   [--vocab_path VOCAB_PATH] [--spm_model_path SPM_MODEL_PATH]
                   [--pretrained_model_path PRETRAINED_MODEL_PATH]
                   --output_model_path OUTPUT_MODEL_PATH
                   [--config_path CONFIG_PATH] [--total_steps TOTAL_STEPS]
                   [--save_checkpoint_steps SAVE_CHECKPOINT_STEPS]
                   [--report_steps REPORT_STEPS]
                   [--accumulation_steps ACCUMULATION_STEPS]
                   [--batch_size BATCH_SIZE]
                   [--instances_buffer_size INSTANCES_BUFFER_SIZE]
                   [--dropout DROPOUT] [--seed SEED] [--embedding {bert,word}]
                   [--encoder {bert,lstm,gru,cnn,gatedcnn,attn,synt,rcnn,crnn,gpt,bilstm}]
                   [--bidirectional] [--target {bert,lm,cls,mlm,bilm}]
                   [--tie_weights] [--factorized_embedding_parameterization]
                   [--parameter_sharing] [--span_masking]
                   [--span_geo_prob SPAN_GEO_PROB]
                   [--span_max_length SPAN_MAX_LENGTH]
                   [--learning_rate LEARNING_RATE] [--warmup WARMUP]
                   [--beta1 BETA1] [--beta2 BETA2] [--fp16]
                   [--fp16_opt_level {O0,O1,O2,O3}] [--world_size WORLD_SIZE]
                   [--gpu_ranks GPU_RANKS [GPU_RANKS ...]]
                   [--master_ip MASTER_IP] [--backend {nccl,gloo}]
```
It is required to explicitly specify model's encoder and target. UER-py consists of the following encoder modules:
- lstm: long short-term memory (LSTM)
- gru: gated recurrent units (GRU)
- bilstm: bi-LSTM (different from *--encoder lstm* with *--bidirectional* , see [the issue](https://github.com/pytorch/pytorch/issues/4930) for more details)
- gatedcnn: gated convolutional networks (GatedCNN)
- bert: the Transformer with fully-visible mask (used in BERT)
- gpt: the Transformer with causal mask (used in GPT)

The target should be coincident with the target in pre-processing stage. Users can try different combinations of encoders and targets by *--encoder* and *--target* .
*--config_path* denotes the path of the configuration file, which specifies the hyper-parameters of the pre-training model. We have put the commonly-used configuration files in *models* folder. Users should choose the proper one according to encoder they use. <br>
*--instances_buffer_size* specifies the buffer size in memory in pre-training stage. <br>
*--tie_weights* denotes the word embedding and softmax weights are tied. <br>

There are two strategies for parameter initialization of pre-training: 1）random initialization; 2）loading a pre-trained model.
#### Random initialization
The example of pre-training on CPU：
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --output_model_path models/output_model.bin \
                    --encoder bert --target bert
```
The input of pre-training is specified by *--dataset_path* .
The example of pre-training on single GPU (the ID of GPU is 3)：
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --output_model_path models/output_model.bin --gpu_ranks 3 \
                    --encoder bert --target bert
```
The example of pre-training on a single machine with 8 GPUs：
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --output_model_path models/output_model.bin --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --encoder bert --target bert
```
*--world_size* specifies the number of processes (and GPUs) used for pre-training. <br>
*--gpu_ranks* specifies the ID for each process and GPU. The IDs are from *0* to *n-1*, where *n* is the number of processes used for pre-training. <br>
Users could use CUDA_VISIBLE_DEVICES if they want to use part of GPUs:
```
CUDA_VISIBLE_DEVICES=1,2,3,5 python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                                                 --output_model_path models/output_model.bin --world_size 4 --gpu_ranks 0 1 2 3 \
                                                 --encoder bert --target bert
```
*--world_size* is set to 4 since only 4 GPUs are used. The IDs of 4 processes (and GPUs) is 0, 1, 2, and 3, which are specified by *--gpu_ranks* .

The example of pre-training on two machines: each machine has 8 GPUs (16 GPUs in total).
We run *pretrain.py* on two machines (Node-0 and Node-1) respectively. *--master_ip* specifies the ip:port of the master mode, which contains process (and GPU) of ID 0.
```
Node-0 : python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                             --output_model_path models/output_model.bin --encoder bert --target bert --world_size 16 --gpu_ranks 0 1 2 3 4 5 6 7 \
                             --total_steps 100000 --save_checkpoint_steps 10000 --report_steps 100 \
                             --master_ip tcp://9.73.138.133:12345
Node-1 : python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                             --output_model_path models/output_model.bin --encoder bert --target bert --world_size 16 --gpu_ranks 8 9 10 11 12 13 14 15 \
                             --total_steps 100000 \
                             --master_ip tcp://9.73.138.133:12345
```
The IP of Node-0 is 9.73.138.133 . <br>
*--total_steps* specifies the training steps. <br>
*--save_checkpoint_steps* specifies how often to save the model checkpoint. We don't need to specify *--save_checkpoint_steps* in Node-1 since only master node saves the pre-trained model. <br>
*--report_steps* specifies how often to report the pre-training information. We don't need to specify *--report_steps* in Node-1 since the information only appears in master node. <br>
Notice that when specifying *--master_ip* one can not select the port that occupied by other programs. <br>
For random initialization, pre-training usually requires larger learning rate. We recommend to use *--learning_rate 1e-4* . The default value is *2e-5* .

#### Load the pre-trained model
We recommend to load a pre-trained model. We can specify the pre-trained model by *--pretrained_model_path* .
The example of pre-training on CPU and single GPU:
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --pretrained_model_path models/google_zh_model.bin \
                    --output_model_path models/output_model.bin \
                    --encoder bert --target bert
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --pretrained_model_path models/google_zh_model.bin \
                    --output_model_path models/output_model.bin --gpu_ranks 3 \
                    --encoder bert --target bert
```
The example of pre-training on a single machine with 8 GPUs：
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --pretrained_model_path models/google_zh_model.bin \
                    --output_model_path models/output_model.bin --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --encoder bert --target bert 
```
The example of pre-training on two machines: each machine has 8 GPUs (16 GPUs in total):
```
Node-0 : python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                             --pretrained_model_path models/google_zh_model.bin \
                             --output_model_path models/output_model.bin --world_size 16 --gpu_ranks 0 1 2 3 4 5 6 7 \
                             --master_ip tcp://9.73.138.133:12345 --encoder bert --target bert  
Node-1 : python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                             --pretrained_model_path models/google_zh_model.bin \
                             --output_model_path models/output_model.bin --world_size 16 --gpu_ranks 8 9 10 11 12 13 14 15 \
                             --master_ip tcp://9.73.138.133:12345 --encoder bert --target bert  
```
The example of pre-training on three machines: each machine has 8 GPUs (24 GPUs in total):
```
Node-0: python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                            --pretrained_model_path models/google_zh_model.bin \
                            --output_model_path models/output_model.bin --world_size 24 --gpu_ranks 0 1 2 3 4 5 6 7 \
                            --master_ip tcp://9.73.138.133:12345 --encoder bert --target bert
Node-1: python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                            --pretrained_model_path models/google_zh_model.bin \
                            --output_model_path models/output_model.bin --world_size 24 --gpu_ranks 8 9 10 11 12 13 14 15 \
                            --master_ip tcp://9.73.138.133:12345 --encoder bert --target bert
Node-2: python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                            --pretrained_model_path models/google_zh_model.bin \
                            --output_model_path models/output_model.bin --world_size 24 --gpu_ranks 16 17 18 19 20 21 22 23 \
                            --master_ip tcp://9.73.138.133:12345 --encoder bert --target bert
```

#### Pretraining model size
In general, large model can achieve better results but lead to more resource consumption. We can specify the pre-trained model size by *--config_path* . Commonly-used configuration files are included in *models* folder. For example, we provide 4 configuration files for BERT model. They are *bert_large_config.json* , *bert_base_config.json* , *bert_small_config.json* , and *bert_tiny_config.json* . We provide different pre-trained models according to different configuration files. See model zoo for more details.
The example of doing incremental pre-training upon BERT-large model:
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --pretrained_model_path models/mixed_corpus_bert_large_model.bin --config_path models/bert_large_config.json \
                    --output_model_path models/output_model.bin --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --encoder bert --target bert
```
The example of doing incremental pre-training upon BERT-small model:
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --pretrained_model_path models/mixed_corpus_bert_small_model.bin --config_path models/bert_small_config.json \
                    --output_model_path models/output_model.bin --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --encoder bert --target bert
```
The example of doing incremental pre-training upon BERT-tiny model:
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --pretrained_model_path models/mixed_corpus_bert_tiny_model.bin --config_path models/bert_tiny_config.json \
                    --output_model_path models/output_model.bin --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --encoder bert --target bert
```

#### Word-based pre-training model
UER-py provides word-based pre-training model. We can download *wiki_bert_word_model.bin* and its vocabulary *wiki_word_vocab.txt* from model zoo. <br>
The following steps show an example of doing incremental pre-training upon *wiki_bert_word_model.bin* : <br>
Suppose that the training corpus is news data from People's Daily. First we do segmentation and obtain *rmrb_seg_bert.txt* . *rmrb_seg_bert.txt* is of bert format and words are separated by space. Then we build vocabulary upon the corpus:
```
python3 scripts/build_vocab.py --corpus_path corpora/rmrb_seg_bert.txt --vocab_path models/rmrb_word_vocab.txt --tokenizer space --min_count 50
```
Then we adapt the pre-trained model *wiki_bert_word_model.bin* . Embedding layer and output layer before softmax are adapted according to the difference between the old vocabulary and the new vocabulary. New embeddings are randomly initialized:
```
python3 scripts/dynamic_vocab_adapter.py --old_model_path models/wiki_bert_word_model.bin --old_vocab_path models/wiki_word_vocab.txt \
                                         --new_vocab_path models/rmrb_word_vocab.txt --new_model_path models/rmrb_bert_word_model.bin
```
Finally, we do incremental pre-training upon the adapted model *rmrb_bert_word_model.bin* :
```
python3 preprocess.py --corpus_path corpora/rmrb_seg_bert.txt --vocab_path models/rmrb_word_vocab.txt \
                      --dataset_path rmrb_word_dataset.pt --processes_num 8 \
                      --target bert --tokenizer space --dynamic_masking --seq_length 256

python3 pretrain.py --dataset_path rmrb_word_dataset.pt --vocab_path models/rmrb_word_vocab.txt \
                    --pretrained_model_path models/rmrb_bert_word_model.bin \
                    --output_model_path models/rmrb_bert_word_incremental_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 250000 --save_checkpoint_steps 50000 --report_steps 1000 \
                    --encoder bert --target bert
```
<br>

## Pretrain models with different encoders and targets
UER-py allows users to combine different components (e.g. embeddings, encoders, and targets). Here are some examples of trying different combinations.

#### RoBERTa
The example of pre-processing and pre-training for RoBERTa:
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_zh_vocab.txt \
                      --dataset_path dataset.pt --processes_num 8 \
                      --dynamic_masking --target mlm
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --output_model_path models/output_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --learning_rate 1e-4 --encoder bert --target mlm
```
RoBERTa uses dynamic masking, mlm target, and allows a sample to contain contents from multiple documents. <br>
We don't recommend to use *--full_sentences* when the document is short (e.g. reviews). <br>
Notice that RoBERTa removes NSP target. The corpus for RoBERTa stores one document per line, which is different from corpus used by BERT. <br>
RoBERTa can load BERT models for incremental pre-training (and vice versa). The example of doing incremental pre-training upon existing BERT model:
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_zh_vocab.txt \
                      --dataset_path dataset.pt --processes_num 8 \
                      --dynamic_masking --target mlm
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --pretrained_model_path models/google_zh_model.bin \
                    --output_model_path models/output_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --learning_rate 2e-5 --encoder bert --target mlm
```

#### ALBERT
The example of pre-processing and pre-training for ALBERT:
```
python3 preprocess.py --corpus_path corpora/book_review_bert.txt --vocab_path models/google_zh_vocab.txt \
                      --dataset_path dataset.pt --processes_num 8 --target albert
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --output_model_path models/output_model.bin \
                    --config_path models/albert_base_config.json \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --learning_rate 1e-4 \
                    --factorized_embedding_parameterization --parameter_sharing --encoder bert --target albert
```
The corpus format of ALBERT is the identical with BERT. <br>
*--target albert* denotes that using ALBERT target, which consists of mlm and sop targets. <br>
*--factorized_embedding_parameterization* denotes that using factorized embedding parameterization to untie the embedding size from the hidden layer size. <br>
*--parameter_sharing* denotes that sharing all parameters (including feed-forward and attention parameters) across layers. <br>
we provide 4 configuration files for ALBERT model in *models* folder, albert_base_config.json , albert_large_config.json , albert_xlarge_config.json , albert_xxlarge_config.json . <br>
The example of doing incremental pre-training upon Google's ALBERT pre-trained models of different sizes (See model zoo for pre-trained weights):
```
python3 preprocess.py --corpus_path corpora/book_review_bert.txt --vocab_path models/google_zh_vocab.txt \
                      --dataset_path dataset.pt --processes_num 8 --target albert 
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --pretrained_model_path models/google_zh_albert_base_model.bin \
                    --output_model_path models/output_model.bin \
                    --config_path models/albert_base_config.json \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --learning_rate 2e-5 \
                    --factorized_embedding_parameterization --parameter_sharing  --encoder bert --target albert
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --pretrained_model_path models/google_zh_albert_xxlarge_model.bin \
                    --output_model_path models/output_model.bin \
                    --config_path models/albert_xxlarge_config.json \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --learning_rate 2e-5 \
                    --factorized_embedding_parameterization --parameter_sharing --encoder bert --target albert
```

#### SpanBERT
SpanBERT introduces span masking and span boundary objective. We only consider span masking here.
The example of pre-processing and pre-training for SpanBERT (static masking):
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_zh_vocab.txt \
                      --dataset_path dataset.pt --processes_num 8 --target mlm --dup_factor 20 \
                      --span_masking --span_geo_prob 0.3 --span_max_length 5 --target mlm
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --output_model_path models/output_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7  --learning_rate 1e-4 \
                    --total_steps 10000 --save_checkpoint 5000 --encoder bert --target mlm
```
*--dup_factor* specifies the number of times to duplicate the input data (with different masks). The default value is 5 .
The example of pre-processing and pre-training for SpanBERT (dynamic masking):
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_zh_vocab.txt \
                      --dataset_path dataset.pt --processes_num 8 \
                      --dynamic_masking --target mlm
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --output_model_path models/output_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7  --learning_rate 1e-4 \
                    --span_masking --span_geo_prob 0.3 --span_max_length 5 \
                    --total_steps 10000 --save_checkpoint 5000 --encoder bert --target mlm
```

#### GPT
The example of pre-processing and pre-training for GPT:
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_zh_vocab.txt \
                      --dataset_path dataset.pt --processes_num 8 --target lm
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --output_model_path models/output_model.bin \
                    --config_path models/bert_base_config.json --learning_rate 1e-4 \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --encoder gpt --target lm
```
The corpus format of GPT is the identical with RoBERTa. We can pre-training GPT through *--encoder gpt* and *--target lm*.
GPT can use the configuration file of BERT.

#### ELMo
The example of pre-processing and pre-training for ELMo:
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_zh_vocab.txt \
                      --dataset_path dataset.pt --processes_num 8 --target bilm
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --output_model_path models/output_model.bin \
                    --config_path models/birnn_config.json --learning_rate 5e-4 \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --embedding word --encoder bilstm --target bilm
```
The corpus format of ELMo is the identical with GPT. We can pre-training ELMo through *--embedding word*, *--encoder bilstm*, and *--target bilm*. <br>
*--embedding word* denotes using traditional word embedding. LSTM does not require position embedding.

#### More combinations
The example of using LSTM encoder and LM target for pre-training:
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_zh_vocab.txt --dataset_path dataset.pt --processes_num 8 --target lm
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --output_model_path models/output_model.bin \
                    --config_path models/rnn_config.json --learning_rate 1e-3 \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --total_steps 20000 --save_checkpoint_steps 5000 \
                    --embedding word --encoder lstm --target lm
```
We use the *models/rnn_config.json* as configuration file.

The example of using GRU encoder and LM target for pre-training:
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_zh_vocab.txt --dataset_path dataset.pt --processes_num 8 --target lm
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --output_model_path models/output_model.bin \
                    --config_path models/rnn_config.json --learning_rate 1e-3 \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --total_steps 20000 --save_checkpoint_steps 5000 \
                    --embedding word --encoder gru --target lm
```

The example of using GatedCNN encoder and LM target for pre-training:
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_zh_vocab.txt --dataset_path dataset.pt --processes_num 8 --target lm
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --output_model_path models/output_model.bin \
                    --config_path models/gatedcnn_9_config.json --learning_rate 1e-4 \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --total_steps 20000 --save_checkpoint_steps 5000 \
                    --embedding word --encoder gatedcnn --target lm
```
<br>

## Finetune on downstream tasks
Currently, UER-py supports the many downstream tasks, including text classification, pair classification, document-based question answering, sequence labeling, machine reading comprehension, etc. The encoder used for downstream task should be coincident with the pre-trained model.

#### Classification
run_classifier.py adds two feedforward layers upon encoder layer.
```
usage: run_classifier.py [-h] [--pretrained_model_path PRETRAINED_MODEL_PATH]
                         [--output_model_path OUTPUT_MODEL_PATH]
                         [--vocab_path VOCAB_PATH]
                         [--spm_model_path SPM_MODEL_PATH] --train_path
                         TRAIN_PATH --dev_path DEV_PATH
                         [--test_path TEST_PATH] [--config_path CONFIG_PATH]
                         [--batch_size BATCH_SIZE] [--seq_length SEQ_LENGTH]
                         [--embedding {bert,word}]
                         [--encoder {bert,lstm,gru,cnn,gatedcnn,attn,synt,rcnn,crnn,gpt,bilstm}]
                         [--bidirectional] [--pooling {mean,max,first,last}]
                         [--factorized_embedding_parameterization]
                         [--parameter_sharing] [--tokenizer {bert,char,space}]
                         [--soft_targets] [--soft_alpha SOFT_ALPHA]
                         [--learning_rate LEARNING_RATE] [--warmup WARMUP]
                         [--fp16] [--fp16_opt_level {O0,O1,O2,O3}]
                         [--dropout DROPOUT] [--epochs_num EPOCHS_NUM]
                         [--report_steps REPORT_STEPS] [--seed SEED]
```
The example of using *run_classifier.py*：
```
python3 run_classifier.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                          --train_path datasets/douban_book_review/train.tsv --dev_path datasets/douban_book_review/dev.tsv --test_path datasets/douban_book_review/test.tsv \
                          --epochs_num 3 --batch_size 64 --encoder bert
```
The example of using *run_classifier.py* for pair classification:
```
python3 run_classifier.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                          --train_path datasets/lcqmc/train.tsv --dev_path datasets/lcqmc/dev.tsv --test_path datasets/lcqmc/test.tsv \
                          --epochs_num 3 --batch_size 64 --encoder bert
```
The example of using *inference/run_classifier_infer.py* to do inference:
```
python3 inference/run_classifier_infer.py --load_model_path models/classifier_model.bin --vocab_path models/google_zh_vocab.txt \
                                          --test_path datasets/douban_book_review/test_nolabel.tsv \
                                          --prediction_path datasets/douban_book_review/prediction.tsv --labels_num 2 \
                                          --seq_length 128 --output_logits --output_prob --encoder bert
```
For classification, texts in *text_a* column are predicted. For pair classification, texts in *text_a* and *text_b* columns are are predicted. <br>
*--labels_num* specifies the number of labels. <br>
*--output_logits* denotes the predicted logits are outputted，whose column name is logits. <br>
*--output_prob* denotes the predicted probabilities are outputted，whose column name is prob. <br>
*--seq_length* specifies the sequence length, which should be the same with setting in training stage.

Notice that BERT and RoBERTa have the same encoder. There is no difference between loading BERT and RoBERTa.

The example of using ALBERT for classification:
```
python3 run_classifier.py --pretrained_model_path models/google_zh_albert_base_model.bin --vocab_path models/google_zh_vocab.txt \
                          --config_path models/albert_base_config.json \
                          --train_path datasets/douban_book_review/train.tsv \
                          --dev_path datasets/douban_book_review/dev.tsv \
                          --test_path datasets/douban_book_review/test.tsv \
                          --learning_rate 4e-5 \
                          --epochs_num 5 --batch_size 32 \
                          --factorized_embedding_parameterization --parameter_sharing --encoder bert
```
The performance of ALBERT is sensitive to hyper-parameter settings. <br>
The example of doing inference for ALBERT:
```
python3 inference/run_classifier_infer.py --load_model_path models/classifier_model.bin --vocab_path models/google_zh_vocab.txt \
                                          --config_path models/albert_base_config.json \
                                          --test_path datasets/douban_book_review/test_nolabel.tsv \
                                          --prediction_path datasets/douban_book_review/prediction.tsv --labels_num 2 \
                                          --factorized_embedding_parameterization --parameter_sharing --encoder bert
```

UER-py supports multi-task learning. Embedding and encoder layers are shared by different tasks. <br>
The example of training two sentiment analysis datasets:
```
python3 run_mt_classifier.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                             --dataset_path_list datasets/douban_book_review/ datasets/chnsenticorp/ \
                             --epochs_num 1 --batch_size 64 --encoder bert
```
*--dataset_path_list* specifies folder path list of different tasks. Each folder should contains train set *train.tsv* and development set *dev.tsv* .


UER-py supports distillation for classification tasks. <br>
First of all, we train a teacher model. We fine-tune upon a Chinese BERT-large model (provided in model zoo):
```
python3 run_classifier.py --pretrained_model_path models/mixed_corpus_bert_large_model.bin \
                          --vocab_path models/google_zh_vocab.txt \
                          --config_path models/bert_large_config.json \
                          --output_model_path models/teacher_classifier_model.bin \
                          --train_path datasets/douban_book_review/train.tsv \
                          --dev_path datasets/douban_book_review/dev.tsv \
                          --test_path datasets/douban_book_review/test.tsv \
                          --epochs_num 3 --batch_size 32 --encoder bert
```
Then we use the teacher model to do inference. The pesudo labels and logits are generated:
```
python3 inference/run_classifier_infer.py --load_model_path models/teacher_classifier_model.bin --vocab_path models/google_zh_vocab.txt \
                                          --config_path models/bert_large_config.json --test_path text.tsv \
                                          --prediction_path label_logits.tsv --labels_num 2 --output_logits --encoder bert
```
The input file *text.tsv* contains text to be predicted (see *datasets/douban_book_review/test_nolabel.tsv*). *text.tsv* could be downstream dataset, e.g. using *datasets/douban_book_review/train.tsv* as input (*--test_path*), or related external data. Larger transfer set often leads to better performance. <br>
The output file *label_logits.tsv* contains label column and logits column. Then we obtain *text_label_logits.tsv* by combining *text.tsv* and *label_logits.tsv* . *text_label_logits.tsv* contains text_a column (text_a column and text_b column for pair classification), label column (hard label), and logits column (soft label). <br>
Student model is a 3-layers BERT-tiny model. The pre-trained model is provided in model zoo.
Then the student model learns the outputs (hard and soft labels) of the teacher model:
```
python3 run_classifier.py --pretrained_model_path mixed_corpus_bert_tiny_model.bin --vocab_path models/google_zh_vocab.txt \
                          --config_path models/bert_tiny_config.json \
                          --train_path text_label_logits.tsv \
                          --dev_path datasets/douban_book_review/dev.tsv \
                          --test_path datasets/douban_book_review/test.tsv \
                          --epochs_num 3 --batch_size 64 --soft_targets --soft_alpha 0.5 --encoder bert
```
*--soft_targets* denotes that the model uses logits (soft label) for training. Mean-squared-error (MSE) is used as loss function. <br>
*--soft_alpha* specifies the weight of the soft label loss. The loss function is weighted average of cross-entropy loss (for hard label) and mean-squared-error loss (for soft label).

#### Document-based question answering
*run_dbqa.py* uses the same network architecture with *run_classifier.py* .
```
usage: run_dbqa.py [-h] [--pretrained_model_path PRETRAINED_MODEL_PATH]
                   [--output_model_path OUTPUT_MODEL_PATH]
                   [--vocab_path VOCAB_PATH] [--spm_model_path SPM_MODEL_PATH]
                   --train_path TRAIN_PATH --dev_path DEV_PATH
                   [--test_path TEST_PATH] [--config_path CONFIG_PATH]
                   [--batch_size BATCH_SIZE] [--seq_length SEQ_LENGTH]
                   [--embedding {bert,word}]
                   [--encoder {bert,lstm,gru,cnn,gatedcnn,attn,synt,rcnn,crnn,gpt,bilstm}]
                   [--bidirectional] [--pooling {mean,max,first,last}]
                   [--factorized_embedding_parameterization]
                   [--parameter_sharing] [--tokenizer {bert,char,space}]
                   [--soft_targets] [--soft_alpha SOFT_ALPHA]
                   [--learning_rate LEARNING_RATE] [--warmup WARMUP] [--fp16]
                   [--fp16_opt_level {O0,O1,O2,O3}] [--dropout DROPOUT]
                   [--epochs_num EPOCHS_NUM] [--report_steps REPORT_STEPS]
                   [--seed SEED]
```
The document-based question answering (DBQA) can be converted to classification task. Column text_a contains question and column text_b contains sentence which may has answer.
The example of using *run_dbqa.py*:
```
python3 run_dbqa.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                    --train_path datasets/nlpcc-dbqa/train.tsv \
                    --dev_path datasets/nlpcc-dbqa/dev.tsv \
                    --test datasets/nlpcc-dbqa/test.tsv \
                    --epochs_num 3 --batch_size 64 --encoder bert
```
The example of using *inference/run_classifier_infer.py* to do inference for DBQA:
```
python3 inference/run_classifier_infer.py --load_model_path models/dbqa_model.bin --vocab_path models/google_zh_vocab.txt \
                                          --test_path datasets/nlpcc-dbqa/test_nolabel.tsv \
                                          --prediction_path datasets/nlpcc-dbqa/prediction.tsv --labels_num 2 \
                                          --output_logits --output_prob --encoder bert
```
The example of using ALBERT for DBQA:
```
python3 run_dbqa.py --pretrained_model_path models/google_zh_albert_base_model.bin --vocab_path models/google_zh_vocab.txt \
                    --config_path models/albert_base_config.json \
                    --train_path datasets/nlpcc-dbqa/train.tsv \
                    --dev_path datasets/nlpcc-dbqa/dev.tsv \
                    --test datasets/nlpcc-dbqa/test.tsv \
                    --epochs_num 3 --batch_size 64 \
                    --factorized_embedding_parameterization --parameter_sharing --encoder bert
```
The example of doing inference for ALBERT:
```
python3 inference/run_classifier_infer.py --load_model_path models/dbqa_model.bin --vocab_path models/google_zh_vocab.txt \
                                          --config_path models/albert_base_config.json \
                                          --test_path datasets/nlpcc-dbqa/test_nolabel.tsv \
                                          --prediction_path datasets/nlpcc-dbqa/prediction.tsv --labels_num 2 \
                                          --factorized_embedding_parameterization --parameter_sharing --encoder bert
```

#### Sequence labeling
*run_ner.py* adds one feedforward layer upon encoder layer.
```
usage: run_ner.py [-h] [--pretrained_model_path PRETRAINED_MODEL_PATH]
                  [--output_model_path OUTPUT_MODEL_PATH]
                  [--vocab_path VOCAB_PATH] [--spm_model_path SPM_MODEL_PATH]
                  --train_path TRAIN_PATH --dev_path DEV_PATH
                  [--test_path TEST_PATH] [--config_path CONFIG_PATH]
                  --label2id_path LABEL2ID_PATH [--batch_size BATCH_SIZE]
                  [--seq_length SEQ_LENGTH] [--embedding {bert,word}]
                  [--encoder {bert,lstm,gru,cnn,gatedcnn,attn,synt,rcnn,crnn,gpt,bilstm}]
                  [--bidirectional] [--factorized_embedding_parameterization]
                  [--parameter_sharing] [--learning_rate LEARNING_RATE]
                  [--warmup WARMUP] [--fp16] [--fp16_opt_level {O0,O1,O2,O3}]
                  [--dropout DROPOUT] [--epochs_num EPOCHS_NUM]
                  [--report_steps REPORT_STEPS] [--seed SEED]
```
The example of using *run_ner.py*:
```
python3 run_ner.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                   --train_path datasets/msra_ner/train.tsv --dev_path datasets/msra_ner/dev.tsv --test_path datasets/msra_ner/test.tsv \
                   --label2id_path datasets/msra_ner/label2id.json --epochs_num 5 --batch_size 16 --encoder bert
```
The example of doing inference:
```
python3 inference/run_ner_infer.py --load_model_path models/ner_model.bin --vocab_path models/google_zh_vocab.txt \
                                          --test_path datasets/msra_ner/test_nolabel.tsv \
                                          --prediction_path datasets/msra_ner/prediction.tsv \
                                          --label2id_path datasets/msra_ner/label2id.json --encoder bert
```
The example of using ALBERT for NER:
```
python3 run_ner.py --pretrained_model_path models/google_zh_albert_base_model.bin --vocab_path models/google_zh_vocab.txt \
                   --config_path models/albert_base_config.json \
                   --train_path datasets/msra_ner/train.tsv --dev_path datasets/msra_ner/dev.tsv --test_path datasets/msra_ner/test.tsv \
                   --label2id_path datasets/msra_ner/label2id.json --epochs_num 5 --batch_size 16 \
                   --learning_rate 1e-4 --factorized_embedding_parameterization --parameter_sharing --encoder bert
```
The example of doing inference for ALBERT:
```
python3 inference/run_ner_infer.py --load_model_path models/ner_model.bin --vocab_path models/google_zh_vocab.txt \
                                          --config_path models/albert_base_config.json \
                                          --test_path datasets/msra_ner/test_nolabel.tsv \
                                          --prediction_path datasets/msra_ner/prediction.tsv \
                                          --label2id_path datasets/msra_ner/label2id.json \
                                          --factorized_embedding_parameterization --parameter_sharing --encoder bert
```

#### Machine reading comprehension
run_cmrc.py adds two feedforward layers upon encoder layer.
```
usage: run_cmrc.py [-h] [--pretrained_model_path PRETRAINED_MODEL_PATH]
                   [--output_model_path OUTPUT_MODEL_PATH]
                   [--vocab_path VOCAB_PATH] [--spm_model_path SPM_MODEL_PATH]
                   --train_path TRAIN_PATH --dev_path DEV_PATH
                   [--test_path TEST_PATH] [--config_path CONFIG_PATH]
                   [--batch_size BATCH_SIZE] [--seq_length SEQ_LENGTH]
                   [--doc_stride DOC_STRIDE] [--embedding {bert,word}]
                   [--encoder {bert,lstm,gru,cnn,gatedcnn,attn,synt,rcnn,crnn,gpt,bilstm}]
                   [--bidirectional] [--factorized_embedding_parameterization]
                   [--parameter_sharing] [--learning_rate LEARNING_RATE]
                   [--warmup WARMUP] [--fp16] [--fp16_opt_level {O0,O1,O2,O3}]
                   [--dropout DROPOUT] [--epochs_num EPOCHS_NUM]
                   [--report_steps REPORT_STEPS] [--seed SEED]
```
The example of using *run_cmrc.py* for Chinese Machine Reading Comprehension (CMRC):
```
python3 run_cmrc.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                   --train_path datasets/cmrc2018/train.json --dev_path datasets/cmrc2018/dev.json \
                   --epochs_num 2 --batch_size 8 --seq_length 512 --encoder bert
```
The *train.json* and *dev.json* are of squad-style. Train set and development set are available [here](https://github.com/ymcui/cmrc2018). *--test_path* option is not specified since test set is not publicly available.

The example of doing inference:
```
python3  inference/run_cmrc_infer.py --load_model_path models/cmrc_model.bin --vocab_path models/google_zh_vocab.txt \
                                     --test_path datasets/cmrc2018/test.json \
                                     --prediction_path datasets/cmrc2018/prediction.json --encoder bert
```
The example of using ALBERT-xxlarge for CMRC:
```
python3 run_cmrc.py --pretrained_model_path models/google_zh_albert_xxlarge_model.bin \
                    --vocab_path models/google_zh_vocab.txt \
                    --config_path models/albert_xxlarge_config.json \
                    --train_path datasets/cmrc2018/train.json --dev_path datasets/cmrc2018/dev.json \
                    --epochs_num 2 --batch_size 8 --seq_length 512 --learning_rate 1e-5 \
                    --factorized_embedding_parameterization --parameter_sharing --encoder bert
```
The example of doing inference for ALBERT:
```
python3 inference/run_cmrc_infer.py --load_model_path models/cmrc_model.bin --vocab_path models/google_zh_vocab.txt \
                                     --config_path models/albert_xxlarge_config.json \
                                     --test_path datasets/cmrc2018/test.json \
                                     --prediction_path datasets/cmrc2018/prediction.json \
                                     --factorized_embedding_parameterization --parameter_sharing --encoder bert
```

#### Multiple choice
run_c3.py adds one feedforward layer upon encoder layer.
```
usage: run_c3.py [-h] [--pretrained_model_path PRETRAINED_MODEL_PATH]
                 [--output_model_path OUTPUT_MODEL_PATH]
                 [--vocab_path VOCAB_PATH] [--spm_model_path SPM_MODEL_PATH]
                 --train_path TRAIN_PATH --dev_path DEV_PATH
                 [--test_path TEST_PATH] [--config_path CONFIG_PATH]
                 [--batch_size BATCH_SIZE] [--seq_length SEQ_LENGTH]
                 [--embedding {bert,word}]
                 [--encoder {bert,lstm,gru,cnn,gatedcnn,attn,synt,rcnn,crnn,gpt,bilstm}]
                 [--bidirectional] [--factorized_embedding_parameterization]
                 [--parameter_sharing] [--max_choices_num MAX_CHOICES_NUM]
                 [--tokenizer {bert,char,space}]
                 [--learning_rate LEARNING_RATE] [--warmup WARMUP] [--fp16]
                 [--fp16_opt_level {O0,O1,O2,O3}] [--dropout DROPOUT]
                 [--epochs_num EPOCHS_NUM] [--report_steps REPORT_STEPS]
                 [--seed SEED]
```
The example of using *run_cmrc.py* for multiple choice task:
```
python3 run_c3.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                  --train_path datasets/c3/train.json --dev_path datasets/c3/dev.json \
                  --epochs_num 8 --batch_size 16 --seq_length 512 --max_choices_num 4 --encoder bert
```
*--test_path* option is not specified since test set of C3 dataset is not publicly available. <br>
The actual batch size is *--batch_size* times *--max_choices_num* . <br>
The question in C3 dataset contains at most 4 candidate answers. *--max_choices_num* is set to 4.

The example of doing inference:
```
python3 inference/run_c3_infer.py --load_model_path models/multichoice_model.bin --vocab_path models/google_zh_vocab.txt \
                                  --test_path datasets/c3/test.json \
                                  --prediction_path datasets/c3/prediction.json --max_choices_num 4 --encoder bert
```
The example of using ALBERT-xlarge for C3:
```
python3 run_c3.py --pretrained_model_path models/google_zh_albert_xlarge_model.bin --vocab_path models/google_zh_vocab.txt \
                  --config_path models/albert_xlarge_config.json \
                  --train_path datasets/c3/train.json --dev_path datasets/c3/dev.json \
                  --epochs_num 8 --batch_size 8 --seq_length 512 --max_choices_num 4 \
                  --factorized_embedding_parameterization --parameter_sharing --encoder bert
```

The example of doing inference for ALBERT-large:
```
python3  inference/run_c3_infer.py --load_model_path models/multichoice_model.bin --vocab_path models/google_zh_vocab.txt \
                                   --config_path models/albert_xlarge_config.json \
                                   --test_path datasets/c3/test.json \
                                   --prediction_path datasets/c3/prediction.json --max_choices_num 4 \
                                   --factorized_embedding_parameterization --parameter_sharing --encoder bert
```
<br>

## Tokenization and vocabulary
UER-py supports multiple tokenization strategies. The most commonly used strategy is BertTokenizer (which is also the default strategy). There are two ways to use BertTokenizer: the first is to specify the vocabulary path through *--vocab_path* and then use BERT's original tokenization strategy to segment sentences according to the vocabulary; the second is to specify the sentencepiece model path by *--spm_model_path* . We import sentencepiece, load the sentencepiece model, and segment the sentence. If user specifies *--spm_model_path*, sentencepiece is used for tokenization. Otherwise, user must specify *--vocab_path* and BERT's original tokenization strategy is used for tokenization. <br>
In addition, the project also provides CharTokenizer and SpaceTokenizer. CharTokenizer tokenizes the text by character. If the text is all Chinese character, CharTokenizer and BertTokenizer are equivalent. CharTokenizer is simple and is faster than BertTokenizer. SpaceTokenizer separates the text by space. One can preprocess the text in advance (such as word segmentation), separate the text by space, and then use SpaceTokenizer. If user specifies *--spm_model_path*, sentencepiece is used for tokenization. Otherwise, user must specify *--vocab_path* and BERT's original tokenization strategy is used for tokenization. For CharTokenizer and SpaceTokenizer, if user specifies *--spm_model_path*, then the vocabulary in sentencepiece model is used. Otherwise, user must specify the vocabulary through *--vocab_path*.


The pre-processing, pre-training, and fine-tuning stages all need vocabulary, which is provided through *--vocab_path* or *--smp_model_path*. If you use your own vocabulary, you need to ensure the following: 1) The ID of the padding character is 0; 2) The starting character, separator character, and mask character are "[CLS]", "[SEP]", "[MASK]"; 3) If *--vocab_path* is specified, the unknown character is "[UNK]". If *--spm_model_path* is spcified, the unknown character is "\<unk\>" .


<br/>

## Scripts
UER-py provides abundant tool scripts for pre-training models.
This section firstly summarizes tool scripts and their functions, and then provides using examples of some scripts.

<table>
<tr align="center"><th> Script <th> Function description
<tr align="center"><td> average_model.py <td> Take the average of pre-trained models. A frequently-used ensemble strategy for deep learning models 
<tr align="center"><td> build_vocab.py <td> Build vocabulary (multi-processing supported)
<tr align="center"><td> check_model.py <td> Check the model (single GPU or multiple GPUs)

<tr align="center"><td> cloze_test.py <td> Randomly mask a word and predict it, top n words are returned
<tr align="center"><td> convert_bert_from_uer_to_google.py <td> convert the BERT of UER format to Google format (TF)
<tr align="center"><td> convert_bert_from_uer_to_huggingface.py <td> convert the BERT of UER format to Huggingface format (PyTorch)
<tr align="center"><td> convert_bert_from_google_to_uer.py <td> convert the BERT of Google format (TF) to UER format
<tr align="center"><td> convert_bert_from_huggingface_to_uer.py <td> convert the BERT of Huggingface format (PyTorch) to UER format
<tr align="center"><td> diff_vocab.py <td> Compare two vocabularies
<tr align="center"><td> dynamic_vocab_adapter.py <td> Change the pre-trained model according to the vocabulary. It can save memory in fine-tuning stage since task-specific vocabulary is much smaller than general-domain vocabulary 
<tr align="center"><td> extract_embedding.py <td> extract the embedding of the pre-trained model
<tr align="center"><td> extract_feature.py <td> extract the hidden states of the last of the pre-trained model
<tr align="center"><td> topn_words_indep.py <td> Finding nearest neighbours with context-independent word embedding
<tr align="center"><td> topn_words_dep.py <td> Finding nearest neighbours with context-dependent word embedding
</table>


### Cloze test
cloze_test.py predicts masked words. Top n words are returned.
```
usage: cloze_test.py [-h] [--pretrained_model_path PRETRAINED_MODEL_PATH]
                [--vocab_path VOCAB_PATH] [--input_path INPUT_PATH]
                [--output_path OUTPUT_PATH] [--config_path CONFIG_PATH]
                [--batch_size BATCH_SIZE] [--seq_length SEQ_LENGTH]
                [--encoder {bert,lstm,gru,cnn,gatedcnn,attn,rcnn,crnn,gpt}]
                [--bidirectional] [--target {bert,lm,cls,mlm,nsp,s2s}]
                [--subword_type {none,char}] [--sub_vocab_path SUB_VOCAB_PATH]
                [--subencoder_type {avg,lstm,gru,cnn}]
                [--tokenizer {bert,char,word,space}] [--topn TOPN]
```
The example of using cloze_test.py：
```
python3 scripts/cloze_test.py --input_path datasets/cloze_input.txt --pretrained_model_path models/google_zh_model.bin \
                              --vocab_path models/google_zh_vocab.txt --output_path output.txt

```

### Feature extractor
extract_feature.py extracts hidden states of the last encoder layer.
```
usage: extract_feature.py [-h] --input_path INPUT_PATH --pretrained_model_path
                          PRETRAINED_MODEL_PATH --vocab_path VOCAB_PATH
                          --output_path OUTPUT_PATH [--seq_length SEQ_LENGTH]
                          [--batch_size BATCH_SIZE]
                          [--config_path CONFIG_PATH]
                          [--embedding {bert,word}]
                          [--encoder {bert,lstm,gru,cnn,gatedcnn,attn,rcnn,crnn,gpt}]
                          [--bidirectional] [--subword_type {none,char}]
                          [--sub_vocab_path SUB_VOCAB_PATH]
                          [--subencoder {avg,lstm,gru,cnn}]
                          [--sub_layers_num SUB_LAYERS_NUM]
                          [--tokenizer {bert,char,space}]
```
The example of using extract_feature.py：
```
python3 scripts/extract_feature.py --input_path datasets/cloze_input.txt --vocab_path models/google_zh_vocab.txt \
                                   --pretrained_model_path models/google_zh_model.bin --output_path feature_output.pt
```

### Finding nearest neighbours
Pre-trained models can learn high-quality word embeddings. Traditional word embeddings such as word2vec and GloVe assign each word a fixed vector (context-independent word embedding). However, polysemy is a pervasive phenomenon in human language, and the meanings of a polysemous word depend on the context. To this end, we use a the hidden state in pre-trained models to represent a word. It is noticeable that Google BERT is a character-based model. To obtain real word embedding (not character embedding), Users should download our [word-based BERT model](https://share.weiyun.com/5s4HVMi) and [vocabulary](https://share.weiyun.com/5NWYbYn).
The example of using scripts/topn_words_indep.py to find nearest neighbours for context-independent word embedding (character-based and word-based models)：
```
python3 scripts/topn_words_indep.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                                    --cand_vocab_path models/google_zh_vocab.txt --target_words_path target_words.txt
python3 scripts/topn_words_indep.py --pretrained_model_path models/bert_wiki_word_model.bin --vocab_path models/wiki_word_vocab.txt \
                                    --cand_vocab_path models/wiki_word_vocab.txt --target_words_path target_words.txt
```
Context-independent word embedding is obtained by model's embedding layer.
The format of the target_words.txt is as follows:
```
word-1
word-2
...
word-n
```
The example of using scripts/topn_words_dep.py to find nearest neighbours for context-dependent word embedding (character-based and word-based models)：
```
python3 scripts/topn_words_dep.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                                  --cand_vocab_path models/google_zh_vocab.txt --sent_path target_words_with_sentences.txt --config_path models/bert_base_config.json \
                                  --batch_size 256 --seq_length 32 --tokenizer bert
python3 scripts/topn_words_dep.py --pretrained_model_path models/bert_wiki_word_model.bin --vocab_path models/wiki_word_vocab.txt \
                                  --cand_vocab_path models/wiki_word_vocab.txt --sent_path target_words_with_sentences.txt --config_path models/bert_base_config.json \
                                  --batch_size 256 --seq_length 32 --tokenizer space
```
We substitute the target word with other words in the vocabulary and feed the sentences into the pretrained model. Hidden state is used as the context-dependent embedding of a word. Users should do word segmentation manually and use space tokenizer if word-based model is used. The format of 
target_words_with_sentences.txt is as follows:
```
sent1 word1
sent1 word1
...
sentn wordn
```
Sentence and word are splitted by \t. 

### Text generator
We could use *generate.py* to generate text. Given a few words or sentences, *generate.py* can continue writing. The example of using *generate.py*:
```
python3 scripts/generate.py --pretrained_model_path models/gpt_model.bin --vocab_path models/google_zh_vocab.txt 
                            --input_path story_beginning.txt --output_path story_full.txt --config_path models/bert_base_config.json 
                            --encoder gpt --target lm --seq_length 128  
```
where *story_beginning* contains the beginning of a text. One can use any models pre-trained with LM target, such as [GPT trained on mixed large corpus](https://share.weiyun.com/51nTP8V). By now we only provide a vanilla version of generator. More mechanisms will be added for better performance and efficiency.


<br/>