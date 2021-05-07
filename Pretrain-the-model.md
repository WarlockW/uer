```
sage: pretrain.py [-h] [--dataset_path DATASET_PATH]
                   [--vocab_path VOCAB_PATH] [--spm_model_path SPM_MODEL_PATH]
                   [--tgt_vocab_path TGT_VOCAB_PATH]
                   [--tgt_spm_model_path TGT_SPM_MODEL_PATH]
                   [--pretrained_model_path PRETRAINED_MODEL_PATH]
                   --output_model_path OUTPUT_MODEL_PATH
                   [--config_path CONFIG_PATH] [--total_steps TOTAL_STEPS]
                   [--save_checkpoint_steps SAVE_CHECKPOINT_STEPS]
                   [--report_steps REPORT_STEPS]
                   [--accumulation_steps ACCUMULATION_STEPS]
                   [--batch_size BATCH_SIZE]
                   [--instances_buffer_size INSTANCES_BUFFER_SIZE]
                   [--labels_num LABELS_NUM] [--dropout DROPOUT] [--seed SEED]
                   [--embedding {word,word_pos,word_pos_seg,word_sinusoidalpos}]
                   [--remove_embedding_layernorm]
                   [--encoder {transformer,rnn,lstm,gru,birnn,bilstm,bigru,gatedcnn}]
                   [--mask {fully_visible,causal}]
                   [--layernorm_positioning {pre,post}] [--bidirectional]
                   [--factorized_embedding_parameterization]
                   [--parameter_sharing]
                   [--tgt_embedding {word,word_pos,word_pos_seg,word_sinusoidalpos}]
                   [--decoder {transformer}] [--pooling {mean,max,first,last}]
                   [--target {bert,lm,mlm,bilm,albert,mt,t5,cls}]
                   [--tie_weights] [--has_lmtarget_bias] [--span_masking]
                   [--span_geo_prob SPAN_GEO_PROB]
                   [--span_max_length SPAN_MAX_LENGTH]
                   [--learning_rate LEARNING_RATE] [--warmup WARMUP] [--fp16]
                   [--fp16_opt_level {O0,O1,O2,O3}] [--beta1 BETA1]
                   [--beta2 BETA2] [--world_size WORLD_SIZE]
                   [--gpu_ranks GPU_RANKS [GPU_RANKS ...]]
                   [--master_ip MASTER_IP] [--backend {nccl,gloo}]
```
It is recommended to explicitly specify model's embedding (*--embedding*), encoder (*--encoder*), and target (*--target*). UER-py consists of the following encoder modules:
- lstm: LSTM
- gru: GRU
- bilstm: bi-LSTM (different from *--encoder lstm* with *--bidirectional* , see [the issue](https://github.com/pytorch/pytorch/issues/4930) for more details)
- gatedcnn: GatedCNN
- transformer: BERT (*--encoder transformer --mask fully_visible*); GPT (*--encoder transformer --mask causal*); GPT-2 (*--encoder transformer --mask causal --layernorm_positioning pre*)

The target should be coincident with the target in pre-processing stage. Users can try different combinations of encoders and targets by *--encoder* and *--target* . More use cases are found in [Pretraining model examples](https://github.com/dbiir/UER-py/wiki/Pretraining-model-examples).

*--config_path* denotes the path of the configuration file, which specifies the hyper-parameters of the pre-training model. We have put the commonly-used configuration files in *models* folder. Users should choose the proper one according to the encoder they use. <br>
*--instances_buffer_size* specifies the buffer size in memory in pre-training stage. <br>
*--remove_embedding_layernorm* removes the layernorm layer behind the embedding layer. <br>
*--layernorm_positioning* denotes the pre-layernorm is used. This option can be used with *--remove_embedding_layernorm* together to specify the configuration of layernorm to reproduce the models such as GPT-2. <br>
*--tie_weights* denotes the word embedding and softmax weights are tied. <br>

There are two strategies for parameter initialization of pre-training: 1）random initialization; 2）loading a pre-trained model.
#### Random initialization
The example of pre-training on CPU：
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --output_model_path models/output_model.bin \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert
```
The input of pre-training is specified by *--dataset_path* .
The example of pre-training on single GPU (the ID of GPU is 3)：
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --output_model_path models/output_model.bin --gpu_ranks 3 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert
```
The example of pre-training on a single machine with 8 GPUs：
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --output_model_path models/output_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert
```
*--world_size* specifies the number of processes (and GPUs) used for pre-training. <br>
*--gpu_ranks* specifies the ID for each process and GPU. The IDs are from *0* to *n-1*, where *n* is the number of processes used for pre-training. <br>
Users could use CUDA_VISIBLE_DEVICES if they want to use part of GPUs:
```
CUDA_VISIBLE_DEVICES=1,2,3,5 python3 pretrain.py --dataset_path dataset.pt \
                                                 --vocab_path models/google_zh_vocab.txt \
                                                 --output_model_path models/output_model.bin \
                                                 --world_size 4 --gpu_ranks 0 1 2 3 \
                                                 --embedding word_pos_seg \
                                                 --encoder transformer --mask fully_visible \
                                                 --target bert
```
*--world_size* is set to 4 since only 4 GPUs are used. The IDs of 4 processes (and GPUs) is 0, 1, 2, and 3, which are specified by *--gpu_ranks* .

The example of pre-training on two machines: each machine has 8 GPUs (16 GPUs in total).
We run *pretrain.py* on two machines (Node-0 and Node-1) respectively. *--master_ip* specifies the ip:port of the master mode, which contains process (and GPU) of ID 0.
```
Node-0 : python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                             --output_model_path models/output_model.bin \
                             --world_size 16 --gpu_ranks 0 1 2 3 4 5 6 7 \
                             --total_steps 100000 --save_checkpoint_steps 10000 --report_steps 100 \
                             --master_ip tcp://9.73.138.133:12345 \
                             --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert

Node-1 : python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                             --output_model_path models/output_model.bin \
                             --world_size 16 --gpu_ranks 8 9 10 11 12 13 14 15 \
                             --total_steps 100000 \
                             --master_ip tcp://9.73.138.133:12345 \
                             --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert
```
The IP of Node-0 is 9.73.138.133 . <br>
*--total_steps* specifies the training steps. <br>
*--save_checkpoint_steps* specifies how often to save the model checkpoint. We don't need to specify *--save_checkpoint_steps* in Node-1 since only master node saves the pre-trained model. <br>
*--report_steps* specifies how often to report the pre-training information. We don't need to specify *--report_steps* in Node-1 since the information only appears in master node. <br>
Notice that when specifying *--master_ip* one can not select the port which is occupied by other programs. <br>
For random initialization, pre-training usually requires larger learning rate. We recommend to use *--learning_rate 1e-4* . The default value is *2e-5* .

#### Load the pre-trained model
We recommend to load a pre-trained model. We can specify the pre-trained model by *--pretrained_model_path* .
The example of pre-training on CPU and single GPU:
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --pretrained_model_path models/google_zh_model.bin \
                    --output_model_path models/output_model.bin \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert

python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --pretrained_model_path models/google_zh_model.bin \
                    --output_model_path models/output_model.bin --gpu_ranks 3 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert
```
The example of pre-training on a single machine with 8 GPUs：
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --pretrained_model_path models/google_zh_model.bin \
                    --output_model_path models/output_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert
```
The example of pre-training on two machines: each machine has 8 GPUs (16 GPUs in total):
```
Node-0 : python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                             --pretrained_model_path models/google_zh_model.bin \
                             --output_model_path models/output_model.bin \
                             --world_size 16 --gpu_ranks 0 1 2 3 4 5 6 7 \
                             --master_ip tcp://9.73.138.133:12345 \
                             --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert

Node-1 : python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                             --pretrained_model_path models/google_zh_model.bin \
                             --output_model_path models/output_model.bin \
                             --world_size 16 --gpu_ranks 8 9 10 11 12 13 14 15 \
                             --master_ip tcp://9.73.138.133:12345 \
                             --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert
```
The example of pre-training on three machines: each machine has 8 GPUs (24 GPUs in total):
```
Node-0: python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                            --pretrained_model_path models/google_zh_model.bin \
                            --output_model_path models/output_model.bin \
                            --world_size 24 --gpu_ranks 0 1 2 3 4 5 6 7 \
                            --master_ip tcp://9.73.138.133:12345 \
                            --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert

Node-1: python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                            --pretrained_model_path models/google_zh_model.bin \
                            --output_model_path models/output_model.bin \
                            --world_size 24 --gpu_ranks 8 9 10 11 12 13 14 15 \
                            --master_ip tcp://9.73.138.133:12345 \
                            --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert

Node-2: python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                            --pretrained_model_path models/google_zh_model.bin \
                            --output_model_path models/output_model.bin \
                            --world_size 24 --gpu_ranks 16 17 18 19 20 21 22 23 \
                            --master_ip tcp://9.73.138.133:12345 \
                            --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert
```

#### Pretraining model size
In general, large model can achieve better results but lead to more resource consumption. We can specify the pre-trained model size by *--config_path* . Commonly-used configuration files are included in *models* folder. For example, we provide 6 configuration files for BERT (and RoBERTa) in *models/bert/*. They are *large_config.json*, *base_config.json*, *medium_config.json*, *small_config.json*,  *mini_config.json*, *tiny_config.json* . We provide pre-trained models of different sizes. See model zoo for more details.
*models/bert/base_config.json* is used as configuration file in default.
We also provide the configuration files for other pre-trained models, e.g. *models/albert/*, *models/gpt2/*, *models/t5/* .

The example of doing incremental pre-training upon [BERT-large model](https://share.weiyun.com/5G90sMJ):
```
python3 pretrain.py --dataset_path dataset.pt \
                    --vocab_path models/google_zh_vocab.txt \
                    --pretrained_model_path models/mixed_corpus_bert_large_model.bin \
                    --config_path models/bert/large_config.json \
                    --output_model_path models/output_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert
```


#### Word-based pre-training model
UER-py provides word-based pre-training models. We can download [*wiki_bert_word_model.bin*](https://share.weiyun.com/5s4HVMi) and its vocabulary [*wiki_word_vocab.txt*](https://share.weiyun.com/5NWYbYn) from model zoo. <br>
The following steps show an example of doing incremental pre-training upon *wiki_bert_word_model.bin* : <br>
Suppose that the training corpus is *corpora/book_review.txt* . First we do segmentation and obtain *book_review_seg.txt* . *book_review_seg.txt* is of MLM target format and words are separated by space. Then we build vocabulary upon the corpus:
```
python3 scripts/build_vocab.py --corpus_path corpora/book_review_seg.txt \
                               --vocab_path models/book_review_word_vocab.txt \
                               --tokenizer space \
                               --min_count 50
```
Then we adapt the pre-trained model *wiki_bert_word_model.bin* . Embedding layer and output layer before softmax are adapted according to the difference between the old vocabulary and the new vocabulary. The embedding of new word is randomly initialized. The adapted model is compatible with the new vocabulary:
```
python3 scripts/dynamic_vocab_adapter.py --old_model_path models/wiki_bert_word_model.bin \
                                         --old_vocab_path models/wiki_word_vocab.txt \
                                         --new_vocab_path models/book_review_word_vocab.txt \
                                         --new_model_path models/book_review_word_model.bin
```
Finally, we do incremental pre-training upon the adapted model *book_review_word_model.bin* . MLM target is used:
```
python3 preprocess.py --corpus_path corpora/book_review_seg.txt \
                      --vocab_path models/book_review_word_vocab.txt \
                      --dataset_path book_review_word_dataset.pt \
                      --processes_num 8 --tokenizer space --seq_length 128 \
                      --dynamic_masking --target mlm

python3 pretrain.py --dataset_path book_review_word_dataset.pt \
                    --vocab_path models/book_review_word_vocab.txt \
                    --pretrained_model_path models/book_review_word_model.bin \
                    --config_path models/bert/base_config.json \
                    --output_model_path models/output_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 20000 --save_checkpoint_steps 10000 --report_steps 1000 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target mlm
```

In addition, We can use SentencePiece to obtain word-based pre-training model：
```
python3 preprocess.py --corpus_path corpora/book_review.txt \
                      --spm_model_path models/cluecorpussmall_spm.model \
                      --dataset_path book_review_word_dataset.pt \
                      --processes_num 8 --seq_length 128 \
                      --dynamic_masking --target mlm

python3 pretrain.py --dataset_path book_review_word_dataset.pt \
                    --spm_model_path models/cluecorpussmall_spm.model \
                    --config_path models/bert/base_config.json \
                    --output_model_path models/output_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 20000 --save_checkpoint_steps 10000 --report_steps 1000 \
                    --learning_rate 1e-4 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target mlm
```
*--spm_model_path* specifies the path of sentencepiece model. *models/cluecorpussmall_spm.model* is the sentencepiece model trained on CLUECorpusSmall corpus.
