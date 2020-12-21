### Pretrain the model
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
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert
```
The input of pre-training is specified by *--dataset_path* .
The example of pre-training on single GPU (the ID of GPU is 3)：
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --output_model_path models/output_model.bin --gpu_ranks 3 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert
```
The example of pre-training on a single machine with 8 GPUs：
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --output_model_path models/output_model.bin --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert
```
*--world_size* specifies the number of processes (and GPUs) used for pre-training. <br>
*--gpu_ranks* specifies the ID for each process and GPU. The IDs are from *0* to *n-1*, where *n* is the number of processes used for pre-training. <br>
Users could use CUDA_VISIBLE_DEVICES if they want to use part of GPUs:
```
CUDA_VISIBLE_DEVICES=1,2,3,5 python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                                                 --output_model_path models/output_model.bin --world_size 4 --gpu_ranks 0 1 2 3 \
                                                 --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert
```
*--world_size* is set to 4 since only 4 GPUs are used. The IDs of 4 processes (and GPUs) is 0, 1, 2, and 3, which are specified by *--gpu_ranks* .

The example of pre-training on two machines: each machine has 8 GPUs (16 GPUs in total).
We run *pretrain.py* on two machines (Node-0 and Node-1) respectively. *--master_ip* specifies the ip:port of the master mode, which contains process (and GPU) of ID 0.
```
Node-0 : python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                             --output_model_path models/output_model.bin --world_size 16 --gpu_ranks 0 1 2 3 4 5 6 7 \
                             --total_steps 100000 --save_checkpoint_steps 10000 --report_steps 100 \
                             --master_ip tcp://9.73.138.133:12345 \
                             --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert
Node-1 : python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                             --output_model_path models/output_model.bin --world_size 16 --gpu_ranks 8 9 10 11 12 13 14 15 \
                             --total_steps 100000 \
                             --master_ip tcp://9.73.138.133:12345 \
                             --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert
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
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --pretrained_model_path models/google_zh_model.bin \
                    --output_model_path models/output_model.bin --gpu_ranks 3 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert
```
The example of pre-training on a single machine with 8 GPUs：
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --pretrained_model_path models/google_zh_model.bin \
                    --output_model_path models/output_model.bin --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert 
```
The example of pre-training on two machines: each machine has 8 GPUs (16 GPUs in total):
```
Node-0 : python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                             --pretrained_model_path models/google_zh_model.bin \
                             --output_model_path models/output_model.bin --world_size 16 --gpu_ranks 0 1 2 3 4 5 6 7 \
                             --master_ip tcp://9.73.138.133:12345 --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert  
Node-1 : python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                             --pretrained_model_path models/google_zh_model.bin \
                             --output_model_path models/output_model.bin --world_size 16 --gpu_ranks 8 9 10 11 12 13 14 15 \
                             --master_ip tcp://9.73.138.133:12345 --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert  
```
The example of pre-training on three machines: each machine has 8 GPUs (24 GPUs in total):
```
Node-0: python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                            --pretrained_model_path models/google_zh_model.bin \
                            --output_model_path models/output_model.bin --world_size 24 --gpu_ranks 0 1 2 3 4 5 6 7 \
                            --master_ip tcp://9.73.138.133:12345 --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert
Node-1: python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                            --pretrained_model_path models/google_zh_model.bin \
                            --output_model_path models/output_model.bin --world_size 24 --gpu_ranks 8 9 10 11 12 13 14 15 \
                            --master_ip tcp://9.73.138.133:12345 --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert
Node-2: python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                            --pretrained_model_path models/google_zh_model.bin \
                            --output_model_path models/output_model.bin --world_size 24 --gpu_ranks 16 17 18 19 20 21 22 23 \
                            --master_ip tcp://9.73.138.133:12345 --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert
```

#### Pretraining model size
In general, large model can achieve better results but lead to more resource consumption. We can specify the pre-trained model size by *--config_path* . Commonly-used configuration files are included in *models* folder. For example, we provide 4 configuration files for BERT model. They are *bert_large_config.json* , *bert_base_config.json* , *bert_small_config.json* , and *bert_tiny_config.json* . We provide different pre-trained models according to different configuration files. See model zoo for more details.
The example of doing incremental pre-training upon BERT-large model:
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --pretrained_model_path models/mixed_corpus_bert_large_model.bin --config_path models/bert_large_config.json \
                    --output_model_path models/output_model.bin --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert
```
The example of doing incremental pre-training upon BERT-small model:
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --pretrained_model_path models/mixed_corpus_bert_small_model.bin --config_path models/bert_small_config.json \
                    --output_model_path models/output_model.bin --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert
```
The example of doing incremental pre-training upon BERT-tiny model:
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --pretrained_model_path models/mixed_corpus_bert_tiny_model.bin --config_path models/bert_tiny_config.json \
                    --output_model_path models/output_model.bin --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert
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
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert
```
