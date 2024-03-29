Currently, UER-py supports the many downstream tasks, including text classification, pair classification, document-based question answering, sequence labeling, machine reading comprehension, etc. The embedding, encoder, and configuration file used for downstream task should be coincident with the pre-trained model. The pre-trained models used in this section can be found in [Modelzoo](https://github.com/dbiir/UER-py/wiki/Modelzoo).

## Classification
run_classifier.py adds two feedforward layers upon encoder layer.
```
usage: run_classifier.py [-h] [--pretrained_model_path PRETRAINED_MODEL_PATH]
                         [--output_model_path OUTPUT_MODEL_PATH]
                         [--vocab_path VOCAB_PATH]
                         [--spm_model_path SPM_MODEL_PATH] --train_path
                         TRAIN_PATH --dev_path DEV_PATH
                         [--test_path TEST_PATH] [--config_path CONFIG_PATH]
                         [--embedding {word,word_pos,word_pos_seg,word_sinusoidalpos}]
                         [--max_seq_length MAX_SEQ_LENGTH]
                         [--relative_position_embedding]
                         [--relative_attention_buckets_num RELATIVE_ATTENTION_BUCKETS_NUM]
                         [--remove_embedding_layernorm]
                         [--remove_attention_scale]
                         [--encoder {transformer,rnn,lstm,gru,birnn,bilstm,bigru,gatedcnn}]
                         [--mask {fully_visible,causal,causal_with_prefix}]
                         [--layernorm_positioning {pre,post}]
                         [--feed_forward {dense,gated}]
                         [--remove_transformer_bias] [--layernorm {normal,t5}]
                         [--bidirectional]
                         [--factorized_embedding_parameterization]
                         [--parameter_sharing] [--learning_rate LEARNING_RATE]
                         [--warmup WARMUP] [--fp16]
                         [--fp16_opt_level {O0,O1,O2,O3}]
                         [--optimizer {adamw,adafactor}]
                         [--scheduler {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}]
                         [--batch_size BATCH_SIZE] [--seq_length SEQ_LENGTH]
                         [--dropout DROPOUT] [--epochs_num EPOCHS_NUM]
                         [--report_steps REPORT_STEPS] [--seed SEED]
                         [--pooling {mean,max,first,last}]
                         [--tokenizer {bert,char,space}] [--soft_targets]
                         [--soft_alpha SOFT_ALPHA]
```
The example of using *run_classifier.py*：
```
python3 finetune/run_classifier.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                                   --train_path datasets/douban_book_review/train.tsv \
                                   --dev_path datasets/douban_book_review/dev.tsv \
                                   --test_path datasets/douban_book_review/test.tsv \
                                   --epochs_num 3 --batch_size 64 \
                                   --embedding word_pos_seg --encoder transformer --mask fully_visible
```
CLS embedding is used for prediction in default (*--pooling first*).

The example of using *run_classifier.py* for pair classification:
```
python3 finetune/run_classifier.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                                   --train_path datasets/lcqmc/train.tsv \
                                   --dev_path datasets/lcqmc/dev.tsv \
                                   --test_path datasets/lcqmc/test.tsv \
                                   --epochs_num 3 --batch_size 64 \
                                   --embedding word_pos_seg --encoder transformer --mask fully_visible
```
One can download the LCQMC dataset in Datasets section and put it in *datasets* folder.

The example of using *inference/run_classifier_infer.py* to do inference:
```
python3 inference/run_classifier_infer.py --load_model_path models/finetuned_model.bin --vocab_path models/google_zh_vocab.txt \
                                          --test_path datasets/douban_book_review/test_nolabel.tsv \
                                          --prediction_path datasets/douban_book_review/prediction.tsv \
                                          --labels_num 2 --seq_length 128 --output_logits --output_prob \
                                          --embedding word_pos_seg --encoder transformer --mask fully_visible
```
For classification, texts in *text_a* column are predicted. For pair classification, texts in *text_a* and *text_b* columns are are predicted. <br>
*--labels_num* specifies the number of labels. <br>
*--output_logits* denotes the predicted logits are outputted，whose column name is logits. <br>
*--output_prob* denotes the predicted probabilities are outputted，whose column name is prob. <br>
*--seq_length* specifies the sequence length, which should be the same with setting in training stage.

Notice that BERT and RoBERTa have the same embedding and encoder. There is no difference between loading BERT and RoBERTa.

The example of using ALBERT for classification:
```
python3 finetune/run_classifier.py --pretrained_model_path models/google_zh_albert_base_model.bin \
                                   --vocab_path models/google_zh_vocab.txt \
                                   --config_path models/albert/base_config.json \
                                   --train_path datasets/douban_book_review/train.tsv \
                                   --dev_path datasets/douban_book_review/dev.tsv \
                                   --test_path datasets/douban_book_review/test.tsv \
                                   --learning_rate 4e-5 \
                                   --epochs_num 5 --batch_size 32 \
                                   --factorized_embedding_parameterization --parameter_sharing \
                                   --embedding word_pos_seg --encoder transformer --mask fully_visible
```
The performance of ALBERT is sensitive to hyper-parameter settings. Dropout is turned off in pre-training stage (See *models/albert/base_config.json*). It is recommended to set dropout to 0.1 in configuration file when fine-tuning ALBERT on downstream tasks. <br>
The example of doing inference for ALBERT:
```
python3 inference/run_classifier_infer.py --load_model_path models/finetuned_model.bin \
                                          --vocab_path models/google_zh_vocab.txt \
                                          --config_path models/albert/base_config.json \
                                          --test_path datasets/douban_book_review/test_nolabel.tsv \
                                          --prediction_path datasets/douban_book_review/prediction.tsv --labels_num 2 \
                                          --factorized_embedding_parameterization --parameter_sharing \
                                          --embedding word_pos_seg --encoder transformer --mask fully_visible
```

The example of using GPT-2 for classification:
```
python3 finetune/run_classifier.py --pretrained_model_path models/cluecorpussmall_gpt2_seq1024_model.bin \
                                   --vocab_path models/google_zh_vocab.txt \
                                   --config_path models/gpt2/config.json \
                                   --train_path datasets/douban_book_review/train.tsv \
                                   --dev_path datasets/douban_book_review/dev.tsv \
                                   --test_path datasets/douban_book_review/test.tsv \
                                   --epochs_num 3 --batch_size 32 \
                                   --embedding word_pos --remove_embedding_layernorm \
                                   --encoder transformer --mask causal --layernorm_positioning pre --pooling mean

python3 inference/run_classifier_infer.py --load_model_path models/finetuned_model.bin \
                                          --vocab_path models/google_zh_vocab.txt \
                                          --config_path models/gpt2/config.json \
                                          --test_path datasets/douban_book_review/test_nolabel.tsv \
                                          --prediction_path datasets/douban_book_review/prediction.tsv --labels_num 2 \
                                          --embedding word_pos --remove_embedding_layernorm \
                                          --encoder transformer --mask causal --layernorm_positioning pre --pooling mean
```
We use *--pooling mean* to obtain text representation. *--pooling max* and *--pooling last* can also be used in above case. *--pooling first* is not suitable since language model is used (*--mask causal*). 

The example of using LSTM for classification:
```
python3 finetune/run_classifier.py --pretrained_model_path models/cluecorpussmall_lstm_lm_model.bin \
                                   --vocab_path models/google_zh_vocab.txt --config_path models/rnn_config.json \
                                   --train_path datasets/douban_book_review/train.tsv \
                                   --dev_path datasets/douban_book_review/dev.tsv \
                                   --test_path datasets/douban_book_review/test.tsv \
                                   --learning_rate 1e-3 --batch_size 64 --epochs_num 5 \
                                   --embedding word --remove_embedding_layernorm --encoder lstm --pooling mean

python3 inference/run_classifier_infer.py --load_model_path models/finetuned_model.bin --vocab_path models/google_zh_vocab.txt \
                                          --config_path models/rnn_config.json \
                                          --test_path datasets/douban_book_review/test_nolabel.tsv \
                                          --prediction_path datasets/douban_book_review/prediction.tsv \
                                          --labels_num 2 \
                                          --embedding word --remove_embedding_layernorm --encoder lstm --pooling mean
```

The example of using ELMo for classification:
```
python3 finetune/run_classifier.py --pretrained_model_path models/chnsenticorp_elmo_model.bin \
                                   --vocab_path models/google_zh_vocab.txt --config_path models/birnn_config.json \
                                   --train_path datasets/douban_book_review/train.tsv \
                                   --dev_path datasets/douban_book_review/dev.tsv \
                                   --test_path datasets/douban_book_review/test.tsv \
                                   --epochs_num 5  --batch_size 64 --seq_length 192 --learning_rate 5e-4 \
                                   --embedding word --remove_embedding_layernorm --encoder bilstm --pooling max

python3 inference/run_classifier_infer.py --load_model_path models/finetuned_model.bin \
                                          --vocab_path models/google_zh_vocab.txt \
                                          --config_path models/birnn_config.json \
                                          --test_path datasets/douban_book_review/test_nolabel.tsv \
                                          --prediction_path datasets/douban_book_review/prediction.tsv \
                                          --seq_length 192 --labels_num 2 \
                                          --embedding word --remove_embedding_layernorm --encoder bilstm --pooling max
```

The example of using GatedCNN for classification:
```
python3 finetune/run_classifier.py --pretrained_model_path models/cluecorpussmall_gatedcnn_lm_model.bin \
                                   --vocab_path models/google_zh_vocab.txt \
                                   --config_path models/gatedcnn_9_config.json \
                                   --train_path datasets/douban_book_review/train.tsv \
                                   --dev_path datasets/douban_book_review/dev.tsv \
                                   --test_path datasets/douban_book_review/test.tsv \
                                   --epochs_num 5  --batch_size 64 --learning_rate 5e-5 \
                                   --embedding word --remove_embedding_layernorm --encoder gatedcnn --pooling mean

python3 inference/run_classifier_infer.py --load_model_path models/finetuned_model.bin \
                                          --vocab_path models/google_zh_vocab.txt \
                                          --config_path models/gatedcnn_9_config.json \
                                          --test_path datasets/douban_book_review/test_nolabel.tsv \
                                          --prediction_path datasets/douban_book_review/prediction.tsv \
                                          --labels_num 2 \
                                          --embedding word --remove_embedding_layernorm --encoder gatedcnn --pooling mean
```

UER-py supports multi-task learning. Embedding and encoder layers are shared by different tasks. <br>
The example of training two sentiment analysis datasets:
```
python3 finetune/run_classifier_mt.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                                      --dataset_path_list datasets/douban_book_review/ datasets/chnsenticorp/ \
                                      --epochs_num 1 --batch_size 64 \
                                      --embedding word_pos_seg --encoder transformer --mask fully_visible
```
*--dataset_path_list* specifies folder path list of different tasks. Each folder should contains train set *train.tsv* and development set *dev.tsv* .

UER-py supports grid search for classification task:
```
python3 finetune/run_classifier_grid.py --vocab_path models/google_zh_vocab.txt \
                                        --config_path models/bert/tiny_config.json \
                                        --train_path datasets/douban_book_review/train.tsv \
                                        --dev_path datasets/douban_book_review/dev.tsv \
                                        --embedding word_pos_seg --encoder transformer --mask fully_visible \
                                        --batch_size_list 32 64 --learning_rate_list 3e-5 1e-4 3e-4 --epochs_num_list 3 5 8
```
We use grid search to find optimal batch size, learning rate, and the number of epochs.

UER-py supports distillation for classification tasks. <br>
First of all, we train a teacher model. We fine-tune upon a Chinese BERT-large model (provided in model zoo):
```
python3 finetune/run_classifier.py --pretrained_model_path models/mixed_corpus_bert_large_model.bin \
                                   --vocab_path models/google_zh_vocab.txt \
                                   --config_path models/bert_large_config.json \
                                   --output_model_path models/teacher_classifier_model.bin \
                                   --train_path datasets/douban_book_review/train.tsv \
                                   --dev_path datasets/douban_book_review/dev.tsv \
                                   --test_path datasets/douban_book_review/test.tsv \
                                   --epochs_num 3 --batch_size 32 \
                                   --embedding word_pos_seg --encoder transformer --mask fully_visible
```
Then we use the teacher model to do inference. The pesudo labels and logits are generated:
```
python3 inference/run_classifier_infer.py --load_model_path models/teacher_classifier_model.bin \
                                          --vocab_path models/google_zh_vocab.txt \
                                          --config_path models/bert_large_config.json --test_path text.tsv \
                                          --prediction_path label_logits.tsv --labels_num 2 --output_logits \
                                          --embedding word_pos_seg --encoder transformer --mask fully_visible
```
The input file *text.tsv* contains text to be predicted (see *datasets/douban_book_review/test_nolabel.tsv*). *text.tsv* could be downstream dataset, e.g. using *datasets/douban_book_review/train.tsv* as input (*--test_path*), or related external data. Larger transfer set often leads to better performance. <br>
The output file *label_logits.tsv* contains label column and logits column. Then we obtain *text_label_logits.tsv* by combining *text.tsv* and *label_logits.tsv* . *text_label_logits.tsv* contains text_a column (text_a column and text_b column for pair classification), label column (hard label), and logits column (soft label). <br>
Student model is a 3-layers BERT-tiny model. The pre-trained model is provided in model zoo.
Then the student model learns the outputs (hard and soft labels) of the teacher model:
```
python3 finetune/run_classifier.py --pretrained_model_path mixed_corpus_bert_tiny_model.bin \
                                   --vocab_path models/google_zh_vocab.txt \
                                   --config_path models/bert_tiny_config.json \
                                   --train_path text_label_logits.tsv \
                                   --dev_path datasets/douban_book_review/dev.tsv \
                                   --test_path datasets/douban_book_review/test.tsv \
                                   --epochs_num 3 --batch_size 64 --soft_targets --soft_alpha 0.5 \
                                   --embedding word_pos_seg --encoder transformer --mask fully_visible
```
*--soft_targets* denotes that the model uses logits (soft label) for training. Mean-squared-error (MSE) is used as loss function. <br>
*--soft_alpha* specifies the weight of the soft label loss. The loss function is weighted average of cross-entropy loss (for hard label) and mean-squared-error loss (for soft label).

## Document-based question answering
*run_dbqa.py* uses the same network architecture with *run_classifier.py* .
```
usage: run_dbqa.py [-h] [--pretrained_model_path PRETRAINED_MODEL_PATH]
                   [--output_model_path OUTPUT_MODEL_PATH]
                   [--vocab_path VOCAB_PATH] [--spm_model_path SPM_MODEL_PATH]
                   --train_path TRAIN_PATH --dev_path DEV_PATH
                   [--test_path TEST_PATH] [--config_path CONFIG_PATH]
                   [--embedding {word,word_pos,word_pos_seg,word_sinusoidalpos}]
                   [--max_seq_length MAX_SEQ_LENGTH]
                   [--relative_position_embedding]
                   [--relative_attention_buckets_num RELATIVE_ATTENTION_BUCKETS_NUM]
                   [--remove_embedding_layernorm] [--remove_attention_scale]
                   [--encoder {transformer,rnn,lstm,gru,birnn,bilstm,bigru,gatedcnn}]
                   [--mask {fully_visible,causal,causal_with_prefix}]
                   [--layernorm_positioning {pre,post}]
                   [--feed_forward {dense,gated}] [--remove_transformer_bias]
                   [--layernorm {normal,t5}] [--bidirectional]
                   [--factorized_embedding_parameterization]
                   [--parameter_sharing] [--learning_rate LEARNING_RATE]
                   [--warmup WARMUP] [--fp16] [--fp16_opt_level {O0,O1,O2,O3}]
                   [--optimizer {adamw,adafactor}]
                   [--scheduler {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}]
                   [--batch_size BATCH_SIZE] [--seq_length SEQ_LENGTH]
                   [--dropout DROPOUT] [--epochs_num EPOCHS_NUM]
                   [--report_steps REPORT_STEPS] [--seed SEED]
                   [--pooling {mean,max,first,last}]
                   [--tokenizer {bert,char,space}] [--soft_targets]
                   [--soft_alpha SOFT_ALPHA]
```
The document-based question answering (DBQA) can be converted to classification task. Column text_a contains question and column text_b contains sentence which may has answer.
The example of using *run_dbqa.py*:
```
python3 finetune/run_dbqa.py --pretrained_model_path models/google_zh_model.bin \
                             --vocab_path models/google_zh_vocab.txt \
                             --train_path datasets/nlpcc-dbqa/train.tsv \
                             --dev_path datasets/nlpcc-dbqa/dev.tsv \
                             --test datasets/nlpcc-dbqa/test.tsv \
                             --epochs_num 3 --batch_size 64 \
                             --embedding word_pos_seg --encoder transformer --mask fully_visible
```
The example of using *inference/run_classifier_infer.py* to do inference for DBQA:
```
python3 inference/run_classifier_infer.py --load_model_path models/finetuned_model.bin \
                                          --vocab_path models/google_zh_vocab.txt \
                                          --test_path datasets/nlpcc-dbqa/test_nolabel.tsv \
                                          --prediction_path datasets/nlpcc-dbqa/prediction.tsv --labels_num 2 \
                                          --output_logits --output_prob \
                                          --embedding word_pos_seg --encoder transformer --mask fully_visible
```
The example of using ALBERT for DBQA:
```
python3 finetune/run_dbqa.py --pretrained_model_path models/google_zh_albert_base_model.bin \
                             --vocab_path models/google_zh_vocab.txt \
                             --config_path models/albert/base_config.json \
                             --train_path datasets/nlpcc-dbqa/train.tsv \
                             --dev_path datasets/nlpcc-dbqa/dev.tsv \
                             --test datasets/nlpcc-dbqa/test.tsv \
                             --epochs_num 3 --batch_size 64 \
                             --factorized_embedding_parameterization --parameter_sharing \
                             --embedding word_pos_seg --encoder transformer --mask fully_visible
```
The example of doing inference for ALBERT:
```
python3 inference/run_classifier_infer.py --load_model_path models/finetuned_model.bin \
                                          --vocab_path models/google_zh_vocab.txt \
                                          --config_path models/albert/base_config.json \
                                          --test_path datasets/nlpcc-dbqa/test_nolabel.tsv \
                                          --prediction_path datasets/nlpcc-dbqa/prediction.tsv --labels_num 2 \
                                          --factorized_embedding_parameterization --parameter_sharing \
                                          --embedding word_pos_seg --encoder transformer --mask fully_visible
```

## Sequence labeling
*run_ner.py* adds one feedforward layer upon encoder layer.
```
usage: run_ner.py [-h] [--pretrained_model_path PRETRAINED_MODEL_PATH]
                  [--output_model_path OUTPUT_MODEL_PATH]
                  [--vocab_path VOCAB_PATH] [--spm_model_path SPM_MODEL_PATH]
                  --train_path TRAIN_PATH --dev_path DEV_PATH
                  [--test_path TEST_PATH] [--config_path CONFIG_PATH]
                  [--embedding {word,word_pos,word_pos_seg,word_sinusoidalpos}]
                  [--max_seq_length MAX_SEQ_LENGTH]
                  [--relative_position_embedding]
                  [--relative_attention_buckets_num RELATIVE_ATTENTION_BUCKETS_NUM]
                  [--remove_embedding_layernorm] [--remove_attention_scale]
                  [--encoder {transformer,rnn,lstm,gru,birnn,bilstm,bigru,gatedcnn}]
                  [--mask {fully_visible,causal,causal_with_prefix}]
                  [--layernorm_positioning {pre,post}]
                  [--feed_forward {dense,gated}] [--remove_transformer_bias]
                  [--layernorm {normal,t5}] [--bidirectional]
                  [--factorized_embedding_parameterization]
                  [--parameter_sharing] [--learning_rate LEARNING_RATE]
                  [--warmup WARMUP] [--fp16] [--fp16_opt_level {O0,O1,O2,O3}]
                  [--optimizer {adamw,adafactor}]
                  [--scheduler {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}]
                  [--batch_size BATCH_SIZE] [--seq_length SEQ_LENGTH]
                  [--dropout DROPOUT] [--epochs_num EPOCHS_NUM]
                  [--report_steps REPORT_STEPS] [--seed SEED] --label2id_path
                  LABEL2ID_PATH
```
The example of using *run_ner.py*:
```
python3 finetune/run_ner.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                            --train_path datasets/msra_ner/train.tsv \
                            --dev_path datasets/msra_ner/dev.tsv \
                            --test_path datasets/msra_ner/test.tsv \
                            --label2id_path datasets/msra_ner/label2id.json \
                            --epochs_num 5 --batch_size 16 \
                            --embedding word_pos_seg --encoder transformer --mask fully_visible
```
The example of doing inference:
```
python3 inference/run_ner_infer.py --load_model_path models/finetuned_model.bin --vocab_path models/google_zh_vocab.txt \
                                   --test_path datasets/msra_ner/test_nolabel.tsv \
                                   --prediction_path datasets/msra_ner/prediction.tsv \
                                   --label2id_path datasets/msra_ner/label2id.json \
                                   --embedding word_pos_seg --encoder transformer --mask fully_visible
```
The example of using ALBERT for NER:
```
python3 finetune/run_ner.py --pretrained_model_path models/google_zh_albert_base_model.bin \
                            --vocab_path models/google_zh_vocab.txt \
                            --config_path models/albert/base_config.json \
                            --train_path datasets/msra_ner/train.tsv \
                            --dev_path datasets/msra_ner/dev.tsv \
                            --test_path datasets/msra_ner/test.tsv \
                            --label2id_path datasets/msra_ner/label2id.json \
                            --epochs_num 5 --batch_size 16 --learning_rate 1e-4 \
                            --factorized_embedding_parameterization --parameter_sharing \
                            --embedding word_pos_seg --encoder transformer --mask fully_visible

python3 inference/run_ner_infer.py --load_model_path models/finetuned_model.bin \
                                   --vocab_path models/google_zh_vocab.txt \
                                   --config_path models/albert/base_config.json \
                                   --test_path datasets/msra_ner/test_nolabel.tsv \
                                   --prediction_path datasets/msra_ner/prediction.tsv \
                                   --label2id_path datasets/msra_ner/label2id.json \
                                   --factorized_embedding_parameterization --parameter_sharing \
                                   --embedding word_pos_seg --encoder transformer --mask fully_visible
```

The example of using ELMo for NER:
```
python3 finetune/run_ner.py --pretrained_model_path models/cluecorpussmall_elmo_model.bin-500000 \
                            --vocab_path models/google_zh_vocab.txt \
                            --config_path models/birnn_config.json \
                            --train_path datasets/msra_ner/train.tsv \
                            --dev_path datasets/msra_ner/dev.tsv \
                            --test_path datasets/msra_ner/test.tsv \
                            --label2id_path datasets/msra_ner/label2id.json \
                            --epochs_num 5  --batch_size 16  --learning_rate 5e-4 \
                            --embedding word --remove_embedding_layernorm --encoder bilstm

python3 inference/run_ner_infer.py --load_model_path models/finetuned_model.bin \
                                   --vocab_path models/google_zh_vocab.txt \
                                   --config_path models/birnn_config.json \
                                   --test_path datasets/msra_ner/test_nolabel.tsv \
                                   --prediction_path datasets/msra_ner/prediction.tsv \
                                   --label2id_path datasets/msra_ner/label2id.json \
                                   --embedding word --remove_embedding_layernorm --encoder bilstm
```

## Machine reading comprehension
run_cmrc.py adds one feedforward layer upon encoder layer.
```
usage: run_cmrc.py [-h] [--pretrained_model_path PRETRAINED_MODEL_PATH]
                   [--output_model_path OUTPUT_MODEL_PATH]
                   [--vocab_path VOCAB_PATH] [--spm_model_path SPM_MODEL_PATH]
                   --train_path TRAIN_PATH --dev_path DEV_PATH
                   [--test_path TEST_PATH] [--config_path CONFIG_PATH]
                   [--embedding {word,word_pos,word_pos_seg,word_sinusoidalpos}]
                   [--max_seq_length MAX_SEQ_LENGTH]
                   [--relative_position_embedding]
                   [--relative_attention_buckets_num RELATIVE_ATTENTION_BUCKETS_NUM]
                   [--remove_embedding_layernorm] [--remove_attention_scale]
                   [--encoder {transformer,rnn,lstm,gru,birnn,bilstm,bigru,gatedcnn}]
                   [--mask {fully_visible,causal,causal_with_prefix}]
                   [--layernorm_positioning {pre,post}]
                   [--feed_forward {dense,gated}] [--remove_transformer_bias]
                   [--layernorm {normal,t5}] [--bidirectional]
                   [--factorized_embedding_parameterization]
                   [--parameter_sharing] [--learning_rate LEARNING_RATE]
                   [--warmup WARMUP] [--fp16] [--fp16_opt_level {O0,O1,O2,O3}]
                   [--optimizer {adamw,adafactor}]
                   [--scheduler {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}]
                   [--batch_size BATCH_SIZE] [--seq_length SEQ_LENGTH]
                   [--dropout DROPOUT] [--epochs_num EPOCHS_NUM]
                   [--report_steps REPORT_STEPS] [--seed SEED]
                   [--doc_stride DOC_STRIDE]
```
The example of using *run_cmrc.py* for Chinese Machine Reading Comprehension (CMRC):
```
python3 finetune/run_cmrc.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                             --train_path datasets/cmrc2018/train.json --dev_path datasets/cmrc2018/dev.json \
                             --epochs_num 2 --batch_size 8 --seq_length 512 \
                             --embedding word_pos_seg --encoder transformer --mask fully_visible
```
The *train.json* and *dev.json* are of squad-style. Train set and development set are available [here](https://github.com/ymcui/cmrc2018). *--test_path* option is not specified since test set is not publicly available.

The example of doing inference:
```
python3  inference/run_cmrc_infer.py --load_model_path models/finetuned_model.bin \
                                     --vocab_path models/google_zh_vocab.txt \
                                     --test_path datasets/cmrc2018/test.json \
                                     --prediction_path datasets/cmrc2018/prediction.json --seq_length 512 \
                                     --embedding word_pos_seg --encoder transformer --mask fully_visible
```
The example of using ALBERT-xxlarge for CMRC:
```
python3 finetune/run_cmrc.py --pretrained_model_path models/google_zh_albert_xxlarge_model.bin \
                             --vocab_path models/google_zh_vocab.txt \
                             --config_path models/albert/xxlarge_config.json \
                             --train_path datasets/cmrc2018/train.json --dev_path datasets/cmrc2018/dev.json \
                             --epochs_num 2 --batch_size 8 --seq_length 512 --learning_rate 1e-5 \
                             --factorized_embedding_parameterization --parameter_sharing \
                             --embedding word_pos_seg --encoder transformer --mask fully_visible
```
The example of doing inference for ALBERT:
```
python3 inference/run_cmrc_infer.py --load_model_path models/finetuned_model.bin \
                                     --vocab_path models/google_zh_vocab.txt \
                                     --config_path models/albert/xxlarge_config.json \
                                     --test_path datasets/cmrc2018/test.json \
                                     --prediction_path datasets/cmrc2018/prediction.json --seq_length 512 \
                                     --factorized_embedding_parameterization --parameter_sharing \
                                     --embedding word_pos_seg --encoder transformer --mask fully_visible
```

## Multiple choice
C3 is a multiple choice dataset. Given context and question, one need to select one answer from four candidate answers. run_c3.py adds one feedforward layer upon encoder layer.
```
usage: run_c3.py [-h] [--pretrained_model_path PRETRAINED_MODEL_PATH]
                 [--output_model_path OUTPUT_MODEL_PATH]
                 [--vocab_path VOCAB_PATH] [--spm_model_path SPM_MODEL_PATH]
                 --train_path TRAIN_PATH --dev_path DEV_PATH
                 [--test_path TEST_PATH] [--config_path CONFIG_PATH]
                 [--embedding {word,word_pos,word_pos_seg,word_sinusoidalpos}]
                 [--max_seq_length MAX_SEQ_LENGTH]
                 [--relative_position_embedding]
                 [--relative_attention_buckets_num RELATIVE_ATTENTION_BUCKETS_NUM]
                 [--remove_embedding_layernorm] [--remove_attention_scale]
                 [--encoder {transformer,rnn,lstm,gru,birnn,bilstm,bigru,gatedcnn}]
                 [--mask {fully_visible,causal,causal_with_prefix}]
                 [--layernorm_positioning {pre,post}]
                 [--feed_forward {dense,gated}] [--remove_transformer_bias]
                 [--layernorm {normal,t5}] [--bidirectional]
                 [--factorized_embedding_parameterization]
                 [--parameter_sharing] [--learning_rate LEARNING_RATE]
                 [--warmup WARMUP] [--fp16] [--fp16_opt_level {O0,O1,O2,O3}]
                 [--optimizer {adamw,adafactor}]
                 [--scheduler {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}]
                 [--batch_size BATCH_SIZE] [--seq_length SEQ_LENGTH]
                 [--dropout DROPOUT] [--epochs_num EPOCHS_NUM]
                 [--report_steps REPORT_STEPS] [--seed SEED]
                 [--max_choices_num MAX_CHOICES_NUM]
                 [--tokenizer {bert,char,space}]
```
The example of using *run_c3.py* for multiple choice task:
```
python3 finetune/run_c3.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                           --train_path datasets/c3/train.json --dev_path datasets/c3/dev.json \
                           --epochs_num 8 --batch_size 8 --seq_length 512 --max_choices_num 4 \
                           --embedding word_pos_seg --encoder transformer --mask fully_visible
```
*--test_path* option is not specified since test set of C3 dataset is not publicly available. <br>
The actual batch size is *--batch_size* times *--max_choices_num* . <br>
The question in C3 dataset contains at most 4 candidate answers. *--max_choices_num* is set to 4.

The example of doing inference:
```
python3 inference/run_c3_infer.py --load_model_path models/finetuned_model.bin \
                                  --vocab_path models/google_zh_vocab.txt \
                                  --test_path datasets/c3/test.json \
                                  --prediction_path datasets/c3/prediction.json \
                                  --max_choices_num 4 --seq_length 512 \
                                  --embedding word_pos_seg --encoder transformer --mask fully_visible
```
The example of using ALBERT-xlarge for C3:
```
python3 finetune/run_c3.py --pretrained_model_path models/google_zh_albert_xlarge_model.bin \
                           --vocab_path models/google_zh_vocab.txt \
                           --config_path models/albert/xlarge_config.json \
                           --train_path datasets/c3/train.json --dev_path datasets/c3/dev.json \
                           --epochs_num 8 --batch_size 8 --seq_length 512 --max_choices_num 4 \
                           --factorized_embedding_parameterization --parameter_sharing \
                           --embedding word_pos_seg --encoder transformer --mask fully_visible
```

The example of doing inference for ALBERT-large:
```
python3  inference/run_c3_infer.py --load_model_path models/finetuned_model.bin \
                                   --vocab_path models/google_zh_vocab.txt \
                                   --config_path models/albert/xlarge_config.json \
                                   --test_path datasets/c3/test.json \
                                   --prediction_path datasets/c3/prediction.json \
                                   --max_choices_num 4 --seq_length 512 \
                                   --factorized_embedding_parameterization --parameter_sharing \
                                   --embedding word_pos_seg --encoder transformer --mask fully_visible
```
