UER-py allows users to combine different components (e.g. embeddings, encoders, and targets). Here are some examples of trying different combinations.

#### RoBERTa
The example of pre-processing and pre-training for RoBERTa:
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_zh_vocab.txt \
                      --dataset_path dataset.pt --processes_num 8 \
                      --dynamic_masking --target mlm
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --output_model_path models/output_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --learning_rate 1e-4 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target mlm
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
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --learning_rate 2e-5 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target mlm
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
                    --factorized_embedding_parameterization --parameter_sharing \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target albert
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
                    --factorized_embedding_parameterization --parameter_sharing \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target albert
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --pretrained_model_path models/google_zh_albert_xxlarge_model.bin \
                    --output_model_path models/output_model.bin \
                    --config_path models/albert_xxlarge_config.json \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --learning_rate 2e-5 \
                    --factorized_embedding_parameterization --parameter_sharing \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target albert
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
                    --total_steps 10000 --save_checkpoint 5000 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target mlm
```

#### GPT
The example of pre-processing and pre-training for GPT:
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_zh_vocab.txt \
                      --dataset_path dataset.pt --processes_num 8 --target lm
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --output_model_path models/output_model.bin \
                    --config_path models/bert_base_config.json --learning_rate 1e-4 \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --embedding word_pos --encoder transformer --mask causal --target lm
```
The corpus format of GPT is the identical with RoBERTa. We can pre-training GPT through *--embedding word_pos --encoder transformer --mask causal --target lm* .
GPT can use the configuration file of BERT.

#### GPT-2
The example of pre-processing and pre-training for GPT-2:
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_zh_vocab.txt \
                      --dataset_path dataset.pt --processes_num 8 --target lm
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --output_model_path models/output_model.bin \
                    --config_path models/bert_base_config.json --learning_rate 1e-4 \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --tie_weight \
                    --embedding word_pos --remove_embedding_layernorm \
                    --encoder transformer --mask causal --layernorm_positioning pre --target lm
```
The corpus format of GPT-2 is the identical with GPT and RoBERTa. Notice that the encoder of GPT-2 is different from the encoder of GPT. The layer normalization is moved to the input of each sub-block (*--layernorm_positioning pre*) and an additional layer normalization is added after the final block. The layer normalization after embedding layer should be removed (*--remove_embedding_layernorm*).

#### ELMo
The example of pre-processing and pre-training for ELMo:
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_zh_vocab.txt \
                      --dataset_path dataset.pt --processes_num 8 --target bilm
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --output_model_path models/output_model.bin \
                    --config_path models/birnn_config.json --learning_rate 5e-4 \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --embedding word --encoder bilstm --target bilm
```
The corpus format of ELMo is identical with GPT. We can pre-training ELMo through *--embedding word*, *--encoder bilstm*, and *--target bilm*. <br>
*--embedding word* denotes using traditional word embedding. LSTM does not require position embedding. In addition, layernorm is not commonly used in traditional RNN related models. So we can use *--remove_embedding_layernorm* . Nevertheless, it doesn't matter if layernorm is added.

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