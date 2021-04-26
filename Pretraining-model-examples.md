UER-py allows users to combine different components (e.g. embeddings, encoders, and targets). Here are some examples of trying different combinations to implement frequently-used pre-training models.

### RoBERTa
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

### ALBERT
The example of pre-processing and pre-training for ALBERT:
```
python3 preprocess.py --corpus_path corpora/book_review_bert.txt --vocab_path models/google_zh_vocab.txt \
                      --dataset_path dataset.pt --processes_num 8 --target albert

python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --output_model_path models/output_model.bin \
                    --config_path models/albert/base_config.json \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --learning_rate 1e-4 \
                    --factorized_embedding_parameterization --parameter_sharing \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target albert
```
The corpus format of ALBERT is the identical with BERT. <br>
*--target albert* denotes that using ALBERT target, which consists of mlm and sop targets. <br>
*--factorized_embedding_parameterization* denotes that using factorized embedding parameterization to untie the embedding size from the hidden layer size. <br>
*--parameter_sharing* denotes that sharing all parameters (including feed-forward and attention parameters) across layers. <br>
we provide 4 configuration files for ALBERT model in *models/albert* folder, *albert_base_config.json* , *albert_large_config.json* , *albert_xlarge_config.json* , *albert_xxlarge_config.json* . <br>
The example of doing incremental pre-training upon Google's ALBERT pre-trained models of different sizes (See model zoo for pre-trained weights):
```
python3 preprocess.py --corpus_path corpora/book_review_bert.txt --vocab_path models/google_zh_vocab.txt \
                      --dataset_path dataset.pt --processes_num 8 --target albert 

python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --pretrained_model_path models/google_zh_albert_base_model.bin \
                    --output_model_path models/output_model.bin \
                    --config_path models/albert/base_config.json \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --learning_rate 2e-5 \
                    --factorized_embedding_parameterization --parameter_sharing \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target albert

python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --pretrained_model_path models/google_zh_albert_xxlarge_model.bin \
                    --output_model_path models/output_model.bin \
                    --config_path models/albert/xxlarge_config.json \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --learning_rate 2e-5 \
                    --factorized_embedding_parameterization --parameter_sharing \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target albert
```

### SpanBERT
SpanBERT introduces span masking and span boundary objective. We only consider span masking here. NSP target is removed by SpanBERT. <br>
The example of pre-processing and pre-training for SpanBERT (static masking):
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_zh_vocab.txt \
                      --dataset_path dataset.pt --processes_num 8 --dup_factor 20 \
                      --span_masking --span_geo_prob 0.3 --span_max_length 5 --target mlm

python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --output_model_path models/output_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7  --learning_rate 1e-4 \
                    --total_steps 10000 --save_checkpoint 5000 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target mlm
```
*--dup_factor* specifies the number of times to duplicate the input data (with different masks). The default value is 5 . <br>
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

### BERT-WWM
BERT-WWM introduces whole word masking. MLM target is used here. <br>
The example of pre-processing and pre-training for BERT-WWM (static masking):
```
python3 preprocess.py --corpus_path corpora/book_review.txt \
                      --vocab_path models/google_zh_vocab.txt \
                      --dataset_path dataset.pt \
                      --processes_num 8 --dup_factor 20 \
                      --whole_word_masking \
                      --target mlm

python3 pretrain.py --dataset_path dataset.pt \
                    --vocab_path models/google_zh_vocab.txt \
                    --output_model_path models/output_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7  --learning_rate 1e-4 \
                    --total_steps 10000 --save_checkpoint 5000 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target mlm
```
*--whole_word_masking* denotes that whole word masking is used. <br>
The example of pre-processing and pre-training for BERT-WMM (dynamic masking):
```
python3 preprocess.py --corpus_path corpora/book_review.txt \
                      --vocab_path models/google_zh_vocab.txt \
                      --dataset_path dataset.pt \
                      --processes_num 8 --dynamic_masking \
                      --target mlm

python3 pretrain.py --dataset_path dataset.pt \
                    --vocab_path models/google_zh_vocab.txt \
                    --output_model_path models/output_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7  --learning_rate 1e-4 \
                    --whole_word_masking \
                    --total_steps 10000 --save_checkpoint 5000 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target mlm
```
BERT-WMM implemented in UER is only applicable to Chinese. [jieba](https://github.com/fxsjy/jieba) is used as word segmentation tool (see *uer/utils/data.py*):
```
import jieba
wordlist = jieba.cut(sentence)
```
One can change the code in *uer/utils/data.py* to substitute jieba for other word segmentation tools.

### GPT
The example of pre-processing and pre-training for GPT:
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_zh_vocab.txt \
                      --dataset_path dataset.pt --processes_num 8 --target lm

python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --output_model_path models/output_model.bin \
                    --config_path models/gpt2/config.json --learning_rate 1e-4 \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --embedding word_pos --encoder transformer --mask causal --target lm
```
The corpus format of GPT is the identical with RoBERTa. We can pre-train GPT through *--embedding word_pos --encoder transformer --mask causal --target lm* .
GPT can use the configuration file of BERT.

### GPT-2
The example of pre-processing and pre-training for GPT-2:
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_zh_vocab.txt \
                      --dataset_path dataset.pt --processes_num 8 --target lm

python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --output_model_path models/output_model.bin \
                    --config_path models/gpt2/config.json --learning_rate 1e-4 \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --tie_weight \
                    --embedding word_pos --remove_embedding_layernorm \
                    --encoder transformer --mask causal --layernorm_positioning pre --target lm
```
The corpus format of GPT-2 is the identical with GPT and RoBERTa. Notice that the encoder of GPT-2 is different from the encoder of GPT. The layer normalization is moved to the input of each sub-block (*--layernorm_positioning pre*) and an additional layer normalization is added after the final block. The layer normalization after embedding layer should be removed (*--remove_embedding_layernorm*).

### ELMo
The example of pre-processing and pre-training for ELMo:
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_zh_vocab.txt \
                      --dataset_path dataset.pt --processes_num 8 --target bilm

python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --output_model_path models/output_model.bin \
                    --config_path models/birnn_config.json --learning_rate 5e-4 \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --embedding word --encoder bilstm --target bilm
```
The corpus format of ELMo is identical with GPT-2. We can pre-train ELMo through *--embedding word*, *--encoder bilstm*, and *--target bilm*. <br>
*--embedding word* denotes using traditional word embedding. LSTM does not require position embedding. In addition, we specify *--remove_embedding_layernorm* and the layernorm after word embedding is removed.

### T5
T5 proposes to use seq2seq model to unify NLU and NLG tasks. With extensive experiments, T5 recommend to use encoder-decoder architecture and BERT-style objective function (the model predicts the masked words). The example of using T5 for pre-training:
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_zh_vocab.txt \
                      --dataset_path dataset.pt --processes_num 8 --seq_length 128 --dynamic_masking --target t5

python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --config_path models/t5/small_config.json \
                    --output_model_path models/output_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --learning_rate 1e-3 --batch_size 64 \
                    --span_masking --span_geo_prob 0.3 --span_max_length 5 \
                    --embedding word --relative_position_embedding --remove_embedding_layernorm --tgt_embedding word \
                    --encoder transformer --mask fully_visible --layernorm_positioning pre --decoder transformer \
                    --target t5 --tie_weights
```
The corpus format of T5 is identical with GPT-2. *--relative_position_embedding* denotes using relative position embedding. *--remove_embedding_layernorm* and *--layernorm_positioning pre* denote that pre-layernorm is used (same with GPT-2). Since T5 uses encoder-decoder architecture, we have to specify *--encoder* and *--decoder*.

### T5-v1_1
T5-v1_1 includes several improvements compared to the original T5 model. The example of using T5-v1_1 for pre-training:
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_zh_vocab.txt \
                      --dataset_path dataset.pt --processes_num 8 --seq_length 128 --dynamic_masking --target t5

python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --config_path models/t5-v1_1/small_config.json \
                    --output_model_path models/output_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --learning_rate 1e-3 --batch_size 64 \
                    --span_masking --span_geo_prob 0.3 --span_max_length 5 \
                    --embedding word --relative_position_embedding --remove_embedding_layernorm --tgt_embedding word \
                    --encoder transformer --mask fully_visible --layernorm_positioning pre --feed_forward gated --decoder transformer \
                    --target t5
```
The corpus format of T5-v1_1 is identical with T5. *--feed_forward* denotes the type of feed-forward layer. *--tie_weights* is removed and there is no parameter sharing between embedding and classifier layer. T5-v1_1 and T5 have different configuration files.

### More combinations
The example of using LSTM encoder and LM target for pre-training:
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_zh_vocab.txt --dataset_path dataset.pt --processes_num 8 --target lm

python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --output_model_path models/output_model.bin \
                    --config_path models/rnn_config.json --learning_rate 1e-3 \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --total_steps 20000 --save_checkpoint_steps 5000 \
                    --embedding word --remove_embedding_layernorm --encoder lstm --target lm
```
We use the *models/rnn_config.json* as configuration file.

The example of using GRU encoder and LM target for pre-training:
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_zh_vocab.txt --dataset_path dataset.pt --processes_num 8 --target lm

python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --output_model_path models/output_model.bin \
                    --config_path models/rnn_config.json --learning_rate 1e-3 \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --total_steps 20000 --save_checkpoint_steps 5000 \
                    --embedding word --remove_embedding_layernorm --encoder gru --target lm
```

The example of using GatedCNN encoder and LM target for pre-training:
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_zh_vocab.txt --dataset_path dataset.pt --processes_num 8 --target lm

python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --output_model_path models/output_model.bin \
                    --config_path models/gatedcnn_9_config.json --learning_rate 1e-4 \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --total_steps 20000 --save_checkpoint_steps 5000 \
                    --embedding word --remove_embedding_layernorm --encoder gatedcnn --target lm
```

The example of using machine translation for pre-training (the objective is the same with CoVe but the Transformer encoder and decoder are used):
```
python3 preprocess.py --corpus_path corpora/iwslt_15_zh_en.tsv \
                      --vocab_path models/google_zh_vocab.txt --tgt_vocab_path models/google_uncased_en_vocab.txt \
                      --dataset_path dataset.pt --seq_length 64 --tgt_seq_length 64 --processes_num 8 --target seq2seq

python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --tgt_vocab_path models/google_uncased_en_vocab.txt \
                    --output_model_path output_model.bin --config_path models/encoder_decoder_config.json \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --learning_rate 1e-4 \
                    --report_steps 1000 --total_steps 50000 --save_checkpoint_steps 10000 \
                    --embedding word_sinusoidalpos --tgt_embedding word_sinusoidalpos \
                    --encoder transformer --mask fully_visible --decoder transformer \
                    --target seq2seq
```
[*iwslt_15_zh_en.tsv*](https://share.weiyun.com/hIKCDpf6) is a Chinese-English parallel corpus. The source and target sequences are separated by \t , which is the corpus format of *--target seq2seq* . The pre-trained encoder can be used for downstream tasks.

The example of using prefix LM for pre-training (which is used in UniLM):
```
python3 preprocess.py --corpus_path corpora/csl_title_abstract.txt --vocab_path models/google_zh_vocab.txt \
                      --dataset_path dataset.pt --seq_length 256 --processes_num 8 --target prefixlm

python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --output_model_path output_model.bin --config_path models/bert/base_config.json \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --learning_rate 1e-4 \
                    --total_steps 5000 --save_checkpoint_steps 100 \
                    --embedding word_pos_seg --encoder transformer --mask causal_with_prefix --target prefixlm
```
[*csl_title_abstract.txt*](https://share.weiyun.com/LwuQwWVl) is a Chinese scientific literature corpus. The title and abstract sequences are separated by \t , which is the corpus format of *--target prefixlm* . We can pre-train prefix LM model through *--mask causal_with_prefix* and *--target prefixlm*. Notice that the model use the segment information to determine which part is prefix. Therefore we have to use *--embedding word_pos_seg*.

The example of using Transformer encoder and classification (CLS) target for pre-training:
```
python3 preprocess.py --corpus_path corpora/book_review_cls.txt --vocab_path models/google_zh_vocab.txt --dataset_path dataset.pt --processes_num 8 --target cls

python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --output_model_path models/output_model.bin \
                    --config_path models/bert/base_config.json \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --total_steps 2000 --save_checkpoint_steps 1000 --learning_rate 2e-5 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --pooling first --target cls --labels_num 2
```
Notice that we need to explicitly specify the number of labels by *--labels_num*. The format of the corpus for classification target is as follows (text and text pair classification):
```
1        instance1
0        instance2
1        instance3
```
```
1        instance1_text_a        instance1_text_b
0        instance2_text_a        instance1_text_b
1        instance3_text_a        instance1_text_b
```
\t is used to separate different columns (see *book_review_cls.txt* in *corpora* folder).

The example of using LSTM encoder and classification (CLS) target for pre-training:
```
python3 preprocess.py --corpus_path corpora/book_review_cls.txt --vocab_path models/google_zh_vocab.txt --dataset_path dataset.pt --processes_num 8 --target cls

python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --output_model_path models/output_model.bin \
                    --config_path models/rnn_config.json \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --total_steps 2000 --save_checkpoint_steps 1000 --learning_rate 1e-3 \
                    --embedding word --remove_embedding_layernorm --encoder lstm --pooling max --target cls --labels_num 2
```
<br>