
With the help of UER, we pre-trained models with different corpora, encoders, and targets. All pre-trained weights introduced in this section are in UER format and can be loaded by UER directly. More pre-trained weights will be released in the near future. Unless otherwise noted, Chinese pre-trained models use BERT tokenizer and *models/google_zh_vocab.txt* as vocabulary (which is used in original BERT project). *models/bert/base_config.json* is used as configuration file in default. Commonly-used vocabulary and configuration files are included in *models/* folder and users do not need to download them. In addition, We use *scripts/convert_xxx_from_uer_to_huggingface.py* to convert pre-trained weights into format that Huggingface Transformers supports, and upload them to [Huggingface model hub (uer)](https://huggingface.co/uer). In the rest of the section, we provide download links of pre-trained weights and the right ways of using them. Notice that, for space constraint, more details of a pre-trained weight are discussed in corresponding Huggingface model hub. We will provide the link of Huggingface model hub when we introduce the pre-trained weight. 

## Chinese RoBERTa Pre-trained Weights
This is the set of 24 Chinese RoBERTa weights. CLUECorpusSmall is used as training corpus. Configuration files are in *models/bert/* folder. We only provide configuration files for Tiny，Mini，Small，Medium，Base，and Large models. To load other models, we need to modify *emb_size*，*feedforward_size*，*hidden_size*，*heads_num*，*layers_num* in the configuration file. Notice that *emb_size* = *emb_size*, *feedforward_size* = 4 * *hidden_size*, *heads_num* = *hidden_size* / 64 . More details of these pre-trained weights are discussed [here](https://huggingface.co/uer/chinese_roberta_L-2_H-128).

The pre-trained Chinese weight links of different layers (L) and hidden sizes (H):

|          |           H=128           |           H=256           |            H=512            |            H=768            |
| -------- | :-----------------------: | :-----------------------: | :-------------------------: | :-------------------------: |
| **L=2**  | [**2/128 (Tiny)**][2_128] |      [2/256][2_256]       |       [2/512][2_512]        |       [2/768][2_768]        |
| **L=4**  |      [4/128][4_128]       | [**4/256 (Mini)**][4_256] | [**4/512 (Small)**][4_512]  |       [4/768][4_768]        |
| **L=6**  |      [6/128][6_128]       |      [6/256][6_256]       |       [6/512][6_512]        |       [6/768][6_768]        |
| **L=8**  |      [8/128][8_128]       |      [8/256][8_256]       | [**8/512 (Medium)**][8_512] |       [8/768][8_768]        |
| **L=10** |     [10/128][10_128]      |     [10/256][10_256]      |      [10/512][10_512]       |      [10/768][10_768]       |
| **L=12** |     [12/128][12_128]      |     [12/256][12_256]      |      [12/512][12_512]       | [**12/768 (Base)**][12_768] |

Take the Tiny weight as an example, we download the Tiny weight through the above link and put it in *models/* folder. We can either conduct further pre-training upon it:
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_zh_vocab.txt --dataset_path dataset.pt \
                      --processes_num 8 --target mlm

python3 pretrain.py --dataset_path dataset.pt --pretrained_model_path models/cluecorpussmall_roberta_tiny_seq512_model.bin \
                    --vocab_path models/google_zh_vocab.txt --config_path models/bert/tiny_config.json \
                    --output_model_path models/output_model.bin --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 5000 --save_checkpoint_steps 2500 --batch_size 64 --embedding word_pos_seg --encoder transformer --mask fully_visible --target mlm
```

or use it on downstream classification dataset：
```
python3 run_classifier.py --pretrained_model_path models/cluecorpussmall_roberta_tiny_seq512_model.bin \
                          --vocab_path models/google_zh_vocab.txt --config_path models/bert/tiny_config.json \
                          --train_path datasets/douban_book_review/train.tsv --dev_path datasets/douban_book_review/dev.tsv --test_path datasets/douban_book_review/test.tsv \
                          --learning_rate 3e-4 --batch_size 64 --epochs_num 8 --embedding word_pos_seg --encoder transformer --mask fully_visible
```

In fine-tuning stage, pre-trained models of different sizes usually require different hyper-parameters. The example of using grid search to find best hyper-parameters:
```
python3 run_classifier_grid.py --pretrained_model_path models/cluecorpussmall_roberta_tiny_seq512_model.bin \
                               --vocab_path models/google_zh_vocab.txt --config_path models/bert/tiny_config.json \
                               --train_path datasets/douban_book_review/train.tsv --dev_path datasets/douban_book_review/dev.tsv \
                               --learning_rate_list 3e-5 1e-4 3e-4 --batch_size_list 32 64 --epochs_num_list 3 5 8 \
                               --embedding word_pos_seg --encoder transformer --mask fully_visible
```
We can reproduce the experimental results reported [here](https://huggingface.co/uer/chinese_roberta_L-2_H-128) through above grid search script.

## Chinese word-based RoBERTa Pre-trained Weights
This is the set of 5 Chinese word-based RoBERTa weights. CLUECorpusSmall is used as training corpus. Configuration files are in *models/bert/* folder. Google sentencepiece is used as tokenizer tool and *models/cluecorpussmall_spm.model* is used as sentencepiece model. Most Chinese pre-trained weights are based on Chinese character. Compared with character-based models, word-based models are faster (because of shorter sequence length) and have better performance according to our experimental results. More details of these pre-trained weights are discussed [here](https://huggingface.co/uer/roberta-tiny-word-chinese-cluecorpussmall)

The pre-trained Chinese weight links of different sizes:

|           Link           |
| :-----------------------:|
| [**L=2/H=128 (Tiny)**][word_tiny] |
| [**L=4/H=256 (Mini)**][word_mini] |
| [**L=4/H=512 (Small)**][word_small] |
| [**L=8/H=512 (Medium)**][word_medium] |
| [**L=12/H=768 (Base)**][word_base] |

Take the word-based Tiny weight as an example, we download the word-based Tiny weight through the above link and put it in *models/* folder. We can either conduct further pre-training upon it:
```
python3 preprocess.py --corpus_path corpora/book_review.txt --spm_model_path models/cluecorpussmall_spm.model --dataset_path dataset.pt \
                      --processes_num 8 --target mlm

python3 pretrain.py --dataset_path dataset.pt --pretrained_model_path models/cluecorpussmall_roberta_tiny_seq512_model.bin \
                    --spm_model_path models/cluecorpussmall_spm.model --config_path models/bert/tiny_config.json \
                    --output_model_path models/output_model.bin --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 5000 --save_checkpoint_steps 2500 --batch_size 64 --embedding word_pos_seg --encoder transformer --mask fully_visible --target mlm
```

or use it on downstream classification dataset：
```
python3 run_classifier.py --pretrained_model_path models/cluecorpussmall_word_roberta_tiny_seq512_model.bin \
                          --spm_model_path models/cluecorpussmall_spm.model --config_path models/bert/tiny_config.json \
                          --train_path datasets/douban_book_review/train.tsv --dev_path datasets/douban_book_review/dev.tsv --test_path datasets/douban_book_review/test.tsv \
                          --learning_rate 3e-4 --batch_size 64 --epochs_num 8 --embedding word_pos_seg --encoder transformer --mask fully_visible
```

The example of using grid search to find best hyper-parameters for word-based model:
```
python3 run_classifier_grid.py --pretrained_model_path models/cluecorpussmall_word_roberta_tiny_seq512_model.bin \
                               --spm_model_path models/cluecorpussmall_spm.model --config_path models/bert/tiny_config.json \
                               --train_path datasets/douban_book_review/train.tsv --dev_path datasets/douban_book_review/dev.tsv \
                               --learning_rate_list 3e-5 1e-4 3e-4 --batch_size_list 32 64 --epochs_num_list 3 5 8 \
                               --embedding word_pos_seg --encoder transformer --mask fully_visible
```
We can reproduce the experimental results reported [here](https://huggingface.co/uer/roberta-tiny-word-chinese-cluecorpussmall) through above grid search script.

## Chinese GPT-2 Pre-trained Weights
This is the set of Chinese GPT-2 pre-trained weights. Configuration files are in *models/gpt2/* folder.

The link and detailed description (Huggingface model hub) of different pre-trained GPT-2 weights:

|           Model link           |           Description link          |
| :-----------------------:| :-----------------------:|
| [**CLUECorpusSmall GPT-2**][gpt2_cluecorpussmall] | https://huggingface.co/uer/gpt2-chinese-cluecorpussmall |
| [**CLUECorpusSmall GPT-2-distil**][gpt2_distil_cluecorpussmall] | https://huggingface.co/uer/gpt2-distil-chinese-cluecorpussmall |
| [**Poem GPT-2**][gpt2_poem] | https://huggingface.co/uer/gpt2-chinese-poem |
| [**Couplet GPT-2**][gpt2_couplet] | https://huggingface.co/uer/gpt2-chinese-couplet |
| [**Lyric GPT-2**][gpt2_lyric] | https://huggingface.co/uer/gpt2-chinese-lyric |
| [**Ancient GPT-2**][gpt2_ancient] | https://huggingface.co/uer/gpt2-chinese-ancient |

Notice that extended vocabularies (*models/google_zh_poem_vocab.txt* and *models/google_zh_ancient_vocab.txt*) are used in Poem and Ancient GPT-2 models. CLUECorpusSmall GPT-2-distil model uses *models/gpt2/distil_config.json* configuration file. *models/gpt2/config.json* are used for other weights.

Take the CLUECorpusSmall GPT-2-distil weight as an example, we download the CLUECorpusSmall GPT-2-distil weight through the above link and put it in *models/* folder. We can either conduct further pre-training upon it:
```
python3 preprocess.py --corpus_path corpora/book_review.txt \
                      --vocab_path models/google_zh_vocab.txt \
                      --dataset_path dataset.pt --processes_num 1 \
                      --seq_length 128 --target lm 

python3 pretrain.py --dataset_path dataset.pt --pretrained_model_path models/cluecorpussmall_gpt2_distil_seq1024_model.bin \
                    --vocab_path models/google_zh_vocab.txt --config_path models/gpt2/distil_config.json \
                    --output_model_path models/book_review_gpt2_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 100000 --save_checkpoint_steps 10000 --report_steps 5000 \
                    --learning_rate 5e-5 --batch_size 64 \
                    --embedding word_pos --remove_embedding_layernorm \
                    --encoder transformer --mask causal --layernorm_positioning pre \
                    --target lm --tie_weights
```
or use it on downstream classification dataset：
```
python3 run_classifier.py --pretrained_model_path models/cluecorpussmall_gpt2_distil_seq1024_model.bin \
                          --vocab_path models/google_zh_vocab.txt --config_path models/gpt2/distil_config.json \
                          --train_path datasets/douban_book_review/train.tsv --dev_path datasets/douban_book_review/dev.tsv --test_path datasets/douban_book_review/test.tsv \
                          --learning_rate 3e-5 --batch_size 64 --epochs_num 8 \
                          --embedding word_pos_seg --remove_embedding_layernorm \
                          --encoder transformer --mask causal --layernorm_positioning pre
```

GPT-2 model can be used for text generation. First of all, we create *story_beginning.txt* and enter the beginning of the text. Then we use *scripts/generate_lm.py* to do text generation:
```
python3 scripts/generate_lm.py --load_model_path models/cluecorpussmall_gpt2_distil_seq1024_model.bin \
                               --vocab_path models/google_zh_vocab.txt \
                               --config_path models/gpt2/distil_config.json --seq_length 128 \
                               --test_path story_beginning.txt --prediction_path story_full.txt \
                               --embedding word_pos --remove_embedding_layernorm \
                               --encoder transformer --mask causal --layernorm_positioning pre \
                               --target lm --tie_weights
```

## More pre-trained Weights
Pre-trained Chinese models from Google (in UER format):
<table>
<tr align="center"><th> Pre-trained model <th> Link <th> Description 
<tr align="center"><td> Wikizh+BertEncoder+BertTarget <td> https://share.weiyun.com/A1C49VPb <td> Google's pre-trained Chinese model from https://github.com/google-research/bert
<tr align="center"><td> CLUECorpus+<br>AlbertEncoder(base)+AlbertTarget <td> https://share.weiyun.com/UnKHNKRG <td> Google's pre-trained Chinese model from https://github.com/google-research/albert . <br>The configuration file is albert_base_config.json
<tr align="center"><td> CLUECorpus+<br>AlbertEncoder(large)+AlbertTarget <td> https://share.weiyun.com/9tTUwALd <td> Google's pre-trained Chinese model from https://github.com/google-research/albert . <br>The configuration file is albert_large_config.json
<tr align="center"><td> CLUECorpus+<br>AlbertEncoder(xlarge)+AlbertTarget <td> https://share.weiyun.com/mUamRQFR <td> Google's pre-trained Chinese model from https://github.com/google-research/albert . <br>The configuration file is albert_xlarge_config.json
<tr align="center"><td> CLUECorpus+<br>AlbertEncoder(xxlarge)+AlbertTarget <td> https://share.weiyun.com/0i2lX62b <td> Google's pre-trained Chinese model from https://github.com/google-research/albert . <br>The configuration file is albert_xxlarge_config.json
</table>

Models pre-trained by UER:
<table>
<tr align="center"><th> Pre-trained model <th> Link <th> Description 
<tr align="center"><td> Wikizh(word-based)+BertEncoder+BertTarget <td> Model: https://share.weiyun.com/5s4HVMi Vocab: https://share.weiyun.com/5NWYbYn <td> Word-based BERT model pre-trained on Wikizh. Training steps: 500,000
<tr align="center"><td> RenMinRiBao+BertEncoder+BertTarget <td> https://share.weiyun.com/5JWVjSE <td> The training corpus is news data from People's Daily (1946-2017).
<tr align="center"><td> Webqa2019+BertEncoder+BertTarget <td> https://share.weiyun.com/5HYbmBh <td> The training corpus is WebQA, which is suitable for datasets related with social media, e.g. LCQMC and XNLI. Training steps: 500,000
<tr align="center"><td> Weibo+BertEncoder+BertTarget <td> https://share.weiyun.com/5ZDZi4A <td> The training corpus is Weibo.
<tr align="center"><td> Weibo+BertEncoder(large)+MlmTarget <td> https://share.weiyun.com/CFKyMkp3 <td> The training corpus is Weibo. The configuration file is bert_large_config.json
<tr align="center"><td> Reviews+BertEncoder+MlmTarget <td> https://share.weiyun.com/tBgaSx77 <td> The training corpus is reviews.
<tr align="center"><td> Reviews+BertEncoder(large)+MlmTarget <td> https://share.weiyun.com/hn7kp9bs <td> The training corpus is reviews. The configuration file is bert_large_config.json
<tr align="center"><td> MixedCorpus+BertEncoder(xlarge)+MlmTarget <td> https://share.weiyun.com/J9rj9WRB <td> Pre-trained on mixed large Chinese corpus. The configuration file is bert_xlarge_config.json
<tr align="center"><td> MixedCorpus+BertEncoder(xlarge)+BertTarget(WWM) <td> https://share.weiyun.com/UsI0OSeR <td> Pre-trained on mixed large Chinese corpus. The configuration file is bert_xlarge_config.json
<tr align="center"><td> MixedCorpus+BertEncoder(large)+MlmTarget <td> https://share.weiyun.com/5G90sMJ <td> Pre-trained on mixed large Chinese corpus. The configuration file is bert_large_config.json
<tr align="center"><td> MixedCorpus+BertEncoder(base)+BertTarget <td> https://share.weiyun.com/5QOzPqq <td> Pre-trained on mixed large Chinese corpus. The configuration file is bert_base_config.json
<tr align="center"><td> MixedCorpus+BertEncoder(small)+BertTarget <td> https://share.weiyun.com/fhcUanfy <td> Pre-trained on mixed large Chinese corpus. The configuration file is bert_small_config.json
<tr align="center"><td> MixedCorpus+BertEncoder(tiny)+BertTarget <td> https://share.weiyun.com/yXx0lfUg <td> Pre-trained on mixed large Chinese corpus. The configuration file is bert_tiny_config.json
<tr align="center"><td> MixedCorpus+GptEncoder+LmTarget <td> https://share.weiyun.com/51nTP8V <td> Pre-trained on mixed large Chinese corpus. Training steps: 500,000 (with sequence lenght of 128) + 100,000 (with sequence length of 512)
<tr align="center"><td> Reviews+LstmEncoder+LmTarget <td> https://share.weiyun.com/57dZhqo  <td> The training corpus is amazon reviews + JDbinary reviews + dainping reviews (11.4M reviews in total). Language model target is used. It is suitable for datasets related with reviews. It achieves over 5 percent improvements on some review datasets compared with random initialization. Set hidden_size in models/rnn_config.json to 512 before using it. Training steps: 200,000; Sequence length: 128;
<tr align="center"><td> (MixedCorpus & Amazon reviews)+LstmEncoder+(LmTarget & ClsTarget) <td> https://share.weiyun.com/5B671Ik  <td> Firstly pre-trained on mixed large Chinese corpus with LM target. And then is pre-trained on Amazon reviews with lm target and cls target. It is suitable for datasets related with reviews. It can achieve comparable results with BERT on some review datasets. Training steps: 500,000 + 100,000; Sequence length: 128
<tr align="center"><td> IfengNews+BertEncoder+BertTarget <td> https://share.weiyun.com/5HVcUWO <td> The training corpus is news data from Ifeng website. We use news title to predict news abstract. Training steps: 100,000; Sequence length: 128
<tr align="center"><td> jdbinary+BertEncoder+ClsTarget <td> https://share.weiyun.com/596k2bu <td> The training corpus is review data from JD (jingdong). CLS target is used for pre-training. It is suitable for datasets related with shopping reviews. Training steps: 50,000; Sequence length: 128
<tr align="center"><td> jdfull+BertEncoder+MlmTarget <td> https://share.weiyun.com/5L6EkUF <td> The training corpus is review data from JD (jingdong). MLM target is used for pre-training. Training steps: 50,000; Sequence length: 128
<tr align="center"><td> Amazonreview+BertEncoder+ClsTarget <td> https://share.weiyun.com/5XuxtFA <td> The training corpus is review data from Amazon (including book reviews, movie reviews, and etc.). Classification target is used for pre-training. It is suitable for datasets related with reviews, e.g. accuracy is improved on Douban book review datasets from 87.6 to 88.5 (compared with Google BERT). Training steps: 20,000; Sequence length: 128
<tr align="center"><td> XNLI+BertEncoder+ClsTarget <td> https://share.weiyun.com/5oXPugA <td> Infersent with BertEncoder
</table>
MixedCorpus contains baidubaike, Wikizh, WebQA, RenMinRiBao, literature, and reviews.

<br/>

[2_128]:https://share.weiyun.com/5mXkEvxN
[2_256]:https://share.weiyun.com/TLn5VpSW
[2_512]:https://share.weiyun.com/nYtFjDBL
[2_768]:https://share.weiyun.com/sHMGMv4c
[4_128]:https://share.weiyun.com/0SlDbnSM
[4_256]:https://share.weiyun.com/0eNSFKw9
[4_512]:https://share.weiyun.com/SEJ0MStj
[4_768]:https://share.weiyun.com/jk42uruV  
[6_128]:https://share.weiyun.com/tWeXueMJ
[6_256]:https://share.weiyun.com/1Ku03xFM
[6_512]:https://share.weiyun.com/WEvmJUOD
[6_768]:https://share.weiyun.com/m5AMthBU
[8_128]:https://share.weiyun.com/r0M1pJsV
[8_256]:https://share.weiyun.com/xk1SN8V9
[8_512]:https://share.weiyun.com/vpnaZ5jp
[8_768]:https://share.weiyun.com/NT67VIap
[10_128]:https://share.weiyun.com/dPKrelIL
[10_256]:https://share.weiyun.com/dbn2bG0t
[10_512]:https://share.weiyun.com/q8yIZWje
[10_768]:https://share.weiyun.com/dz4sgiCx
[12_128]:https://share.weiyun.com/93UYnnSC
[12_256]:https://share.weiyun.com/czAR5KNu
[12_512]:https://share.weiyun.com/gv4zARxk
[12_768]:https://share.weiyun.com/2rEWrSQz

[word_tiny]:https://share.weiyun.com/6mUaN18A
[word_mini]:https://share.weiyun.com/og1Km7qM
[word_small]:https://share.weiyun.com/SqbCfIgp
[word_medium]:https://share.weiyun.com/yDz44dlS
[word_base]:https://share.weiyun.com/5OXC8Rzt

[gpt2_cluecorpussmall]:https://share.weiyun.com/0eAlQWRB
[gpt2_distil_cluecorpussmall]:https://share.weiyun.com/IAvDbjKR
[gpt2_poem]:https://share.weiyun.com/DKAmuOLU
[gpt2_couplet]:https://share.weiyun.com/LbMecOGj
[gpt2_lyric]:https://share.weiyun.com/jBv5weES
[gpt2_ancient]:https://share.weiyun.com/d3FHbVMx