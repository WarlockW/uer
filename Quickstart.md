## Pre-training and text classification with BERT
This section uses several commonly-used examples to demonstrate how to use UER-py. More details are discussed in Instructions. We firstly use BERT model on [Douban book review classification dataset](https://embedding.github.io/evaluation/). We pre-train model on book review corpus and then fine-tune it on classification dataset. There are three input files: book review corpus, book review classification dataset, and vocabulary. All files are encoded in UTF-8 and included in this project.

The format of the corpus for BERT is as follows (one sentence per line and documents are delimited by empty lines)：
```
doc1-sent1
doc1-sent2
doc1-sent3

doc2-sent1

doc3-sent1
doc3-sent2
```
The book review corpus is obtained from book review classification dataset. We remove labels and split a review into two parts from the middle (See *book_review_bert.txt* in *corpora* folder). 

The format of the classification dataset is as follows:
```
label    text_a
1        instance1
0        instance2
1        instance3
```
Label and instance are separated by \t . The first row is a list of column names. The label ID should be an integer between (and including) 0 and n-1 for n-way classification.

We use Google's Chinese vocabulary file *models/google_zh_vocab.txt*, which contains 21128 Chinese characters.

We firstly pre-process the book review corpus. We need to specify the model's target in pre-processing stage (*--target*):
```
python3 preprocess.py --corpus_path corpora/book_review_bert.txt --vocab_path models/google_zh_vocab.txt --dataset_path dataset.pt \
                      --processes_num 8 --target bert
```
Notice that ``six>=1.12.0`` is required.

Pre-processing is time-consuming. Using multiple processes can largely accelerate the pre-processing speed (*--processes_num*). BERT tokenizer is used in default (*--tokenizer bert*). After pre-processing, the raw text is converted to *dataset.pt*, which is the input of *pretrain.py*. Then we download Google's pre-trained Chinese BERT model [*google_zh_model.bin*](https://share.weiyun.com/A1C49VPb) (in UER format and the original model is from [here](https://github.com/google-research/bert)), and put it in *models* folder. We load the pre-trained Chinese BERT model and further pre-train it on book review corpus. Pre-training model is composed of embedding, encoder, and target layers. To build a pre-training model, we should explicitly specify model's embedding (*--embedding*), encoder (*--encoder* and *--mask*), and target (*--target*). Suppose we have a machine with 8 GPUs:
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --pretrained_model_path models/google_zh_model.bin \
                    --output_model_path models/book_review_model.bin  --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 5000 --save_checkpoint_steps 1000 --batch_size 32 --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert

mv models/book_review_model.bin-5000 models/book_review_model.bin
```
*--mask* specifies the attention mask types. BERT uses bidirectional LM. The word token can attend to all tokens and therefore we use *fully_visible* mask type. The embedding layer of BERT is the sum of word (token), position, and segment embeddings and therefore *--embedding word_pos_seg* is specified. By default, *models/bert/base_config.json* is used as configuration file, which specifies the model hyper-parameters. 
Notice that the model trained by *pretrain.py* is attacted with the suffix which records the training step (*--total_steps*). We could remove the suffix for ease of use.


Then we fine-tune the pre-trained model on downstream classification dataset. We use [*book_review_model.bin*](https://share.weiyun.com/xOFsYxZA), which is the output of *pretrain.py*:
```
python3 run_classifier.py --pretrained_model_path models/book_review_model.bin --vocab_path models/google_zh_vocab.txt \
                          --train_path datasets/douban_book_review/train.tsv --dev_path datasets/douban_book_review/dev.tsv --test_path datasets/douban_book_review/test.tsv \
                          --epochs_num 3 --batch_size 32 --embedding word_pos_seg --encoder transformer --mask fully_visible
``` 
The result of *book_review_model.bin* on test set is 88.2. It is also noticeable that we don't need to specify the target in fine-tuning stage. Pre-training target is replaced with task-specific target.

The default path of the fine-tuned classifier model is *models/finetuned_model.bin* . Then we do inference with the fine-tuned model. 
```
python3 inference/run_classifier_infer.py --load_model_path models/finetuned_model.bin --vocab_path models/google_zh_vocab.txt \
                                          --test_path datasets/douban_book_review/test_nolabel.tsv \
                                          --prediction_path datasets/douban_book_review/prediction.tsv --labels_num 2 \
                                          --embedding word_pos_seg --encoder transformer --mask fully_visible
```
*--test_path* specifies the path of the file to be predicted. The file should contain text_a column. <br>
*--prediction_path* specifies the path of the file with prediction results. <br>
We need to explicitly specify the number of labels by *--labels_num*. Douban book review is a two-way classification dataset.

We can also use *google_zh_model.bin* and fine-tune it on downstream classification dataset:
```
python3 run_classifier.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                          --train_path datasets/douban_book_review/train.tsv --dev_path datasets/douban_book_review/dev.tsv --test_path datasets/douban_book_review/test.tsv \
                          --epochs_num 3 --batch_size 32 --embedding word_pos_seg --encoder transformer --mask fully_visible
```
It turns out that the result of Google's model is 87.5.

<br>

## Specifying which GPUs are used
We recommend to use *CUDA_VISIBLE_DEVICES* to specify which GPUs are visible (all GPUs are used in default). Suppose GPU 0 and GPU 2 are available:
```
python3 preprocess.py --corpus_path corpora/book_review_bert.txt --vocab_path models/google_zh_vocab.txt --dataset_path dataset.pt \
                      --processes_num 8 --target bert

CUDA_VISIBLE_DEVICES=0,2 python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --pretrained_model_path models/google_zh_model.bin \
                                             --output_model_path models/book_review_model.bin  --world_size 2 --gpu_ranks 0 1 \
                                             --total_steps 5000 --save_checkpoint_steps 1000 --batch_size 32 --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert

mv models/book_review_model.bin-5000 models/book_review_model.bin

CUDA_VISIBLE_DEVICES=0,2 python3 run_classifier.py --pretrained_model_path models/book_review_model.bin --vocab_path models/google_zh_vocab.txt \
                                                   --train_path datasets/douban_book_review/train.tsv --dev_path datasets/douban_book_review/dev.tsv --test_path datasets/douban_book_review/test.tsv \
                                                   --output_model_path models/classifier_model.bin \
                                                   --epochs_num 3 --batch_size 32 --embedding word_pos_seg --encoder transformer --mask fully_visible

CUDA_VISIBLE_DEVICES=0,2 python3 inference/run_classifier_infer.py --load_model_path models/classifier_model.bin --vocab_path models/google_zh_vocab.txt \
                                                                   --test_path datasets/douban_book_review/test_nolabel.tsv \
                                                                   --prediction_path datasets/douban_book_review/prediction.tsv --labels_num 2 \
                                                                   --embedding word_pos_seg --encoder transformer --mask fully_visible
```
Notice that we explicitly specify the fine-tuned model path by *--output_model_path* in fine-tuning stage. The actual batch size of pre-training is *--batch_size* times *--world_size* ; The actual batch size of classification is *--batch_size* . 

<br>

## Pre-training with MLM target
BERT consists of next sentence prediction (NSP) target. However, NSP target is not suitable for sentence-level reviews since we have to split a sentence into multiple parts to construct document. UER-py facilitates the use of different targets. Using masked language modeling (MLM) as target could be a properer choice for pre-training of reviews:
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_zh_vocab.txt --dataset_path dataset.pt \
                      --processes_num 8 --target mlm

python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --pretrained_model_path models/google_zh_model.bin \
                    --output_model_path models/book_review_mlm_model.bin  --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 5000 --save_checkpoint_steps 2500 --batch_size 32 --embedding word_pos_seg --encoder transformer --mask fully_visible --target mlm

mv models/book_review_mlm_model.bin-5000 models/book_review_mlm_model.bin

CUDA_VISIBLE_DEVICES=0,1 python3 run_classifier.py --pretrained_model_path models/book_review_mlm_model.bin --vocab_path models/google_zh_vocab.txt \
                                                   --train_path datasets/douban_book_review/train.tsv --dev_path datasets/douban_book_review/dev.tsv --test_path datasets/douban_book_review/test.tsv \
                                                   --epochs_num 3 --batch_size 32 --embedding word_pos_seg --encoder transformer --mask fully_visible
```
Different targets require different corpus formats. The format of the corpus for MLM target is as follows (one document per line):
```
doc1
doc2
doc3
``` 
Notice that *corpora/book_review.txt* (instead of *corpora/book_review_bert.txt*) is used when the target is switched to MLM. 

<br>

## Using more encoders besides Transformer
BERT is slow. It could be great if we can speed up the model and still achieve competitive performance. To achieve this goal, we select a 2-layers LSTM encoder to substitute 12-layers Transformer encoder. We firstly download [*cluecorpussmall_lstm_lm_model.bin*](https://share.weiyun.com/XFc4hcn6) for 2-layers LSTM encoder. The model is pre-trained on [CLUECorpusSmall](https://github.com/CLUEbenchmark/CLUECorpus2020) corpus for 500,000 steps:
```
python3 preprocess.py --corpus_path corpora/cluecorpussmall.txt --vocab_path models/google_zh_vocab.txt --dataset_path dataset.pt \
                     --processes_num 8 --seq_length 256 --target lm

python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --output_model_path models/cluecorpussmall_lstm_lm_model.bin \
                    --config_path models/rnn_config.json \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 500000 --save_checkpoint_steps 100000 \
                    --learning_rate 1e-3 --batch_size 64 \
                    --embedding word --remove_embedding_layernorm --encoder lstm --target lm
```
Then we remove the training step suffix of pre-trained model and fine-tune it on downstream classification dataset:
```
python3 run_classifier.py --pretrained_model_path models/cluecorpussmall_lstm_lm_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/rnn_config.json \
                          --train_path datasets/douban_book_review/train.tsv --dev_path datasets/douban_book_review/dev.tsv --test_path datasets/douban_book_review/test.tsv \
                          --learning_rate 1e-3 --batch_size 64 --epochs_num 5 \
                          --embedding word --remove_embedding_layernorm --encoder lstm --pooling mean

python3 inference/run_classifier_infer.py --load_model_path models/finetuned_model.bin --vocab_path models/google_zh_vocab.txt \
                                          --config_path models/rnn_config.json \
                                          --test_path datasets/douban_book_review/test_nolabel.tsv \
                                          --prediction_path datasets/douban_book_review/prediction.tsv \
                                          --labels_num 2 --embedding word --remove_embedding_layernorm --encoder lstm --pooling mean
```
We can achieve over 84.6 accuracy on testset, which is a competitive result. Using the same LSTM encoder without pre-training can only achieve around 81 accuracy.
<br>

UER-py also includes many other pre-training models. <br>
We download [*cluecorpussmall_elmo_model.bin*](https://share.weiyun.com/xezGTd86) for pre-trained ELMo model. The model is pre-trained on CLUECorpusSmall corpus for 500,000 steps:
```
python3 preprocess.py --corpus_path corpora/cluecorpussmall.txt --vocab_path models/google_zh_vocab.txt --dataset_path dataset.pt \
                     --processes_num 8 --seq_length 256 --target bilm

python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --output_model_path models/cluecorpussmall_elmo_model.bin \
                    --config_path models/birnn_config.json \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 500000 --save_checkpoint_steps 100000 \
                    --learning_rate 5e-4 --batch_size 64 \
                    --embedding word --remove_embedding_layernorm --encoder bilstm --target bilm
```
We remove the training step suffix of pre-trained model. Then we do further pre-training and fine-tune on Chnsenticorp sentiment classification dataset:
```
python3 preprocess.py --corpus_path corpora/chnsenticorp.txt --vocab_path models/google_zh_vocab.txt --dataset_path dataset.pt \
                      --processes_num 8 --seq_length 192 --target bilm

python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --pretrained_model_path models/cluecorpussmall_elmo_model.bin \
                    --config_path models/birnn_config.json \
                    --output_model_path models/chnsenticorp_elmo_model.bin --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 5000 --save_checkpoint_steps 2500 --batch_size 64 --learning_rate 5e-4 \
                    --embedding word --remove_embedding_layernorm --encoder bilstm --target bilm

mv models/chnsenticorp_elmo_model.bin-5000 models/chnsenticorp_elmo_model.bin

python3 run_classifier.py --pretrained_model_path models/chnsenticorp_elmo_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/birnn_config.json \
                          --train_path datasets/chnsenticorp/train.tsv --dev_path datasets/chnsenticorp/dev.tsv --test_path datasets/chnsenticorp/test.tsv \
                          --epochs_num 5  --batch_size 64 --seq_length 192 --learning_rate 5e-4 \
                          --embedding word --remove_embedding_layernorm --encoder bilstm --pooling max

python3 inference/run_classifier_infer.py --load_model_path models/finetuned_model.bin --vocab_path models/google_zh_vocab.txt \
                                          --config_path models/birnn_config.json \
                                          --test_path datasets/chnsenticorp/test_nolabel.tsv \
                                          --prediction_path datasets/chnsenticorp/prediction.tsv \
                                          --labels_num 2 --embedding word --remove_embedding_layernorm --encoder bilstm --pooling max
```
*corpora/chnsenticorp.txt* is obtained from Chnsenticorp dataset and labels are removed.

The example of fine-tuning GatedCNN on Chnsenticorp dataset:
```
python3 run_classifier.py --pretrained_model_path models/cluecorpussmall_gatedcnn_lm_model.bin \
                          --vocab_path models/google_zh_vocab.txt \
                          --config_path models/gatedcnn_9_config.json \
                          --train_path datasets/chnsenticorp/train.tsv --dev_path datasets/chnsenticorp/dev.tsv --test_path datasets/chnsenticorp/test.tsv \
                          --epochs_num 5  --batch_size 64 --learning_rate 5e-5 \
                          --embedding word --remove_embedding_layernorm --encoder gatedcnn --pooling mean

python3 inference/run_classifier_infer.py --load_model_path models/finetuned_model.bin --vocab_path models/google_zh_vocab.txt \
                                          --config_path models/gatedcnn_9_config.json \
                                          --test_path datasets/chnsenticorp/test_nolabel.tsv \
                                          --prediction_path datasets/chnsenticorp/prediction.tsv \
                                          --labels_num 2 --embedding word --remove_embedding_layernorm --encoder gatedcnn --pooling mean
```
Users can download *cluecorpussmall_gatedcnn_lm_model.bin* from [here](https://share.weiyun.com/VLe8O6kM). The model is pre-trained on CLUECorpusSmall corpus for 500,000 steps:
```
python3 preprocess.py --corpus_path corpora/cluecorpussmall.txt --vocab_path models/google_zh_vocab.txt --dataset_path dataset.pt \
                     --processes_num 8 --seq_length 256 --target lm

python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --config_path models/gatedcnn_9_config.json \
                    --output_model_path models/cluecorpussmall_gatedcnn_lm_model.bin --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 500000 --save_checkpoint_steps 100000 --report_steps 100 --learning_rate 1e-4 --batch_size 64 \
                    --embedding word --remove_embedding_layernorm --encoder gatedcnn --target lm
```

<br>

## Cross validation for classification
UER-py supports cross validation for classification. The example of using cross validation on [SMP2020-EWECT](http://39.97.118.137/), a competition dataset:
```
CUDA_VISIBLE_DEVICES=0 python3 run_classifier_cv.py --pretrained_model_path models/google_zh_model.bin \
                                                    --vocab_path models/google_zh_vocab.txt \
                                                    --config_path models/bert/base_config.json \
                                                    --output_model_path models/classifier_model.bin \
                                                    --train_features_path datasets/smp2020-ewect/virus/train_features.npy \
                                                    --train_path datasets/smp2020-ewect/virus/train.tsv \
                                                    --epochs_num 3 --batch_size 32 --folds_num 5 \
                                                    --embedding word_pos_seg --encoder transformer --mask fully_visible
```
The results of *google_zh_model.bin* are 79.1/63.8 (Accuracy/Marco F1). <br>
*--folds_num* specifies the number of rounds of cross-validation. <br>
*--output_path* specifies the path of the fine-tuned model. *--folds_num* models are saved and the *fold ID* suffix is added to the model's name. <br>
*--train_features_path* specifies the path of out-of-fold (OOF) predictions. *run_classifier_cv.py* generates probabilities over classes on each fold by training a model on the other folds in the dataset. *train_features.npy* can be used as features for stacking. More details are introduced in [*Competition solutions*](https://github.com/dbiir/UER-py/wiki/Competition-solutions) section. <br>

We can further try different pre-trained models. For example, we download [*RoBERTa-wwm-ext-large from HIT*](https://github.com/ymcui/Chinese-BERT-wwm) and convert it into UER's format:
```
python3 scripts/convert_bert_from_huggingface_to_uer.py --input_model_path models/chinese_roberta_wwm_large_ext_pytorch/pytorch_model.bin \
                                                        --output_model_path models/chinese_roberta_wwm_large_ext_pytorch/pytorch_model_uer.bin \
                                                        --layers_num 24

CUDA_VISIBLE_DEVICES=0,1 python3 run_classifier_cv.py --pretrained_model_path models/chinese_roberta_wwm_large_ext_pytorch/pytorch_model_uer.bin \
                                                      --vocab_path models/google_zh_vocab.txt \
                                                      --config_path models/bert/large_config.json \
                                                      --train_path datasets/smp2020-ewect/virus/train.tsv \
                                                      --train_features_path datasets/smp2020-ewect/virus/train_features.npy \
                                                      --epochs_num 3 --batch_size 64 --folds_num 5 \
                                                      --embedding word_pos_seg --encoder transformer --mask fully_visible
```
The results of *RoBERTa-wwm-ext-large* are 80.3/66.8 (Accuracy/Marco F1). <br>
The example of using our [review-corpus RoBERTa-large](https://share.weiyun.com/hn7kp9bs) pre-trained model:
```
CUDA_VISIBLE_DEVICES=0,1 python3 run_classifier_cv.py --pretrained_model_path models/reviews_bert_large_mlm_model.bin \
                                                      --vocab_path models/google_zh_vocab.txt \
                                                      --config_path models/bert/large_config.json \
                                                      --train_path datasets/smp2020-ewect/virus/train.tsv \
                                                      --train_features_path datasets/smp2020-ewect/virus/train_features.npy \
                                                      --folds_num 5 --epochs_num 3 --batch_size 64 --seed 17 \
                                                      --embedding word_pos_seg --encoder transformer --mask fully_visible
```
The results are 81.3/68.4 (Accuracy/Marco F1), which are very competitive compared with other open-source pre-trained models. The corpus used by the above pre-trained model is highly similar with SMP2020-EWECT, a Weibo review dataset. <br>
Sometimes large model does not converge. We need to try different random seeds by specifying *--seed*. 

<br>

## Downstream task fine-tuning with BERT
Besides classification, UER-py also supports other downstream tasks. For example, *run_classifier.py* can be also used for text pair classification. We can download the text pair classification dataset LCQMC in Datasets section and fine-tune the pre-trained model on it: 
```
python3 run_classifier.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                          --train_path datasets/lcqmc/train.tsv --dev_path datasets/lcqmc/dev.tsv --test_path datasets/lcqmc/test.tsv \
                          --output_model_path models/classifier_model.bin \
                          --batch_size 32 --epochs_num 3 --seq_length 128 \
                          --embedding word_pos_seg --encoder transformer --mask fully_visible
```
For text pair classification, the dataset should contain text_a, text_b, and label columns.

Then we do inference with the fine-tuned text pair classification model:
```
python3 inference/run_classifier_infer.py --load_model_path models/classifier_model.bin --vocab_path models/google_zh_vocab.txt \
                                          --test_path datasets/lcqmc/test.tsv \
                                          --prediction_path datasets/lcqmc/prediction.tsv --labels_num 2 --seq_length 128 \
                                          --embedding word_pos_seg --encoder transformer --mask fully_visible
```
The file to be predicted (*--test_path*) should contain text_a and text_b columns.
<br>

We could use *run_ner.py* for named entity recognition:
```
python3 run_ner.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                   --train_path datasets/msra_ner/train.tsv --dev_path datasets/msra_ner/dev.tsv --test_path datasets/msra_ner/test.tsv \
                   --output_model_path models/ner_model.bin \
                   --label2id_path datasets/msra_ner/label2id.json --epochs_num 5 --batch_size 16 \
                   --embedding word_pos_seg --encoder transformer --mask fully_visible
```
*--label2id_path* specifies the path of label2id file for named entity recognition.
Then we do inference with the fine-tuned ner model:
```
python3 inference/run_ner_infer.py --load_model_path models/ner_model.bin --vocab_path models/google_zh_vocab.txt \
                                   --test_path datasets/msra_ner/test_nolabel.tsv \
                                   --prediction_path datasets/msra_ner/prediction.tsv \
                                   --label2id_path datasets/msra_ner/label2id.json \
                                   --embedding word_pos_seg --encoder transformer --mask fully_visible
```
<br>

We could use *run_cmrc.py* for machine reading comprehension:
```
python3 run_cmrc.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                    --train_path datasets/cmrc2018/train.json --dev_path datasets/cmrc2018/dev.json \
                    --output_model_path models/cmrc_model.bin \
                    --epochs_num 2 --batch_size 8 --seq_length 512 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible
```
We don't specify the *--test_path* because CMRC2018 dataset doesn't provide labels for testset. 
Then we do inference with the fine-tuned cmrc model:
```
python3 inference/run_cmrc_infer.py --load_model_path models/cmrc_model.bin --vocab_path models/google_zh_vocab.txt \
                                    --test_path datasets/cmrc2018/test.json \
                                    --prediction_path datasets/cmrc2018/prediction.json --seq_length 512 \
                                    --embedding word_pos_seg --encoder transformer --mask fully_visible
```

<br>

## Downstream task fine-tuning and text generation with language model
The example of fine-tuning GPT-2 on classification dataset:
```
python3 run_classifier.py --pretrained_model_path models/cluecorpussmall_gpt2_seq1024_model.bin --vocab_path models/google_zh_vocab.txt \
                          --config_path models/gpt2/config.json \
                          --train_path datasets/douban_book_review/train.tsv --dev_path datasets/douban_book_review/dev.tsv --test_path datasets/douban_book_review/test.tsv \
                          --epochs_num 3 --batch_size 32 \
                          --embedding word_pos --remove_embedding_layernorm \
                          --encoder transformer --mask causal --layernorm_positioning pre --pooling mean
```
The example of using GPT-2 to generate text:
```
python3 scripts/generate_lm.py --load_model_path models/cluecorpussmall_gpt2_seq1024_model.bin-250000 \
                               --vocab_path models/google_zh_vocab.txt \
                               --test_path beginning.txt --prediction_path generated_text.txt \
                               --config_path models/gpt2/config.json --seq_length 128 \
                               --embedding word_pos --remove_embedding_layernorm \
                               --encoder transformer --mask causal --layernorm_positioning pre \
                               --target lm --tie_weights
```
Users can download *cluecorpussmall_gpt2_seq1024_model.bin* from [here](https://share.weiyun.com/0eAlQWRB).

The example of using [LSTM language model](https://share.weiyun.com/XFc4hcn6) to generate text:
```
python3 scripts/generate_lm.py --load_model_path models/cluecorpussmall_lstm_lm_model.bin --vocab_path models/google_zh_vocab.txt \
                               --test_path beginning.txt --prediction_path generated_text.txt \
                               --config_path models/rnn_config.json --seq_length 128 \
                               --embedding word --remove_embedding_layernorm \
                               --encoder lstm --target lm
```

The example of using [GatedCNN language model](https://share.weiyun.com/VLe8O6kM) to generate text:
```
python3 scripts/generate_lm.py --load_model_path models/cluecorpussmall_gatedcnn_lm_model.bin --vocab_path models/google_zh_vocab.txt \
                               --test_path beginning.txt --prediction_path generated_text.txt \
                               --config_path models/gatedcnn_9_config.json --seq_length 128 \
                               --embedding word --remove_embedding_layernorm \
                               --encoder gatedcnn --target lm
```

<br>

## Using different tokenizers and vocabularies
In most cases, we use *--vocab_path models/google_zh_vocab.txt* and *--tokenizer bert* to tokenize the text. Since most scripts in this project use *--tokenizer bert* in default, *--tokenizer* is not usually explicitly specified. Next we show more use cases of tokenizers and vocabularies.

*--tokenizer bert* is based on character when processing Chinese. To pre-train word-based model and fine-tine it, we firstly do word segmentation on corpus and words are separated by spaces. Then we build vocabulary based on the corpus:
```
python3 scripts/build_vocab.py --corpus_path corpora/book_review_seg.txt \
                               --vocab_path models/book_review_word_vocab.txt \
                               --tokenizer space --workers_num 8 --min_count 5 
```
*--tokenizer space* is used in pre-process and pre-training stages since spaces are used to separate words. The examples of pre-process and pre-train word-based model:
```
python3 preprocess.py --corpus_path corpora/book_review_seg.txt \
                      --vocab_path models/book_review_word_vocab.txt  --tokenizer space \
                      --dataset_path book_review_word_dataset.pt \
                      --processes_num 8 --seq_length 128 --dynamic_masking --target mlm

python3 pretrain.py --dataset_path book_review_word_dataset.pt \
                    --vocab_path models/book_review_word_vocab.txt  --tokenizer space \
                    --output_model_path models/book_review_word_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 5000 --save_checkpoint_steps 2500 --report_steps 500 \
                    --learning_rate 1e-4 --batch_size 64 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target mlm --tie_weights
```
In fine-tuning and inference stages, we also need to explicitly specify *--vocab_path models/book_review_word_vocab.txt* and *--tokenizer space*. The text in train/dev/text datasets (text_a and text_b columns) should be processed by the same word segmentation tool. We do word segmentation on *datasets/douban_book_review/* dataset to obtain *datasets/douban_book_review_seg/*:
```
mv models/book_review_word_model.bin-5000 models/book_review_word_model.bin

python3 run_classifier.py --pretrained_model_path models/book_review_word_model.bin \
                          --vocab_path models/book_review_word_vocab.txt  --tokenizer space \
                          --train_path datasets/douban_book_review_seg/train.tsv --dev_path datasets/douban_book_review_seg/dev.tsv --test_path datasets/douban_book_review_seg/test.tsv \
                          --epochs_num 3 --batch_size 32 --embedding word_pos_seg --encoder transformer --mask fully_visible

python3 inference/run_classifier_infer.py --load_model_path models/finetuned_model.bin \
                                          --vocab_path models/book_review_word_vocab.txt  --tokenizer space \
                                          --test_path datasets/douban_book_review_seg/test_nolabel.tsv \
                                          --prediction_path datasets/douban_book_review_seg/prediction.tsv --labels_num 2 \
                                          --embedding word_pos_seg --encoder transformer --mask fully_visible
```

The example of using SentencePiece:
```
python3 preprocess.py --corpus_path corpora/book_review.txt \
                      --spm_model_path models/cluecorpussmall_spm.model \
                      --dataset_path book_review_word_sp_dataset.pt \
                      --processes_num 8 --seq_length 128 --dynamic_masking --target mlm

python3 pretrain.py --dataset_path book_review_word_sp_dataset.pt \
                    --spm_model_path models/cluecorpussmall_spm.model \
                    --output_model_path models/book_review_word_sp_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 5000 --save_checkpoint_steps 2500 --report_steps 500 \
                    --learning_rate 1e-4 --batch_size 64 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target mlm --tie_weights

mv models/book_review_word_sp_model.bin-5000 models/book_review_word_sp_model.bin

python3 run_classifier.py --pretrained_model_path models/book_review_word_sp_model.bin \
                          --spm_model_path models/cluecorpussmall_spm.model \
                          --train_path datasets/douban_book_review/train.tsv --dev_path datasets/douban_book_review/dev.tsv --test_path datasets/douban_book_review/test.tsv \
                          --epochs_num 3 --batch_size 32 --embedding word_pos_seg --encoder transformer --mask fully_visible

python3 inference/run_classifier_infer.py --load_model_path models/finetuned_model.bin \
                                          --spm_model_path models/cluecorpussmall_spm.model \
                                          --test_path datasets/douban_book_review/test_nolabel.tsv \
                                          --prediction_path datasets/douban_book_review/prediction.tsv --labels_num 2 \
                                          --embedding word_pos_seg --encoder transformer --mask fully_visible
```

To use character-based tokenizer, one can use *--vocab_path models/google_zh_vocab.txt* and *--tokenizer char* to substitute *--spm_model_path models/cluecorpussmall_spm.model* and other options are the same as above.
*--vocab_path models/google_zh_vocab.txt* can be used since it is also character-based for Chinese.

More details can be found in [Tokenization and vocabulary](https://github.com/dbiir/UER-py/wiki/https://github.com/dbiir/UER-py/wiki/Tokenization-and-vocabulary)