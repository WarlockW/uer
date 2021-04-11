This section uses several commonly-used examples to demonstrate how to use UER-py. More details are discussed in [Instructions](https://github.com/dbiir/UER-py/wiki/instructions). We firstly use BERT model on [Douban book review classification dataset](https://embedding.github.io/evaluation/). We pre-train model on book review corpus and then fine-tune it on classification dataset. There are three input files: book review corpus, book review classification dataset, and vocabulary. All files are encoded in UTF-8 and are included in this project.

The format of the corpus for BERT is as follows (one sentence per line and documents are delimited by empty lines)：
```
doc1-sent1
doc1-sent2
doc1-sent3

doc2-sent1

doc3-sent1
doc3-sent2
```
The book review corpus is obtained by book review classification dataset. We remove labels and split a review into two parts from the middle (See *book_review_bert.txt* in *corpora* folder). 

The format of the classification dataset is as follows:
```
label    text_a
1        instance1
0        instance2
1        instance3
```
Label and instance are separated by \t . The first row is a list of column names. The label ID should be an integer between (and including) 0 and n-1 for n-way classification.

We use Google's Chinese vocabulary file *models/google_zh_vocab.txt*, which contains 21128 Chinese characters.

We firstly preprocess the book review corpus. We need to specify the model's target in pre-processing stage (*--target*):
```
python3 preprocess.py --corpus_path corpora/book_review_bert.txt --vocab_path models/google_zh_vocab.txt --dataset_path dataset.pt \
                      --processes_num 8 --target bert
```
Notice that ``six>=1.12.0`` is required.

Pre-processing is time-consuming. Using multiple processes can largely accelerate the pre-processing speed (*--processes_num*). After pre-processing, the raw text is converted to *dataset.pt*, which is the input of *pretrain.py*. Then we download Google's original pre-trained Chinese BERT model [*google_zh_model.bin*](https://share.weiyun.com/A1C49VPb) (in UER's format), and put it in *models* folder. We load the pre-trained Chinese BERT model and further pre-train it on book review corpus. Pre-training model is composed of embedding, encoder, and target. To build a pre-training model, we should explicitly specify model's embedding (*--embedding*), encoder (*--encoder* and *--mask*), and target (*--target*). Suppose we have a machine with 8 GPUs.:
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --pretrained_model_path models/google_zh_model.bin \
                    --output_model_path models/book_review_model.bin  --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 5000 --save_checkpoint_steps 1000 --batch_size 32 --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert

mv models/book_review_model.bin-5000 models/book_review_model.bin
```
*--mask* specifies the attention mask types. BERT uses bidirectional LM. The word token can attend to all tokens and therefore we use *fully_visible* mask type. By default, *models/bert/base_config.json* is used as configuration file, which specifies the model hyper-parameters. 
Notice that the model trained by *pretrain.py* is attacted with the suffix which records the training step. We could remove the suffix for ease of use.


Then we fine-tune pre-trained models on downstream classification dataset. We can use *google_zh_model.bin*:
```
python3 run_classifier.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                          --train_path datasets/douban_book_review/train.tsv --dev_path datasets/douban_book_review/dev.tsv --test_path datasets/douban_book_review/test.tsv \
                          --epochs_num 3 --batch_size 32 --embedding word_pos_seg --encoder transformer --mask fully_visible
```
or use our [*book_review_model.bin*](https://share.weiyun.com/xOFsYxZA), which is the output of *pretrain.py*：
```
python3 run_classifier.py --pretrained_model_path models/book_review_model.bin --vocab_path models/google_zh_vocab.txt \
                          --train_path datasets/douban_book_review/train.tsv --dev_path datasets/douban_book_review/dev.tsv --test_path datasets/douban_book_review/test.tsv \
                          --epochs_num 3 --batch_size 32 --embedding word_pos_seg --encoder transformer --mask fully_visible
``` 
It turns out that the result of Google's model is 87.5; The result of *book_review_model.bin* is 88.2. It is also noticeable that we don't need to specify the target in fine-tuning stage. Pre-training target is replaced with task-specific target.

The default path of the fine-tuned classifier model is *models/finetuned_model.bin* . Then we do inference with the fine-tuned model. 
```
python3 inference/run_classifier_infer.py --load_model_path models/finetuned_model.bin --vocab_path models/google_zh_vocab.txt \
                                          --test_path datasets/douban_book_review/test_nolabel.tsv \
                                          --prediction_path datasets/douban_book_review/prediction.tsv --labels_num 2 \
                                          --embedding word_pos_seg --encoder transformer --mask fully_visible
```
*--test_path* specifies the path of the file to be predicted. <br>
*--prediction_path* specifies the path of the file with prediction results. <br>
We need to explicitly specify the number of labels by *--labels_num*. Douban book review is a two-way classification dataset.

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

BERT is slow. It could be great if we can speed up the model and still achieve competitive performance. To achieve this goal, we select a 2-layers LSTM encoder to substitute 12-layers Transformer encoder. We firstly download [*reviews_lstm_lm_model.bin*](https://share.weiyun.com/57dZhqo) for 2-layers LSTM encoder. Then we fine-tune it on downstream classification dataset:
```
python3 run_classifier.py --pretrained_model_path models/reviews_lstm_lm_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/rnn_config.json \
                          --train_path datasets/douban_book_review/train.tsv --dev_path datasets/douban_book_review/dev.tsv --test_path datasets/douban_book_review/test.tsv \
                          --epochs_num 5  --batch_size 64 --learning_rate 1e-3 --embedding word --encoder lstm --pooling mean

python3 inference/run_classifier_infer.py --load_model_path models/finetuned_model.bin --vocab_path models/google_zh_vocab.txt \
                                          --config_path models/rnn_config.json --test_path datasets/douban_book_review/test_nolabel.tsv \
                                          --prediction_path datasets/douban_book_review/prediction.tsv \
                                          --labels_num 2 --embedding word --encoder lstm --pooling mean
```
We can achieve over 85.4 accuracy on testset, which is a competitive result. Using the same LSTM encoder without pre-training can only achieve around 81 accuracy.
<br>

UER-py also provides many other encoders and corresponding pre-trained models. <br>
The example of pre-training and fine-tuning ELMo on Chnsenticorp dataset:
```
python3 preprocess.py --corpus_path corpora/chnsenticorp.txt --vocab_path models/google_zh_vocab.txt --dataset_path dataset.pt \
                      --processes_num 8 --seq_length 192 --target bilm

python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --pretrained_model_path models/mixed_corpus_elmo_model.bin \
                    --config_path models/birnn_config.json \
                    --output_model_path models/chnsenticorp_elmo_model.bin --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 5000 --save_checkpoint_steps 2500 --batch_size 64 --learning_rate 5e-4 \
                    --embedding word --encoder bilstm --target bilm

mv models/chnsenticorp_elmo_model.bin-5000 models/chnsenticorp_elmo_model.bin

python3 run_classifier.py --pretrained_model_path models/chnsenticorp_elmo_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/birnn_config.json \
                          --train_path datasets/chnsenticorp/train.tsv --dev_path datasets/chnsenticorp/dev.tsv --test_path datasets/chnsenticorp/test.tsv \
                          --epochs_num 5  --batch_size 64 --seq_length 192 --learning_rate 5e-4 \
                          --embedding word --encoder bilstm --pooling mean
```
Users can download *mixed_corpus_elmo_model.bin* from [here](https://share.weiyun.com/5Qihztq).

The example of fine-tuning GatedCNN on Chnsenticorp dataset:
```
python3 run_classifier.py --pretrained_model_path models/wikizh_gatedcnn_lm_model.bin \
                          --vocab_path models/google_zh_vocab.txt \
                          --config_path models/gatedcnn_9_config.json \
                          --train_path datasets/chnsenticorp/train.tsv --dev_path datasets/chnsenticorp/dev.tsv --test_path datasets/chnsenticorp/test.tsv \
                          --epochs_num 5  --batch_size 64 --learning_rate 5e-5 \
                          --embedding word --encoder gatedcnn --pooling max

python3 inference/run_classifier_infer.py --load_model_path models/finetuned_model.bin --vocab_path models/google_zh_vocab.txt \
                                          --config_path models/gatedcnn_9_config.json \
                                          --test_path datasets/chnsenticorp/test_nolabel.tsv \
                                          --prediction_path datasets/chnsenticorp/prediction.tsv \
                                          --labels_num 2 --embedding word --encoder gatedcnn --pooling max
```
Users can download *wikizh_gatedcnn_lm_model.bin* from [here](https://share.weiyun.com/W2gmPPeA).
<br>

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
The example of using our pre-trained model [*Reviews+BertEncoder(large)+MlmTarget*](https://share.weiyun.com/hn7kp9bs) (see model zoo for more details):
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

Besides classification, UER-py also provides scripts for other downstream tasks. For example, we could use *run_ner.py* for named entity recognition:
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
