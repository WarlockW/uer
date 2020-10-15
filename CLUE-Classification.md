Here is a short summary of our solution on [CLUE classification benchmark](https://www.cluebenchmarks.com/classification.html). We submitted two results, *UER* and *UER-ensemble* to the benchmark. The results of *UER* is based on the [*mixed_corpus_bert_large_model.bin*](https://share.weiyun.com/5G90sMJ) pre-trained weights. The results of *UER-ensemble* is based on the ensemble of a large number of models. This section mainly focuses on single model. More details of ensemble are discussed in [here](https://github.com/dbiir/UER-py/wiki/SMP2020-EWECT).

### AFQMC
We firstly do multi-task learning. We select LCQMC and XNLI as auxiliary tasks:
```
python3 run_mt_classifier.py --pretrained_model_path models/mixed_corpus_bert_large_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                             --dataset_path_list datasets/afqmc/ datasets/lcqmc/ datasets/xnli/ \
                             --output_model_path models/multitask_classifier_model.bin \
                             --epochs_num 1 --batch_size 64 --encoder bert
```
Then we load *multitask_classifier_model.bin* and fine-tune it on AFQMC:
```
python3 run_classifier.py --pretrained_model_path models/multitask_classifier_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                          --train_path datasets/douban_book_review/train.tsv --dev_path datasets/douban_book_review/dev.tsv --test_path datasets/douban_book_review/test.tsv \
                          --output_model_path models/afqmc_classifier_model.bin \
                          --epochs_num 3 --batch_size 32 --encoder bert
```
The example of doing inference:
```
python3 inference/run_classifier_infer.py --load_model_path models/afqmc_classifier_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                                          --test_path datasets/afqmc/test_nolabel.tsv \
                                          --prediction_path datasets/afqmc/prediction.tsv --labels_num 2 \
                                          --seq_length 128 --encoder bert
```
