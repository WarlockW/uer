Here is a short summary of our solution on [CLUE classification benchmark](https://www.cluebenchmarks.com/classification.html). We submitted two results, *UER* and *UER-ensemble* to the benchmark. The results of *UER* is based on the [*mixed_corpus_bert_large_model.bin*](https://share.weiyun.com/5G90sMJ) pre-trained weights. The results of *UER-ensemble* is based on the ensemble of a large number of models. This section mainly focuses on single model. More details of ensemble are discussed in [here](https://github.com/dbiir/UER-py/wiki/SMP2020-EWECT).

### AFQMC
We firstly do multi-task learning. We select LCQMC and XNLI as auxiliary tasks:
```
python3 run_mt_classifier.py --pretrained_model_path models/mixed_corpus_bert_large_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                             --dataset_path_list datasets/afqmc/ datasets/lcqmc/ datasets/xnli/ \
                             --output_model_path models/afqmc_multitask_classifier_model.bin \
                             --epochs_num 1 --batch_size 64 --encoder bert
```
Then we load *afqmc_multitask_classifier_model.bin* and fine-tune it on AFQMC:
```
python3 run_classifier.py --pretrained_model_path models/afqmc_multitask_classifier_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                          --train_path datasets/afqmc/train.tsv --dev_path datasets/afqmc/dev.tsv \
                          --output_model_path models/afqmc_classifier_model.bin \
                          --epochs_num 3 --batch_size 32 --encoder bert
```
Then we do inference with *afqmc_classifier_model.bin*:
```
python3 inference/run_classifier_infer.py --load_model_path models/afqmc_classifier_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                                          --test_path datasets/afqmc/test_nolabel.tsv \
                                          --prediction_path datasets/afqmc/prediction.tsv --labels_num 2 \
                                          --seq_length 128 --encoder bert
```

### CMNLI
We firstly do multi-task learning. We select XNLI as auxiliary task:
```
python3 run_mt_classifier.py --pretrained_model_path models/mixed_corpus_bert_large_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                             --dataset_path_list datasets/cmnli/ datasets/xnli/ \
                             --output_model_path models/cmnli_multitask_classifier_model.bin \
                             --epochs_num 1 --batch_size 64 --encoder bert
```
Then we load *cmnli_multitask_classifier_model.bin* and fine-tune it on CMNLI:
```
python3 run_classifier.py --pretrained_model_path models/cmnli_multitask_classifier_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                          --train_path datasets/cmnli/train.tsv --dev_path datasets/cmnli/dev.tsv \
                          --output_model_path models/cmnli_classifier_model.bin \
                          --epochs_num 1 --batch_size 64 --encoder bert
```
Then we do inference with *cmnli_classifier_model.bin*:
```
python3 inference/run_classifier_infer.py --load_model_path models/cmnli_classifier_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                                          --test_path datasets/cmnli/test_nolabel.tsv \
                                          --prediction_path datasets/cmnli/prediction.tsv --labels_num 3 \
                                          --seq_length 128 --encoder bert
```

### IFLYTEK
The example of fine-tuning and doing inference on IFLYTEK dataset:
```
python3 run_classifier.py --pretrained_model_path models/mixed_corpus_bert_large_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                          --train_path datasets/iflytek/train.tsv --dev_path datasets/iflytek/dev.tsv \
                          --output_model_path models/iflytek_classifier_model.bin \
                          --epochs_num 3 --batch_size 32 --seq_length 256 --encoder bert

python3 inference/run_classifier_infer.py --load_model_path models/iflytek_classifier_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                                          --test_path datasets/iflytek/test_nolabel.tsv \
                                          --prediction_path datasets/iflytek/prediction.tsv --labels_num 119 \
                                          --seq_length 256 --encoder bert
```

### CSL
Chinese Scientific Literature task is to tell whether the given keywords are real keywords of a paper or not. The key of achieving competitive results on CSL is to use a special symbol to split keywords. We find that the pseudo keywords in CSL dataset are usually short. Special symbols can explicitly tell the model the length of keywords.
The example of fine-tuning and doing inference on CSL dataset:
```
python3 run_classifier.py --pretrained_model_path models/mixed_corpus_bert_large_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                          --train_path datasets/csl/train.tsv --dev_path datasets/csl/dev.tsv \
                          --output_model_path models/csl_classifier_model.bin \
                          --epochs_num 3 --batch_size 32 --seq_length 384 --encoder bert

python3 inference/run_classifier_infer.py --load_model_path models/csl_classifier_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                                          --test_path datasets/csl/test_nolabel.tsv \
                                          --prediction_path datasets/csl/prediction.tsv --labels_num 2 \
                                          --seq_length 384 --encoder bert
```

### CLUEWSC2020：
The example of fine-tuning and doing inference on CLUEWSC2020 dataset:
```
python3 run_classifier.py --pretrained_model_path models/mixed_corpus_bert_large_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                          --train_path datasets/cluewsc2020/train.tsv --dev_path datasets/cluewsc2020/dev.tsv \
                          --output_model_path models/cluewsc2020_classifier_model.bin \
                          --epochs_num 20 --batch_size 8 --learning_rate 5e-6 --encoder bert

python3 inference/run_classifier_infer.py --load_model_path models/cluewsc2020_classifier_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                                          --test_path datasets/cluewsc2020/test_nolabel.tsv \
                                          --prediction_path datasets/cluewsc2020/prediction.tsv --labels_num 119 \
                                          --seq_length 128 --encoder bert
```
A useful trick for CLUEWSC2020 is to use the trainset of WSC (the former version of CLUEWSC2020) as training samples.

### TNEWS
The example of fine-tuning and doing inference on TNEWS dataset:
```
python3 run_classifier.py --pretrained_model_path models/mixed_corpus_bert_large_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                          --train_path datasets/tnews/train.tsv --dev_path datasets/tnews/dev.tsv \
                          --output_model_path models/tnews_classifier_model.bin \
                          --epochs_num 3 --batch_size 32 --encoder bert

python3 inference/run_classifier_infer.py --load_model_path models/tnews_classifier_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                                          --test_path datasets/tnews/test_nolabel.tsv \
                                          --prediction_path datasets/tnews/prediction.tsv --labels_num 15 \
                                          --seq_length 128 --encoder bert
```

### OCNLI
We firstly do multi-task learning. We select XNLI and CMNLI as auxiliary tasks:
```
python3 run_mt_classifier.py --pretrained_model_path models/mixed_corpus_bert_large_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                             --dataset_path_list datasets/ocnli/ datasets/cmnli/ datasets/xnli/ \
                             --output_model_path models/ocnli_multitask_classifier_model.bin \
                             --epochs_num 1 --batch_size 64 --encoder bert
```
Then we load *ocnli_multitask_classifier_model.bin* and fine-tune it on OCNLI:
```
python3 run_classifier.py --pretrained_model_path models/ocnli_multitask_classifier_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                          --train_path datasets/ocnli/train.tsv --dev_path datasets/ocnli/dev.tsv \
                          --output_model_path models/ocnli_classifier_model.bin \
                          --epochs_num 1 --batch_size 64 --encoder bert
```
Then we do inference with *ocnli_classifier_model.bin*:
```
python3 inference/run_classifier_infer.py --load_model_path models/ocnli_classifier_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                                          --test_path datasets/ocnli/test_nolabel.tsv \
                                          --prediction_path datasets/ocnli/prediction.tsv --labels_num 3 \
                                          --seq_length 128 --encoder bert
```