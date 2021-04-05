以下是我们[CLUE分类任务](https://www.cluebenchmarks.com/classification.html)解决方案的简要介绍。 我们提交了两个结果， *UER* 和 *UER-ensemble* ， *UER* 的结果基于[ *mixed_corpus_bert_large_model.bin* ](https://share.weiyun.com/5G90sMJ) 预训练权重； *UER-ensemble* 的结果基于大量模型的集成。本节主要关注单模型。关于模型集成的更多详细信息，请参见[这里](https://github.com/dbiir/UER-py/wiki/SMP2020-EWECT)。

### AFQMC
首先做多任务学习，选择LCQMC和XNLI作为辅助任务：
```
python3 run_classifier_mt.py --pretrained_model_path models/mixed_corpus_bert_large_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                             --dataset_path_list datasets/afqmc/ datasets/lcqmc/ datasets/xnli/ \
                             --output_model_path models/afqmc_multitask_classifier_model.bin \
                             --epochs_num 1 --batch_size 64 --embedding word_pos_seg --encoder transformer --mask fully_visible
```
之后加载 *afqmc_multitask_classifier_model.bin* 在AFQMC上微调：
```
python3 run_classifier.py --pretrained_model_path models/afqmc_multitask_classifier_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                          --train_path datasets/afqmc/train.tsv --dev_path datasets/afqmc/dev.tsv \
                          --output_model_path models/afqmc_classifier_model.bin \
                          --epochs_num 3 --batch_size 32 --embedding word_pos_seg --encoder transformer --mask fully_visible
```
最后用 *afqmc_classifier_model.bin* 做预测：
```
python3 inference/run_classifier_infer.py --load_model_path models/afqmc_classifier_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                                          --test_path datasets/afqmc/test_nolabel.tsv \
                                          --prediction_path datasets/afqmc/prediction.tsv --labels_num 2 \
                                          --seq_length 128 --embedding word_pos_seg --encoder transformer --mask fully_visible
```

### CMNLI
首先做多任务学习，选择XNLI作为辅助任务：
```
python3 run_classifier_mt.py --pretrained_model_path models/mixed_corpus_bert_large_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                             --dataset_path_list datasets/cmnli/ datasets/xnli/ \
                             --output_model_path models/cmnli_multitask_classifier_model.bin \
                             --epochs_num 1 --batch_size 64 --embedding word_pos_seg --encoder transformer --mask fully_visible
```
之后加载 *cmnli_multitask_classifier_model.bin* 在CMNLI上微调：
```
python3 run_classifier.py --pretrained_model_path models/cmnli_multitask_classifier_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                          --train_path datasets/cmnli/train.tsv --dev_path datasets/cmnli/dev.tsv \
                          --output_model_path models/cmnli_classifier_model.bin \
                          --epochs_num 1 --batch_size 64 --embedding word_pos_seg --encoder transformer --mask fully_visible
```
最后用 *cmnli_classifier_model.bin* 做预测:
```
python3 inference/run_classifier_infer.py --load_model_path models/cmnli_classifier_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                                          --test_path datasets/cmnli/test_nolabel.tsv \
                                          --prediction_path datasets/cmnli/prediction.tsv --labels_num 3 \
                                          --seq_length 128 --embedding word_pos_seg --encoder transformer --mask fully_visible
```

### IFLYTEK
在IFLYTEK数据集上做微调和预测示例：
```
python3 run_classifier.py --pretrained_model_path models/mixed_corpus_bert_large_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                          --train_path datasets/iflytek/train.tsv --dev_path datasets/iflytek/dev.tsv \
                          --output_model_path models/iflytek_classifier_model.bin \
                          --epochs_num 3 --batch_size 32 --seq_length 256 --embedding word_pos_seg --encoder transformer --mask fully_visible

python3 inference/run_classifier_infer.py --load_model_path models/iflytek_classifier_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                                          --test_path datasets/iflytek/test_nolabel.tsv \
                                          --prediction_path datasets/iflytek/prediction.tsv --labels_num 119 \
                                          --seq_length 256 --embedding word_pos_seg --encoder transformer --mask fully_visible
```

### CSL
中国科学文献任务判断给定的关键词是否是论文的真实关键词。在CSL上取得好结果的关键是使用特殊符号来分割关键词。我们发现CSL中的伪造的关键词通常很短，而特殊符号可以明确告知模型关键词的长度。
在CSL数据集上做微调和预测的示例：
```
python3 run_classifier.py --pretrained_model_path models/mixed_corpus_bert_large_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                          --train_path datasets/csl/train.tsv --dev_path datasets/csl/dev.tsv \
                          --output_model_path models/csl_classifier_model.bin \
                          --epochs_num 3 --batch_size 32 --seq_length 384 --embedding word_pos_seg --encoder transformer --mask fully_visible

python3 inference/run_classifier_infer.py --load_model_path models/csl_classifier_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                                          --test_path datasets/csl/test_nolabel.tsv \
                                          --prediction_path datasets/csl/prediction.tsv --labels_num 2 \
                                          --seq_length 384 --embedding word_pos_seg --encoder transformer --mask fully_visible
```

### CLUEWSC2020：
在CLUEWSC2020数据集上做微调和预测示例：
```
python3 run_classifier.py --pretrained_model_path models/mixed_corpus_bert_large_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                          --train_path datasets/cluewsc2020/train.tsv --dev_path datasets/cluewsc2020/dev.tsv \
                          --output_model_path models/cluewsc2020_classifier_model.bin \
                          --epochs_num 20 --batch_size 8 --learning_rate 5e-6 --embedding word_pos_seg --encoder transformer --mask fully_visible

python3 inference/run_classifier_infer.py --load_model_path models/cluewsc2020_classifier_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                                          --test_path datasets/cluewsc2020/test_nolabel.tsv \
                                          --prediction_path datasets/cluewsc2020/prediction.tsv --labels_num 119 \
                                          --seq_length 128 --embedding word_pos_seg --encoder transformer --mask fully_visible
```
CLUEWSC2020的一个技巧是使用WSC的训练集（CLUEWSC2020的旧版本）作为训练样本。

### TNEWS
在TNEWS数据集上做微调和预测示例：
```
python3 run_classifier.py --pretrained_model_path models/mixed_corpus_bert_large_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                          --train_path datasets/tnews/train.tsv --dev_path datasets/tnews/dev.tsv \
                          --output_model_path models/tnews_classifier_model.bin \
                          --epochs_num 3 --batch_size 32 --embedding word_pos_seg --encoder transformer --mask fully_visible

python3 inference/run_classifier_infer.py --load_model_path models/tnews_classifier_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                                          --test_path datasets/tnews/test_nolabel.tsv \
                                          --prediction_path datasets/tnews/prediction.tsv --labels_num 15 \
                                          --seq_length 128 --embedding word_pos_seg --encoder transformer --mask fully_visible
```

### OCNLI
首先做多任务学习，选择XNLI和CMNLI作为辅助任务：
```
python3 run_classifier_mt.py --pretrained_model_path models/mixed_corpus_bert_large_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                             --dataset_path_list datasets/ocnli/ datasets/cmnli/ datasets/xnli/ \
                             --output_model_path models/ocnli_multitask_classifier_model.bin \
                             --epochs_num 1 --batch_size 64 --embedding word_pos_seg --encoder transformer --mask fully_visible
```
之后加载 *ocnli_multitask_classifier_model.bin* 在OCNLI上微调：
```
python3 run_classifier.py --pretrained_model_path models/ocnli_multitask_classifier_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                          --train_path datasets/ocnli/train.tsv --dev_path datasets/ocnli/dev.tsv \
                          --output_model_path models/ocnli_classifier_model.bin \
                          --epochs_num 1 --batch_size 64 --embedding word_pos_seg --encoder transformer --mask fully_visible
```
最后用 *ocnli_classifier_model.bin* 做预测:
```
python3 inference/run_classifier_infer.py --load_model_path models/ocnli_classifier_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/bert_large_config.json \
                                          --test_path datasets/ocnli/test_nolabel.tsv \
                                          --prediction_path datasets/ocnli/prediction.tsv --labels_num 3 \
                                          --seq_length 128 --embedding word_pos_seg --encoder transformer --mask fully_visible
```