以下是[CLUE命名实体识别](https://www.cluebenchmarks.com/ner.html)解决方案的简要介绍。

### CLUENER2020
利用[*google_zh_model.bin*](https://share.weiyun.com/A1C49VPb)在CLUENER2020数据集上做微调和预测示例：
```
python3 run_ner.py --pretrained_model_path models/google_zh_model.bin \
                   --vocab_path models/google_zh_vocab.txt \
                   --train_path datasets/cluener2020/train.tsv \
                   --dev_path datasets/cluener2020/dev.tsv \
                   --label2id_path datasets/cluener2020/label2id.json \
                   --output_model_path models/ner_model.bin \
                   --epochs_num 5 --batch_size 16 \
                   --embedding word_pos_seg --encoder transformer --mask fully_visible

python3 inference/run_ner_infer.py --load_model_path models/ner_model.bin \
                                   --vocab_path models/google_zh_vocab.txt \
                                   --test_path datasets/cluener2020/test_nolabel.tsv \
                                   --prediction_path datasets/cluener2020/prediction.tsv \
                                   --label2id_path datasets/cluener2020/label2id.json \
                                   --embedding word_pos_seg --encoder transformer --mask fully_visible
```

利用[*mixed_corpus_bert_large_model.bin*](https://share.weiyun.com/5G90sMJ)在CLUENER2020数据集上做微调和预测示例：
```
python3 run_ner.py --pretrained_model_path models/mixed_corpus_bert_large_model.bin \
                   --vocab_path models/google_zh_vocab.txt \
                   --config_path models/bert/large_config.json \
                   --train_path datasets/cluener2020/train.tsv \
                   --dev_path datasets/cluener2020/dev.tsv \
                   --output_model_path models/ner_model.bin \
                   --label2id_path datasets/cluener2020/label2id.json \
                   --epochs_num 5 --batch_size 16 \
                   --embedding word_pos_seg --encoder transformer --mask fully_visible

python3 inference/run_ner_infer.py --load_model_path models/ner_model.bin \
                                   --vocab_path models/google_zh_vocab.txt \
                                   --config_path models/bert/large_config.json \
                                   --test_path datasets/cluener2020/test_nolabel.tsv \
                                   --prediction_path datasets/cluener2020/prediction.tsv \
                                   --label2id_path datasets/cluener2020/label2id.json \
                                   --embedding word_pos_seg --encoder transformer --mask fully_visible
```