Here is a short summary of our solution on [CLUE NER benchmark](https://www.cluebenchmarks.com/ner.html).

### CLUENER2020
The example of fine-tuning and doing inference on CLUENER2020 dataset with [*google_zh_model.bin*](https://share.weiyun.com/A1C49VPb):
```
python3 finetune/run_ner.py --pretrained_model_path models/google_zh_model.bin \
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

The example of fine-tuning and doing inference on CLUENER2020 dataset with [*mixed_corpus_bert_large_model.bin*](https://share.weiyun.com/5G90sMJ):
```
python3 finetune/run_ner.py --pretrained_model_path models/mixed_corpus_bert_large_model.bin \
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