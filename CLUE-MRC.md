Here is a short summary of our solution on [CLUE MRC benchmark](https://cluebenchmarks.com/rc.html).

### C3
The example of fine-tuning and doing inference on C3 dataset with [*mixed_corpus_bert_large_model.bin*](https://share.weiyun.com/5G90sMJ):
```
python3 run_c3.py --pretrained_model_path models/mixed_corpus_bert_large_model.bin --vocab_path models/google_zh_vocab.txt \
                  --config_path models/bert_large_config.json \
                  --train_path datasets/c3/train.json --dev_path datasets/c3/dev.json \
                  --epochs_num 5 --batch_size 8 --seq_length 512 --max_choices_num 4 \
                  --encoder bert --learning_rate 1e-5

python3 inference/run_c3_infer.py --load_model_path models/multichoice_model.bin --vocab_path models/google_zh_vocab.txt \
                                  --config_path models/bert_large_config.json --test_path datasets/c3/test.json \
                                  --prediction_path datasets/c3/prediction.json \
                                  --max_choices_num 4 --encoder bert
```

