Here is a short summary of our solution on [CLUE MRC benchmark](https://cluebenchmarks.com/rc.html).

### CMRC2018
The example of fine-tuning and doing inference on CMRC2018 dataset with [*mixed_corpus_bert_large_model.bin*](https://share.weiyun.com/5G90sMJ):
```
python3 run_cmrc.py --pretrained_model_path models/mixed_corpus_bert_large_model.bin --vocab_path models/google_zh_vocab.txt \
                    --config_path models/bert_large_config.json \
                    --train_path datasets/cmrc2018/train.json --dev_path datasets/cmrc2018/dev.json \
                    --epochs_num 2 --batch_size 8 --seq_length 512 --encoder bert

python3  inference/run_cmrc_infer.py --load_model_path models/cmrc_model.bin --vocab_path models/google_zh_vocab.txt \
                                     --config_path models/bert_large_config.json --test_path datasets/cmrc2018/test.json \
                                     --prediction_path datasets/cmrc2018/prediction.json --encoder bert
```

### ChID
The example of fine-tuning and doing inference on ChID dataset with [*mixed_corpus_bert_large_model.bin*](https://share.weiyun.com/5G90sMJ):
```
python3 run_chid.py --pretrained_model_path models/mixed_corpus_bert_large_model.bin --vocab_path models/google_zh_vocab.txt \
                    --config_path models/bert_large_config.json \                    
                    --train_path datasets/chid/train.json --train_answer_path datasets/chid/train_answer.json \
                    --dev_path datasets/chid/dev.json --dev_answer_path datasets/chid/dev_answer.json \
                    --batch_size 32 --seq_length 64 --max_choices_num 10 --epochs_num 8 --report_steps 1000 --encoder bert 

python3 inference/run_chid_infer.py --load_model_path models/multichoice_model.bin --vocab_path models/google_zh_vocab.txt \
                                    --config_path models/bert_large_config.json --test_path datasets/chid/test.json \
                                    --prediction_path datasets/chid/prediction.json \
                                    --seq_length 64 --max_choices_num 10 --encoder bert
```

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