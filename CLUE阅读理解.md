以下是[CLUE阅读理解](https://cluebenchmarks.com/rc.html)解决方案的简要介绍。

### CMRC2018
利用[*mixed_corpus_bert_large_model.bin*](https://share.weiyun.com/5G90sMJ)在CMRC2018数据集上做微调和预测示例：
```
python3 run_cmrc.py --pretrained_model_path models/mixed_corpus_bert_large_model.bin --vocab_path models/google_zh_vocab.txt \
                    --config_path models/bert/large_config.json \
                    --train_path datasets/cmrc2018/train.json --dev_path datasets/cmrc2018/dev.json \
                    --output_model_path models/cmrc_model.bin \
                    --epochs_num 2 --batch_size 8 --seq_length 512 --embedding word_pos_seg --encoder transformer --mask fully_visible

python3 inference/run_cmrc_infer.py --load_model_path models/cmrc_model.bin --vocab_path models/google_zh_vocab.txt \
                                    --config_path models/bert/large_config.json --test_path datasets/cmrc2018/test.json \
                                    --prediction_path datasets/cmrc2018/prediction.json \
                                    --seq_length 512 \
                                    --embedding word_pos_seg --encoder transformer --mask fully_visible
```

### ChID
利用[*mixed_corpus_bert_large_model.bin*](https://share.weiyun.com/5G90sMJ)在ChID数据集上做微调和预测示例：
```
python3 run_chid.py --pretrained_model_path models/mixed_corpus_bert_large_model.bin --vocab_path models/google_zh_vocab.txt \
                    --config_path models/bert/large_config.json \
                    --train_path datasets/chid/train.json --train_answer_path datasets/chid/train_answer.json \
                    --dev_path datasets/chid/dev.json --dev_answer_path datasets/chid/dev_answer.json \
                    --output_model_path models/multichoice_model.bin \
                    --batch_size 16 --seq_length 64 --max_choices_num 10 --epochs_num 3 --report_steps 1000 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible 

python3 inference/run_chid_infer.py --load_model_path models/multichoice_model.bin --vocab_path models/google_zh_vocab.txt \
                                    --config_path models/bert/large_config.json --test_path datasets/chid/test.json \
                                    --prediction_path datasets/chid/prediction.json \
                                    --seq_length 64 --max_choices_num 10 \
                                    --embedding word_pos_seg --encoder transformer --mask fully_visible
```
注意到需要在推理阶段使用函数 *postprocess_chid_predictions* 对预测结果进行后处理。这能显著提升模型在ChID数据集上的表现。

### C3
利用[*mixed_corpus_bert_large_model.bin*](https://share.weiyun.com/5G90sMJ)在C3数据集上做微调和预测示例：
```
python3 run_c3.py --pretrained_model_path models/mixed_corpus_bert_large_model.bin --vocab_path models/google_zh_vocab.txt \
                  --config_path models/bert/large_config.json \
                  --train_path datasets/c3/train.json --dev_path datasets/c3/dev.json \
                  --output_model_path models/multichoice_model.bin \
                  --epochs_num 5 --batch_size 8 --seq_length 512 --max_choices_num 4 \
                  --learning_rate 1e-5 --embedding word_pos_seg --encoder transformer --mask fully_visible

python3 inference/run_c3_infer.py --load_model_path models/multichoice_model.bin --vocab_path models/google_zh_vocab.txt \
                                  --config_path models/bert/large_config.json --test_path datasets/c3/test.json \
                                  --prediction_path datasets/c3/prediction.json \
                                  --seq_length 512 --max_choices_num 4 \
                                  --embedding word_pos_seg --encoder transformer --mask fully_visible
```
