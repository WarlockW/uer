Here is an example of doing stacking on SMP2019-ECISA dataset. More information about ensemble can be found [here](https://github.com/dbiir/UER-py/wiki/SMP2020-EWECT). One can download SMP2019-ECISA dataset from Datasets section. The example of fine-tuning (K-fold cross validation) to obtain classifiers and features on train set:
```
CUDA_VISIBLE_DEVICES=0,1 python3 finetune/run_classifier_cv.py --pretrained_model_path models/google_zh_model.bin \
                                                               --vocab_path models/google_zh_vocab.txt \
                                                               --output_model_path models/ecisa_classifier_model_0.bin \
                                                               --config_path models/bert/base_config.json \
                                                               --train_path datasets/smp2019-ecisa/train.tsv \
                                                               --train_features_path datasets/smp2019-ecisa/train_features_0.npy \
                                                               --folds_num 5 --epochs_num 3 --batch_size 32 \
                                                               --embedding word_pos_seg --encoder transformer --mask fully_visible

CUDA_VISIBLE_DEVICES=0,1 python3 finetune/run_classifier_cv.py --pretrained_model_path models/review_roberta_large_model.bin \
                                                               --vocab_path models/google_zh_vocab.txt \
                                                               --output_model_path models/ecisa_classifier_model_1.bin \
                                                               --config_path models/bert/large_config.json \
                                                               --train_path datasets/smp2019-ecisa/train.tsv \
                                                               --train_features_path datasets/smp2019-ecisa/train_features_1.npy \
                                                               --folds_num 5 --epochs_num 3 --batch_size 32 \
                                                               --embedding word_pos_seg --encoder transformer --mask fully_visible

CUDA_VISIBLE_DEVICES=0 python3 finetune/run_classifier_cv.py --pretrained_model_path models/cluecorpussmall_roberta_base_seq512_model.bin-250000 \
                                                             --vocab_path models/google_zh_vocab.txt \
                                                             --output_model_path models/ecisa_classifier_model_2.bin \
                                                             --config_path models/bert/base_config.json \
                                                             --train_path datasets/smp2019-ecisa/train.tsv \
                                                             --train_features_path datasets/smp2019-ecisa/train_features_2.npy \
                                                             --folds_num 5 --epochs_num 3 --batch_size 32 --seq_length 160 \
                                                             --embedding word_pos_seg --encoder transformer --mask fully_visible

CUDA_VISIBLE_DEVICES=0 python3 finetune/run_classifier_cv.py --pned_model_path models/cluecorpussmall_gpt2_seq1024_model.bin-250000 \
                                                             --vocab_path models/google_zh_vocab.txt \
                                                             --output_model_path models/ecisa_classifier_model_3.bin \
                                                             --config_path models/gpt2/config.json \
                                                             --train_path datasets/smp2019-ecisa/train.tsv \
                                                             --train_features_path datasets/smp2019-ecisa/train_features_3.npy \
                                                             --folds_num 5 --epochs_num 3 --batch_size 32 --seq_length 100 \
                                                             --embedding word_pos --remove_embedding_layernorm \
                                                             --encoder transformer --mask causal --layernorm_position pre --pooling mean

CUDA_VISIBLE_DEVICES=0,1 python3 finetune/run_classifier_cv.py --pretrained_model_path models/mixed_corpus_bert_large_model.bin \
                                                               --vocab_path models/google_zh_vocab.txt \
                                                               --output_model_path models/ecisa_classifier_model_4.bin \
                                                               --config_path models/bert/large_config.json \
                                                               --train_path datasets/smp2019-ecisa/train.tsv \
                                                               --train_features_path datasets/smp2019-ecisa/train_features_4.npy \
                                                               --folds_num 5 --epochs_num 3 --batch_size 32 \
                                                               --embedding word_pos_seg --encoder transformer --mask fully_visible

CUDA_VISIBLE_DEVICES=0,1 python3 finetune/run_classifier_cv.py --pretrained_model_path models/chinese_roberta_wwm_large_ext_pytorch.bin \
                                                               --vocab_path models/google_zh_vocab.txt \
                                                               --output_model_path models/ecisa_classifier_model_5.bin \
                                                               --config_path models/bert/large_config.json \
                                                               --train_path datasets/smp2019-ecisa/train.tsv \
                                                               --train_features_path datasets/smp2019-ecisa/train_features_5.npy \
                                                               --folds_num 5 --epochs_num 3 --batch_size 32 \
                                                               --embedding word_pos_seg --encoder transformer --mask fully_visible
```

The example of doing inference (K-fold cross validation) to obtain features on development set:
```
CUDA_VISIBLE_DEVICES=0 python3 inference/run_classifier_infer_cv.py --load_model_path models/ecisa_classifier_model_0.bin \
                                                                    --vocab_path models/google_zh_vocab.txt \
                                                                    --config_path models/bert/base_config.json \
                                                                    --test_path datasets/smp2019-ecisa/dev.tsv \
                                                                    --test_features_path datasets/smp2019-ecisa/test_features_0.npy \
                                                                    --folds_num 5 --labels_num 3 \
                                                                    --embedding word_pos_seg --encoder transformer --mask fully_visible

CUDA_VISIBLE_DEVICES=0 python3 inference/run_classifier_infer_cv.py --load_model_path models/ecisa_classifier_model_1.bin \
                                                                    --vocab_path models/google_zh_vocab.txt \
                                                                    --config_path models/bert/large_config.json \
                                                                    --test_path datasets/smp2019-ecisa/dev.tsv \
                                                                    --test_features_path datasets/smp2019-ecisa/test_features_1.npy \
                                                                    --folds_num 5 --labels_num 3 \
                                                                    --embedding word_pos_seg --encoder transformer --mask fully_visible

CUDA_VISIBLE_DEVICES=0 python3 inference/run_classifier_infer_cv.py --load_model_path models/ecisa_classifier_model_2.bin \
                                                                    --vocab_path models/google_zh_vocab.txt \
                                                                    --config_path models/bert/base_config.json \
                                                                    --test_path datasets/smp2019-ecisa/dev.tsv \
                                                                    --test_features_path datasets/smp2019-ecisa/test_features_2.npy \
                                                                    --folds_num 5 --labels_num 3 \
                                                                    --embedding word_pos_seg --encoder transformer --mask fully_visible

CUDA_VISIBLE_DEVICES=0 python3 inference/run_classifier_infer_cv.py --load_model_path models/ecisa_classifier_model_3.bin \
                                                                    --vocab_path models/google_zh_vocab.txt \
                                                                    --config_path models/gpt2/config.json \
                                                                    --test_path datasets/smp2019-ecisa/dev.tsv \
                                                                    --test_features_path datasets/smp2019-ecisa/test_features_3.npy \
                                                                    --folds_num 5 --labels_num 3 --seq_length 100 \
                                                                    --embedding word_pos --remove_embedding_layernorm \
                                                                    --encoder transformer --mask causal --layernorm_position pre --pooling mean

CUDA_VISIBLE_DEVICES=0 python3 inference/run_classifier_infer_cv.py --load_model_path models/ecisa_classifier_model_4.bin \
                                                                    --vocab_path models/google_zh_vocab.txt \
                                                                    --config_path models/bert/large_config.json \
                                                                    --test_path datasets/smp2019-ecisa/dev.tsv \
                                                                    --test_features_path datasets/smp2019-ecisa/test_features_4.npy \
                                                                    --folds_num 5 --labels_num 3 \
                                                                    --embedding word_pos_seg --encoder transformer --mask fully_visible

CUDA_VISIBLE_DEVICES=0 python3 inference/run_classifier_infer_cv.py --load_model_path models/ecisa_classifier_model_5.bin \
                                                                    --vocab_path models/google_zh_vocab.txt \
                                                                    --config_path models/bert/large_config.json \
                                                                    --test_path datasets/smp2019-ecisa/dev.tsv \
                                                                    --test_features_path datasets/smp2019-ecisa/test_features_5.npy \
                                                                    --folds_num 5 --labels_num 3 \
                                                                    --embedding word_pos_seg --encoder transformer --mask fully_visible
```

The example of searching hyper-parameters for LightGBM:
```
python3 scripts/run_lgb_cv_bayesopt.py --train_path datasets/smp2019-ecisa/train.tsv \
                                       --train_features_path datasets/smp2019-ecisa/ \
                                       --models_num 6 --folds_num 5 --labels_num 3 --epochs_num 100
```

The example of using LightGBM to train and validate:
```
python3 scripts/run_lgb.py --train_path datasets/smp2019-ecisa/train.tsv --test_path datasets/smp2019-ecisa/dev.tsv \
                           --train_features_path datasets/smp2019-ecisa/ --test_features_path datasets/smp2019-ecisa/ \
                           --models_num 6 --labels_num 3
```

One could find more details on [competition's homepage](https://www.biendata.xyz/competition/smpecisa2019/).
