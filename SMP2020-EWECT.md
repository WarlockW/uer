### SMP2020-EWECT Usual task
We randomly select some pre-trained models and fine-tune upon them. K-fold cross validation is performed to make predictions for the training dataset. The features (predicted probabilities) are generated for stacking:
```
CUDA_VISIBLE_DEVICES=0,1 python3 run_classifier_cv.py --pretrained_model_path models/reviews_bert_large_model.bin \
                                                      --vocab_path models/google_zh_vocab.txt \
                                                      --output_model_path models/ewect_usual_classifier_model_0.bin \
                                                      --config_path models/bert/large_config.json \
                                                      --train_path datasets/smp2020-ewect/usual/train.tsv \
                                                      --train_features_path datasets/smp2020-ewect/usual/train_features_0.npy \
                                                      --folds_num 5 --epochs_num 3 --batch_size 64 --seed 17 \
                                                      --embedding word_pos_seg --encoder transformer --mask fully_visible

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run_classifier_cv.py --pretrained_model_path models/mixed_corpus_bert_large_model.bin \
                                                          --vocab_path models/google_zh_vocab.txt \
                                                          --output_model_path models/ewect_usual_classifier_model_1.bin \
                                                          --config_path models/bert/large_config.json \
                                                          --train_path datasets/smp2020-ewect/usual/train.tsv \
                                                          --train_features_path datasets/smp2020-ewect/usual/train_features_1.npy \
                                                          --folds_num 5 --epochs_num 3 --batch_size 64 --seq_length 160 \
                                                          --embedding word_pos_seg --encoder transformer --mask fully_visible

CUDA_VISIBLE_DEVICES=0 python3 run_classifier_cv.py --pretrained_model_path models/mixed_corpus_gpt_base_model.bin \
                                                    --vocab_path models/google_zh_vocab.txt \
                                                    --output_model_path models/ewect_usual_classifier_model_2.bin \
                                                    --config_path models/bert/base_config.json \
                                                    --train_path datasets/smp2020-ewect/usual/train.tsv \
                                                    --train_features_path datasets/smp2020-ewect/usual/train_features_2.npy \
                                                    --folds_num 5 --epochs_num 3 --batch_size 32 --seq_length 100 \
                                                    --embedding word_pos_seg --encoder transformer --mask causal --pooling mean

CUDA_VISIBLE_DEVICES=0 python3 run_classifier_cv.py --pretrained_model_path models/mixed_corpus_elmo_model.bin \
                                                    --vocab_path models/google_zh_vocab.txt \
                                                    --config_path models/birnn_config.json \
                                                    --output_model_path models/ewect_usual_classifier_model_3.bin \
                                                    --train_path datasets/smp2020-ewect/usual/train.tsv \
                                                    --train_features_path datasets/smp2020-ewect/virus/usual_features_3.npy \
                                                    --epochs_num 3 --batch_size 64 --learning_rate 5e-4 --folds_num 5 \
                                                    --embedding word --encoder bilstm --pooling mean

CUDA_VISIBLE_DEVICES=0 python3 run_classifier_cv.py --pretrained_model_path models/wikizh_gatedcnn_model.bin \
                                                    --vocab_path models/google_zh_vocab.txt \
                                                    --config_path models/gatedcnn_9_config.json \
                                                    --output_model_path models/ewect_usual_classifier_model_4.bin \
                                                    --train_path datasets/smp2020-ewect/usual/train.tsv \
                                                    --train_features_path datasets/smp2020-ewect/usual/train_features_4.npy \
                                                    --epochs_num 3 --batch_size 64 --learning_rate 5e-5 --folds_num 5 \
                                                    --embedding word --encoder gatedcnn --pooling max
```
We then use tree based model to handle the extracted features. To find the proper hyper-parameters for tree based model, we exploit bayesian optimization:
```
python3 scripts/run_bayesopt.py --train_path datasets/smp2020-ewect/usual/train.tsv \
                                --train_features_path datasets/smp2020-ewect/usual/ \
                                --models_num 5 --folds_num 5 --labels_num 6 --epochs_num 100
```

When hyper-parameters of the tree based model are determined, we extract features for the devlopment dataset and make predictions:
```
CUDA_VISIBLE_DEVICES=0 python3 inference/run_classifier_infer_cv.py --load_model_path models/ewect_usual_classifier_model_0.bin \
                                                                    --vocab_path models/google_zh_vocab.txt \
                                                                    --config_path models/bert/large_config.json \
                                                                    --test_path datasets/smp2020-ewect/usual/dev.tsv \
                                                                    --test_features_path datasets/smp2020-ewect/usual/test_features_0.npy \
                                                                    --folds_num 5 --labels_num 6 \
                                                                    --embedding word_pos_seg --encoder transformer --mask fully_visible

CUDA_VISIBLE_DEVICES=0 python3 inference/run_classifier_infer_cv.py --load_model_path models/ewect_usual_classifier_model_1.bin \
                                                                    --vocab_path models/google_zh_vocab.txt \
                                                                    --config_path models/bert/large_config.json \
                                                                    --test_path datasets/smp2020-ewect/usual/dev.tsv \
                                                                    --test_features_path datasets/smp2020-ewect/usual/test_features_1.npy \
                                                                    --folds_num 5 --labels_num 6 --seq_length 160 \
                                                                    --embedding word_pos_seg --encoder transformer --mask fully_visible

CUDA_VISIBLE_DEVICES=0 python3 inference/run_classifier_infer_cv.py --load_model_path models/ewect_usual_classifier_model_2.bin \
                                                                    --vocab_path models/google_zh_vocab.txt \
                                                                    --config_path models/bert/base_config.json \
                                                                    --test_path datasets/smp2020-ewect/usual/dev.tsv \
                                                                    --test_features_path datasets/smp2020-ewect/usual/test_features_2.npy \
                                                                    --folds_num 5 --labels_num 6 --seq_length 100 \
                                                                    --embedding word_pos_seg --encoder transformer --mask causal --pooling mean

CUDA_VISIBLE_DEVICES=0 python3 inference/run_classifier_infer_cv.py --load_model_path models/ewect_usual_classifier_model_3.bin \
                                                                    --vocab_path models/google_zh_vocab.txt \
                                                                    --config_path models/birnn_config.json \
                                                                    --test_path datasets/smp2020-ewect/usual/dev.tsv \
                                                                    --test_features_path datasets/smp2020-ewect/usual/test_features_3.npy \
                                                                     --folds_num 5 --labels_num 6 \
                                                                    --embedding word --encoder bilstm --pooling mean

CUDA_VISIBLE_DEVICES=0 python3 inference/run_classifier_infer_cv.py --load_model_path models/ewect_usual_classifier_model_4.bin \
                                                                    --vocab_path models/google_zh_vocab.txt \
                                                                    --config_path models/gatedcnn_9_config.json \
                                                                    --test_path datasets/smp2020-ewect/usual/dev.tsv \
                                                                    --test_features_path datasets/smp2020-ewect/usual/test_features_4.npy \
                                                                    --folds_num 5 --labels_num 6 \
                                                                    --embedding word --encoder gatedcnn --pooling max

python3 scripts/run_lgb.py --train_path datasets/smp2020-ewect/usual/train.tsv \
                           --test_path datasets/smp2020-ewect/usual/dev.tsv \
                           --train_features_path datasets/smp2020-ewect/usual/ \
                           --test_features_path datasets/smp2020-ewect/usual/ \
                           --models_num 5 --labels_num 6
```
Users can change hyper-parameters in *run_lgb.py* .

It is a simple demonstration of the using stacked model. Nevertheless, the above operations can bring us very competitive results (top 5). More improvement will be obtained when we continue to add models with different pre-processing, pre-training, and fine-tuning strategies. One could find more details on [competition's homepage](http://39.97.118.137/).
