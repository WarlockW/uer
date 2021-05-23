SMP2020-EWECT consists of two tasks, usual task and virus task. We take usual task as an example to demonstrate the use of stacking. We randomly select some pre-trained models and fine-tune upon them. K-fold cross validation is performed to make predictions for the training dataset. The features (predicted probabilities) are generated for stacking. *--train_features_path* specifies the path of generated features. One can obtain the pre-trained models used below from [Modelzoo](https://github.com/dbiir/UER-py/wiki/Modelzoo) section:
```
CUDA_VISIBLE_DEVICES=0,1 python3 finetune/run_classifier_cv.py --pretrained_model_path models/review_roberta_large_model.bin \
                                                               --vocab_path models/google_zh_vocab.txt \
                                                               --output_model_path models/ewect_usual_classifier_model_0.bin \
                                                               --config_path models/bert/large_config.json \
                                                               --train_path datasets/smp2020-ewect/usual/train.tsv \
                                                               --train_features_path datasets/smp2020-ewect/usual/train_features_0.npy \
                                                               --folds_num 5 --epochs_num 3 --batch_size 64 --seed 17 \
                                                               --embedding word_pos_seg --encoder transformer --mask fully_visible

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 finetune/run_classifier_cv.py --pretrained_model_path models/mixed_corpus_bert_large_model.bin \
                                                                   --vocab_path models/google_zh_vocab.txt \
                                                                   --output_model_path models/ewect_usual_classifier_model_1.bin \
                                                                   --config_path models/bert/large_config.json \
                                                                   --train_path datasets/smp2020-ewect/usual/train.tsv \
                                                                   --train_features_path datasets/smp2020-ewect/usual/train_features_1.npy \
                                                                   --folds_num 5 --epochs_num 3 --batch_size 64 --seq_length 160 \
                                                                   --embedding word_pos_seg --encoder transformer --mask fully_visible

CUDA_VISIBLE_DEVICES=0 python3 finetune/run_classifier_cv.py --pretrained_model_path models/cluecorpussmall_gpt2_seq1024_model.bin-250000 \
                                                             --vocab_path models/google_zh_vocab.txt \
                                                             --output_model_path models/ewect_usual_classifier_model_2.bin \
                                                             --config_path models/gpt2/config.json \
                                                             --train_path datasets/smp2020-ewect/usual/train.tsv \
                                                             --train_features_path datasets/smp2020-ewect/usual/train_features_2.npy \
                                                             --folds_num 5 --epochs_num 3 --batch_size 32 --seq_length 100 \
                                                             --embedding word_pos --remove_embedding_layernorm \
                                                             --encoder transformer --mask causal --layernorm_position pre --pooling mean

CUDA_VISIBLE_DEVICES=0 python3 finetune/run_classifier_cv.py --pretrained_model_path models/cluecorpussmall_elmo_model.bin-500000 \
                                                             --vocab_path models/google_zh_vocab.txt \
                                                             --config_path models/birnn_config.json \
                                                             --output_model_path models/ewect_usual_classifier_model_3.bin \
                                                             --train_path datasets/smp2020-ewect/usual/train.tsv \
                                                             --train_features_path datasets/smp2020-ewect/usual/train_features_3.npy \
                                                             --epochs_num 3 --batch_size 64 --learning_rate 5e-4 --folds_num 5 \
                                                             --embedding word --remove_embedding_layernorm --encoder bilstm --pooling max

CUDA_VISIBLE_DEVICES=0 python3 finetune/run_classifier_cv.py --pretrained_model_path models/cluecorpussmall_gatedcnn_lm_model.bin-500000 \
                                                             --vocab_path models/google_zh_vocab.txt \
                                                             --config_path models/gatedcnn_9_config.json \
                                                             --output_model_path models/ewect_usual_classifier_model_4.bin \
                                                             --train_path datasets/smp2020-ewect/usual/train.tsv \
                                                             --train_features_path datasets/smp2020-ewect/usual/train_features_4.npy \
                                                             --epochs_num 3 --batch_size 64 --learning_rate 5e-5 --folds_num 5 \
                                                             --embedding word --remove_embedding_layernorm --encoder gatedcnn --pooling mean
```
*--output_model_path* specifies the path of fine-tuned classifier models. K classifier models are obtained in K-fold cross validation. These classifier models are then used for inference on development set to obtain features (*--test_features_path*) :
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
                                                                    --config_path models/gpt2/config.json \
                                                                    --test_path datasets/smp2020-ewect/usual/dev.tsv \
                                                                    --test_features_path datasets/smp2020-ewect/usual/test_features_2.npy \
                                                                    --folds_num 5 --labels_num 6 --seq_length 100 \
                                                                    --embedding word_pos --remove_embedding_layernorm \
                                                                    --encoder transformer --mask causal --layernorm_position pre --pooling mean

CUDA_VISIBLE_DEVICES=0 python3 inference/run_classifier_infer_cv.py --load_model_path models/ewect_usual_classifier_model_3.bin \
                                                                    --vocab_path models/google_zh_vocab.txt \
                                                                    --config_path models/birnn_config.json \
                                                                    --test_path datasets/smp2020-ewect/usual/dev.tsv \
                                                                    --test_features_path datasets/smp2020-ewect/usual/test_features_3.npy \
                                                                     --folds_num 5 --labels_num 6 \
                                                                    --embedding word --remove_embedding_layernorm --encoder bilstm --pooling max

CUDA_VISIBLE_DEVICES=0 python3 inference/run_classifier_infer_cv.py --load_model_path models/ewect_usual_classifier_model_4.bin \
                                                                    --vocab_path models/google_zh_vocab.txt \
                                                                    --config_path models/gatedcnn_9_config.json \
                                                                    --test_path datasets/smp2020-ewect/usual/dev.tsv \
                                                                    --test_features_path datasets/smp2020-ewect/usual/test_features_4.npy \
                                                                    --folds_num 5 --labels_num 6 \
                                                                    --embedding word --remove_embedding_layernorm --encoder gatedcnn --pooling mean
```

We then use the LightGBM to handle the features of train set and development set. To find the proper hyper-parameters for LightGBM, we exploit bayesian optimization and cross validation is used for evaluation:
```
python3 scripts/run_lgb_cv_bayesopt.py --train_path datasets/smp2020-ewect/usual/train.tsv \
                                       --train_features_path datasets/smp2020-ewect/usual/ \
                                       --models_num 5 --folds_num 5 --labels_num 6 --epochs_num 100
```
It is usually beneficial to do ensemble on multiple models (*--models_num*).

When hyper-parameters are determined, we train LightGBM and evaluate it on development set:
```
python3 scripts/run_lgb.py --train_path datasets/smp2020-ewect/usual/train.tsv \
                           --test_path datasets/smp2020-ewect/usual/dev.tsv \
                           --train_features_path datasets/smp2020-ewect/usual/ \
                           --test_features_path datasets/smp2020-ewect/usual/ \
                           --models_num 5 --labels_num 6
```
One can change the hyper-parameters in *scripts/run_lgb.py* . *--train_path* and *--test_path* provides the labels of train set and development set. *--train_features_path* and *--test_features_path* provides the features of train set and development set.

It is a simple demonstration of the using stacked model. Nevertheless, the above operations can bring us very competitive results. More improvement will be obtained when we continue to add models with different pre-processing, pre-training, and fine-tuning strategies. One could find more details on [competition's homepage](http://39.97.118.137/).
