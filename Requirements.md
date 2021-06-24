* Python 3.6
* torch >= 1.1
* six >= 1.12.0
* argparse
* packaging
* For the mixed precision training you will need apex from NVIDIA
* For the pre-trained model conversion (related with TensorFlow) you will need TensorFlow
* For the tokenization with sentencepiece model you will need [SentencePiece](https://github.com/google/sentencepiece)
* For developing a stacking model you will need LightGBM and [BayesianOptimization](https://github.com/fmfn/BayesianOptimization)
* For the pre-training with whole word masking you will need word segmentation tool such as [jieba](https://github.com/fxsjy/jieba)