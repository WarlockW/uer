## Requirements
* Python 3.6
* torch >= 1.1
* six >= 1.12.0
* argparse
* packaging
* For the mixed precision training you will need apex from NVIDIA
* For the pre-trained model conversion (related with TensorFlow) you will need TensorFlow
* For the tokenization with sentencepiece model you will need SentencePiece
* For developing a stacking model you will need LightGBM and [BayesianOptimization](https://github.com/fmfn/BayesianOptimization)