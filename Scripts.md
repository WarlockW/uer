UER-py provides abundant tool scripts for pre-training models. This section firstly summarizes tool scripts and their functions, and then provides using examples of some scripts.
<table>
<tr align="center"><th> Script <th> Function description
<tr align="center"><td> average_model.py <td> Take the average of pre-trained models. A frequently-used ensemble strategy for deep learning models
<tr align="center"><td> build_vocab.py <td> Build vocabulary (multi-processing supported)
<tr align="center"><td> check_model.py <td> Check the model (single GPU or multiple GPUs)
<tr align="center"><td> cloze_test.py <td> Randomly mask a word and predict it, top n words are returned
<tr align="center"><td> convert_bert_from_uer_to_google.py <td> convert the BERT of UER format to Google format (TF)
<tr align="center"><td> convert_bert_from_uer_to_huggingface.py <td> convert the BERT of UER format to Huggingface format (PyTorch)
<tr align="center"><td> convert_bert_from_google_to_uer.py <td> convert the BERT of Google format (TF) to UER format
<tr align="center"><td> convert_bert_from_huggingface_to_uer.py <td> convert the BERT of Huggingface format (PyTorch) to UER format
<tr align="center"><td> diff_vocab.py <td> Compare two vocabularies
<tr align="center"><td> dynamic_vocab_adapter.py <td> Change the pre-trained model according to the vocabulary. It can save memory in fine-tuning stage since task-specific vocabulary is much smaller than general-domain vocabulary
<tr align="center"><td> extract_embeddings.py <td> extract the embedding of the pre-trained model
<tr align="center"><td> extract_features.py <td> extract the hidden states of the last of the pre-trained model
<tr align="center"><td> topn_words_dep.py <td> Finding nearest neighbours with context-independent word embedding
<tr align="center"><td> topn_words_indep.py <td> Finding nearest neighbours with context-dependent word embedding
</table>

#### Cloze test
*cloze_test.py* uses MLM target to predict masked word. Top n words are returned. Cloze test can be used for operations such as data augmentation. The example of using *cloze_test.py*:
```
python3 scripts/cloze_test.py --load_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                              --config_path models/bert/base_config.json \
                              --test_path datasets/tencent_profile.txt --prediction_path output.txt \
                              --target bert
```
Notice that *cloze_test.py* only supports bert，mlm，and albert targets.

#### Feature extractor
The text is encoded into a fixed-length embedding by *extract_features.py* (through embedding, encoder, and pooling layers). The example of using *extract_features.py*:
```
python3 scripts/extract_features.py --load_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                                    --config_path models/bert/base_config.json \
                                    --test_path datasets/tencent_profile.txt --prediction_path features.pt \
                                    --pooling first
```
CLS embedding (*--pooling first*) is commonly used as the text embedding. When cosine similarity is used to measure the relationship between two texts, CLS embedding is not a proper choice. According to recent work, it is necessary to perform whitening operation:
```
python3 scripts/extract_features.py --load_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                                    --config_path models/bert/base_config.json \
                                    --test_path datasets/tencent_profile.txt --prediction_path features.pt \
                                    --pooling first --whitening_size 64
```
*--whitening_size 64* 表明会使用白化操作，并且向量经过变化后，维度变为64。如果不指定 *--whitening_size* ，则不会使用白化操作。推荐在特征抽取过程中使用白化操作。

#### Embedding extractor
*extract_embeddings.py* 从预训练模型权重embedding层中抽取词向量。这里的词向量指传统的上下文无关词向量。抽取出的词向量可以用于初始化其他模型（比如CNN）词向量层初始化。*extract_embeddings.py* 使用示例：
```
python3 scripts/extract_embeddings.py --load_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                                      --word_embedding_path embeddings.txt
```
*--word_embedding_path* 指定输出词向量文件的路径。词向量文件的格式遵循[这里](https://github.com/Embedding/Chinese-Word-Vectors)，可以被主流项目直接使用。

#### Finding nearest neighbours
预训练模型能够产生高质量的词向量。传统的词向量（比如word2vec和GloVe）给定一个单词固定的向量（上下文无关向量）。然而，一词多义是人类语言中的常见现象。一个单词的意思依赖于其上下文。我们可以使用预训练模型的隐层去表示单词。值得注意的是大多数的中文预训练模型是基于字的。如果需要真正的词向量而不是字向量，用户需要下载[基于词的BERT模型](https://share.weiyun.com/5s4HVMi)和[词典](https://share.weiyun.com/5NWYbYn)。上下文无关词向量以词搜词 *scripts/topn_words_indep.py* 使用示例（基于字和基于词）：
```
python3 scripts/topn_words_indep.py --load_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                                    --test_path target_words.txt

python3 scripts/topn_words_indep.py --load_model_path models/wiki_bert_word_model.bin --vocab_path models/wiki_word_vocab.txt \
                                    --test_path target_words.txt
```
上下文无关词向量来自于模型的embedding层， *target_words.txt* 的格式如下所示：
```
word-1
word-2
...
word-n
```
下面给出上下文相关词向量以词搜词 *scripts/topn_words_dep.py* 使用示例（基于字和基于词）：
```
python3 scripts/topn_words_dep.py --load_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                                  --cand_vocab_path models/google_zh_vocab.txt --test_path target_words_with_sentences.txt --config_path models/bert/base_config.json \
                                  --batch_size 256 --seq_length 32 --tokenizer bert

python3 scripts/topn_words_dep.py --load_model_path models/bert_wiki_word_model.bin --vocab_path models/wiki_word_vocab.txt \
                                  --cand_vocab_path models/wiki_word_vocab.txt --test_path target_words_with_sentences.txt --config_path models/bert/base_config.json \
                                  --batch_size 256 --seq_length 32 --tokenizer space
```
我们把目标词替换成词典中其它的词（候选词），将序列送入网络。我们把目标词/候选词对应位置的隐层（最后一层）看作是目标词/候选词的上下文相关词向量。如果两个单词在特定上下文中的隐层向量接近，那么它们可能在特定的上下文中有相似的意思。<br>
*--cand_vocab_path* 指定候选词文件的路径。由于需要将目标词替换成所有的候选词，然后经过网络，因此我们可以选择较小的候选词词典。<br>
如果用户使用基于词的模型，需要对 *target_words_with_sentences.txt* 文件的句子进行分词。<br>
*target_words_with_sentences.txt* 文件的格式如下：
```
word1 sent1
word2 sent2 
...
wordn sentn
```
单词与句子之间使用\t分隔。

#### 模型平均
*average_models.py* 对多个有相同结构的预训练模型权重取平均。*average_models.py* 使用示例：
```
python3 scripts/average_models.py --model_list_path models/book_review_model.bin-4000 models/book_review_model.bin-5000 \
                                  --output_model_path models/book_review_model.bin
```
在预训练阶段，我们训练5000步，每隔1000步存储一次。我们对训练4000步和5000步的预训练模型进行平均。

#### Text generator (language model)
我们可以使用 *generate_lm.py* 来用语言模型生成文本。给定文本开头，模型根据开头续写。使用 *generate.py* 加载GPT-2模型并进行生成示例：
```
python3 scripts/generate_lm.py --load_model_path models/gpt_model.bin --vocab_path models/google_zh_vocab.txt \
                               --test_path beginning.txt --prediction_path generated_sentence.txt \
                               --config_path models/gpt2/distil_config.json --seq_length 128 \
                               --embedding word_pos --remove_embedding_layernorm \
                               --encoder transformer --mask causal --layernorm_positioning pre \
                               --target lm --tie_weight
```
*beginning.txt* 包含了文本的开头。*generated_sentence.txt* 包含了文本的开头以及模型续写的内容。*--load_model_path* 可以使用经过LM目标预先训练的模型。
