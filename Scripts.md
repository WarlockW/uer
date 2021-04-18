UER-py provides abundant tool scripts for pre-training models. This section firstly summarizes tool scripts and their functions, and then provides using examples of some scripts.
|           Script         |   Function description   |
| :----------------------- | :----------------------- |
|      average_model.py    |  Take the average of pre-trained models  |
|      build_vocab.py      |  Build vocabulary given corpus and tokenizer  |
|      cloze_test.py       |  Randomly mask a word and predict it, top n words are returned  |
|      convert_bert_extractive_qa_from_huggingface_to_uer.py      |  Convert extractive QA BERT of Huggingface format (PyTorch) to UER format  |
|      convert_bert_extractive_qa_from_uer_to_huggingface.py      |  Convert extractive QA BERT of UER format to Huggingface format (PyTorch)  |
|      convert_bert_from_google_to_uer.py                         |  Convert BERT of Google format (TF) to UER format  |
|      convert_bert_from_huggingface_to_uer.py                    |  Convert BERT of Huggingface format (PyTorch) to UER format  |
|      convert_bert_from_uer_to_google.py                         |  Convert BERT of UER format to Google format (TF)  |
|      convert_bert_from_uer_to_huggingface.py                    |  Convert BERT of UER format to Huggingface format (PyTorch)  |
|      convert_bert_text_classification_from_huggingface_to_uer.py|  Convert text classification BERT of Huggingface format (PyTorch) to UER format  |
|      convert_bert_text_classification_from_uer_to_huggingface.py|  Convert text classification BERT of UER format to Huggingface format (PyTorch)  |
|      convert_gpt2_from_huggingface_to_uer.py                    |  Convert GPT-2 of Huggingface format (PyTorch) to UER format  |
|      convert_gpt2_from_uer_to_huggingface.py                    |  Convert GPT-2 of UER format to Huggingface format (PyTorch)  |
|      convert_t5_from_huggingface_to_uer.py                      |  Convert T5 of Huggingface format (PyTorch) to UER format  |
|      convert_t5_from_uer_to_huggingface.py                      |  Convert T5 of UER format to Huggingface format (PyTorch)  |
|      diff_vocab.py           |  Compare two vocabularies  |
|      dynamic_vocab_adapter.py|  Adapt the pre-trained model according to the vocabulary  |
|      extract_embeddings.py   |  Extract the embedding of the pre-trained model  |
|      extract_features.py     |  Obtain text representation  |
|      generate_lm.py          |  Generate text with language model  |
|      generate_seq2seq.py     |  Generate text with seq2seq model  |
|      run_bayesopt.py         |  Search hyper-parameters for LightGBM by bayesian optimization  |
|      run_lgb.py              |  Model ensemble with LightGBM (classification)  |
|      topn_words_dep.py       |  Find nearest neighbors with context-dependent word embedding  |
|      topn_words_indep.py     |  Find nearest neighbors with context-independent word embedding  |


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
*--whitening_size 64* indicates that the whitening operation is used and the dimension of the text embedding is *64*.

#### Embedding extractor
*extract_embeddings.py* extracts embedding layer from the pre-trained model. The extracted context-independent embedding can be used to initialize other models' (e.g. CNN) embedding layer. The example of using *extract_embeddings.py*:
```
python3 scripts/extract_embeddings.py --load_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                                      --word_embedding_path embeddings.txt
```
*--word_embedding_path* specifies the path of the output word embedding file. The format of word embedding file follows [here](https://github.com/Embedding/Chinese-Word-Vectors), which can be loaded directly by mainstream projects.

#### Finding nearest neighbours
The pre-trained model contains word embeddings. Traditional word embeddings such as word2vec and GloVe assign each word a fixed vector (context-independent word embedding). However, polysemy is a pervasive phenomenon in human language, and the meanings of a polysemous word depend on the context. To this end, we use the hidden state in pre-trained model to represent a word. It is noticeable that most Chinese pre-trained models are based on character. To obtain real word embedding (not character embedding), users can download [word-based BERT model](https://share.weiyun.com/5s4HVMi) and its [vocabulary](https://share.weiyun.com/5NWYbYn). The example of using *scripts/topn_words_indep.py* to find nearest neighbours for context-independent word embedding (character-based and word-based models):
```
python3 scripts/topn_words_indep.py --load_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                                    --test_path target_words.txt

python3 scripts/topn_words_indep.py --load_model_path models/wiki_bert_word_model.bin --vocab_path models/wiki_word_vocab.txt \
                                    --test_path target_words.txt
```
Context-independent word embedding comes from embedding layer. The format of the *target_words.txt* is as follows: 
```
word-1
word-2
...
word-n
```
The example of using *scripts/topn_words_dep.py* to find nearest neighbours for context-dependent word embedding (character-based and word-based models):
```
python3 scripts/topn_words_dep.py --load_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                                  --cand_vocab_path models/google_zh_vocab.txt --test_path target_words_with_sentences.txt --config_path models/bert/base_config.json \
                                  --batch_size 256 --seq_length 32 --tokenizer bert

python3 scripts/topn_words_dep.py --load_model_path models/bert_wiki_word_model.bin --vocab_path models/wiki_word_vocab.txt \
                                  --cand_vocab_path models/wiki_word_vocab.txt --test_path target_words_with_sentences.txt --config_path models/bert/base_config.json \
                                  --batch_size 256 --seq_length 32 --tokenizer space
```
We substitute the target word with other words in the vocabulary and feed the sentences into the pre-trained model. Hidden state is used as the context-dependent embedding of a word. <br>
*--cand_vocab_path* specifies the path of candidate word file. For faster speed one can use a smaller candidate vocabulary. <br> 
Users should do word segmentation manually and use space tokenizer if word-based model is used. The format of *target_words_with_sentences.txt* is as follows:
```
word1 sent1
word2 sent2 
...
wordn sentn
```
Sentence and word are split by \t.

#### Model average
*average_models.py* takes the average of multiple weights for probably more robust performance. The example of using *average_models.py*：
```
python3 scripts/average_models.py --model_list_path models/book_review_model.bin-4000 models/book_review_model.bin-5000 \
                                  --output_model_path models/book_review_model.bin
```

#### Text generator (language model)
We could use *generate_lm.py* to generate text through language model. Given a few words, *generate_lm.py* can continue writing. The example of using *generate_lm.py* to load GPT-2 weight and continue writing:
```
python3 scripts/generate_lm.py --load_model_path models/gpt_model.bin --vocab_path models/google_zh_vocab.txt \
                               --test_path beginning.txt --prediction_path generated_text.txt \
                               --config_path models/gpt2/distil_config.json --seq_length 128 \
                               --embedding word_pos --remove_embedding_layernorm \
                               --encoder transformer --mask causal --layernorm_positioning pre \
                               --target lm --tie_weight
```
where *beginning.txt* contains the beginning of a text and *generated_text.txt* contains the text that the model writes.
