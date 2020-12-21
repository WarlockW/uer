UER-py provides abundant tool scripts for pre-training models.
This section firstly summarizes tool scripts and their functions, and then provides using examples of some scripts.

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
<tr align="center"><td> topn_words_indep.py <td> Finding nearest neighbours with context-independent word embedding
<tr align="center"><td> topn_words_dep.py <td> Finding nearest neighbours with context-dependent word embedding
</table>


#### Cloze test
cloze_test.py predicts masked words. Top n words are returned.
```
usage: cloze_test.py [-h] [--pretrained_model_path PRETRAINED_MODEL_PATH]
                [--vocab_path VOCAB_PATH] [--input_path INPUT_PATH]
                [--output_path OUTPUT_PATH] [--config_path CONFIG_PATH]
                [--batch_size BATCH_SIZE] [--seq_length SEQ_LENGTH]
                [--encoder {bert,lstm,gru,cnn,gatedcnn,attn,rcnn,crnn,gpt}]
                [--bidirectional] [--target {bert,lm,cls,mlm,nsp,s2s}]
                [--subword_type {none,char}] [--sub_vocab_path SUB_VOCAB_PATH]
                [--subencoder_type {avg,lstm,gru,cnn}]
                [--tokenizer {bert,char,word,space}] [--topn TOPN]
```
The example of using cloze_test.py：
```
python3 scripts/cloze_test.py --input_path datasets/cloze_input.txt --pretrained_model_path models/google_zh_model.bin \
                              --vocab_path models/google_zh_vocab.txt --output_path output.txt

```

#### Feature extractor
extract_features.py extracts hidden states of the last encoder layer.
```
usage: extract_features.py [-h] --input_path INPUT_PATH --pretrained_model_path
                         PRETRAINED_MODEL_PATH --vocab_path VOCAB_PATH
                         --output_path OUTPUT_PATH [--seq_length SEQ_LENGTH]
                         [--spm_model_path SPM_MODEL_PATH]
                         [--batch_size BATCH_SIZE]
                         [--config_path CONFIG_PATH]
                         [--embedding {bert,word}]
                         [--encoder {bert,lstm,gru,cnn,gatedcnn,attn,rcnn,crnn,gpt,gpt2}]
                         [--bidirectional] [--parameter_sharing] 
                         [--factorized_embedding_parameterization]
                         [--tie_weights]
                         [--tokenizer {bert,char,space}]
```
The example of using extract_features.py：
```
python3 scripts/extract_features.py --input_path datasets/cloze_input.txt --vocab_path models/google_zh_vocab.txt \
                                   --pretrained_model_path models/google_zh_model.bin --output_path feature_output.pt
```

#### Finding nearest neighbours
Pre-trained models can learn high-quality word embeddings. Traditional word embeddings such as word2vec and GloVe assign each word a fixed vector (context-independent word embedding). However, polysemy is a pervasive phenomenon in human language, and the meanings of a polysemous word depend on the context. To this end, we use a the hidden state in pre-trained models to represent a word. It is noticeable that Google BERT is a character-based model. To obtain real word embedding (not character embedding), Users should download our [word-based BERT model](https://share.weiyun.com/5s4HVMi) and [vocabulary](https://share.weiyun.com/5NWYbYn).
The example of using scripts/topn_words_indep.py to find nearest neighbours for context-independent word embedding (character-based and word-based models)：
```
python3 scripts/topn_words_indep.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                                    --cand_vocab_path models/google_zh_vocab.txt --target_words_path target_words.txt
python3 scripts/topn_words_indep.py --pretrained_model_path models/bert_wiki_word_model.bin --vocab_path models/wiki_word_vocab.txt \
                                    --cand_vocab_path models/wiki_word_vocab.txt --target_words_path target_words.txt
```
Context-independent word embedding is obtained by model's embedding layer.
The format of the target_words.txt is as follows:
```
word-1
word-2
...
word-n
```
The example of using scripts/topn_words_dep.py to find nearest neighbours for context-dependent word embedding (character-based and word-based models)：
```
python3 scripts/topn_words_dep.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                                  --cand_vocab_path models/google_zh_vocab.txt --sent_path target_words_with_sentences.txt --config_path models/bert_base_config.json \
                                  --batch_size 256 --seq_length 32 --tokenizer bert
python3 scripts/topn_words_dep.py --pretrained_model_path models/bert_wiki_word_model.bin --vocab_path models/wiki_word_vocab.txt \
                                  --cand_vocab_path models/wiki_word_vocab.txt --sent_path target_words_with_sentences.txt --config_path models/bert_base_config.json \
                                  --batch_size 256 --seq_length 32 --tokenizer space
```
We substitute the target word with other words in the vocabulary and feed the sentences into the pretrained model. Hidden state is used as the context-dependent embedding of a word. Users should do word segmentation manually and use space tokenizer if word-based model is used. The format of 
target_words_with_sentences.txt is as follows:
```
sent1 word1
sent1 word1
...
sentn wordn
```
Sentence and word are splitted by \t. 

#### Text generator
We could use *generate.py* to generate text. Given a few words or sentences, *generate.py* can continue writing. The example of using *generate.py*:
```
python3 scripts/generate.py --pretrained_model_path models/gpt_model.bin --vocab_path models/google_zh_vocab.txt 
                            --input_path story_beginning.txt --output_path story_full.txt --config_path models/bert_base_config.json 
                            --encoder gpt --target lm --seq_length 128  
```
where *story_beginning* contains the beginning of a text. One can use any models pre-trained with LM target, such as [GPT trained on mixed large corpus](https://share.weiyun.com/51nTP8V). By now we only provide a vanilla version of generator. More mechanisms will be added for better performance and efficiency.
