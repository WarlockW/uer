```
usage: preprocess.py [-h] --corpus_path CORPUS_PATH [--vocab_path VOCAB_PATH]
                     [--spm_model_path SPM_MODEL_PATH]
                     [--tgt_vocab_path TGT_VOCAB_PATH]
                     [--tgt_spm_model_path TGT_SPM_MODEL_PATH]
                     [--dataset_path DATASET_PATH]
                     [--tokenizer {bert,char,space}]
                     [--tgt_tokenizer {bert,char,space}]
                     [--processes_num PROCESSES_NUM]
                     [--target {bert,lm,mlm,bilm,albert,seq2seq,t5,cls,prefixlm}]
                     [--docs_buffer_size DOCS_BUFFER_SIZE]
                     [--seq_length SEQ_LENGTH]
                     [--tgt_seq_length TGT_SEQ_LENGTH]
                     [--dup_factor DUP_FACTOR]
                     [--short_seq_prob SHORT_SEQ_PROB] [--full_sentences]
                     [--seed SEED] [--dynamic_masking] [--whole_word_masking]
                     [--span_masking] [--span_geo_prob SPAN_GEO_PROB]
                     [--span_max_length SPAN_MAX_LENGTH]
```
Users have to preprocess the corpus before pre-training. <br> 
The example of pre-processing on a single machine：
```
python3 preprocess.py --corpus_path corpora/book_review_bert.txt --vocab_path models/google_zh_vocab.txt --dataset_path dataset.pt \
                      --processes_num 8 --target bert
```
If multiple machines are available, users can run *preprocess.py* on one machine and copy the *dataset.pt* to other machines. <br>
The output of pre-processing stage is *dataset.pt* (*--dataset_path*), which is the input of *pretrain.py* .


We need to specify model's target (*--target*) in pre-processing stage since different targets require different data formats in pre-training stage. Currently, UER-py consists of the following target modules:
- lm: language model
- mlm: masked language model (in cloze test style)
- cls: classification
- bilm: bi-directional language model
- bert: masked language model + next sentence prediction
- albert: masked language model + sentence order prediction
- t5：masked language model (in sequence to sequence style)
- prefixlm：prefix language model
- seq2seq：sequence to sequence

Notice that we should use the corpus (*--corpus_path*) whose format is in accordance with the target. More examples are found in [Pretrain models with different encoders and targets](https://github.com/dbiir/UER-py/wiki/Pretrain-models-with-different-encoders-and-targets).

*--processes_num n* denotes that n processes are used for pre-processing. More processes can speed up the preprocess stage but lead to more memory consumption. <br>
*--dynamic_masking* denotes that the words are masked during the pre-training stage, which is used in RoBERTa. <br>
*--full_sentences* allows a sample to include contents from multiple documents, which is used in RoBERTa. <br>
*--span_masking* denotes that masking consecutive words, which is used in SpanBERT. If dynamic masking is used, we should specify *--span_masking* in pre-training stage, otherwise we should specify *--span_masking* in pre-processing stage. <br>
*--docs_buffer_size* specifies the buffer size in memory in pre-processing stage. <br>
Sequence length is specified in pre-processing stage by *--seq_length* . The default value is 128. <br>

Vocabulary and tokenizer are also specified in pre-processing stage. More details are discussed in [Tokenization and vocabulary](https://github.com/dbiir/UER-py/wiki/Tokenization-and-vocabulary) section.
