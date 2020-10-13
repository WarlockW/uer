With the help of UER, we pre-trained models with different corpora, encoders, and targets. All pre-trained models can be loaded by UER directly. More pre-trained models will be released in the near future. Unless otherwise noted, Chinese pre-trained models use *models/google_zh_vocab.txt* as vocabulary, which is used in original BERT project. *models/bert_base_config.json* is used as configuration file in default. Commonly-used vocabulary and configuration files are included in *models* folder and users do not need to download them.

Pre-trained Chinese models from Google (in UER format):
<table>
<tr align="center"><th> Pre-trained model <th> Link <th> Description 
<tr align="center"><td> Wikizh+BertEncoder+BertTarget <td> https://share.weiyun.com/A1C49VPb <td> Google's pre-trained Chinese model from https://github.com/google-research/bert
<tr align="center"><td> CLUECorpus+<br>AlbertEncoder(base)+AlbertTarget <td> https://share.weiyun.com/UnKHNKRG <td> Google's pre-trained Chinese model from https://github.com/google-research/albert . <br>The configuration file is albert_base_config.json
<tr align="center"><td> CLUECorpus+<br>AlbertEncoder(large)+AlbertTarget <td> https://share.weiyun.com/9tTUwALd <td> Google's pre-trained Chinese model from https://github.com/google-research/albert . <br>The configuration file is albert_large_config.json
<tr align="center"><td> CLUECorpus+<br>AlbertEncoder(xlarge)+AlbertTarget <td> https://share.weiyun.com/mUamRQFR <td> Google's pre-trained Chinese model from https://github.com/google-research/albert . <br>The configuration file is albert_xlarge_config.json
<tr align="center"><td> CLUECorpus+<br>AlbertEncoder(xxlarge)+AlbertTarget <td> https://share.weiyun.com/0i2lX62b <td> Google's pre-trained Chinese model from https://github.com/google-research/albert . <br>The configuration file is albert_xxlarge_config.json
</table>

Models pre-trained by UER:
<table>
<tr align="center"><th> Pre-trained model <th> Link <th> Description 
<tr align="center"><td> Wikizh(word-based)+BertEncoder+BertTarget <td> Model: https://share.weiyun.com/5s4HVMi Vocab: https://share.weiyun.com/5NWYbYn <td> Word-based BERT model pre-trained on Wikizh. Training steps: 500,000
<tr align="center"><td> RenMinRiBao+BertEncoder+BertTarget <td> https://share.weiyun.com/5JWVjSE <td> The training corpus is news data from People's Daily (1946-2017).
<tr align="center"><td> Webqa2019+BertEncoder+BertTarget <td> https://share.weiyun.com/5HYbmBh <td> The training corpus is WebQA, which is suitable for datasets related with social media, e.g. LCQMC and XNLI. Training steps: 500,000
<tr align="center"><td> Weibo+BertEncoder+BertTarget <td> https://share.weiyun.com/5ZDZi4A <td> The training corpus is Weibo.
<tr align="center"><td> Weibo+BertEncoder(large)+MlmTarget <td> https://share.weiyun.com/CFKyMkp3 <td> The training corpus is Weibo. The configuration file is bert_large_config.json
<tr align="center"><td> Reviews+BertEncoder+MlmTarget <td> https://share.weiyun.com/tBgaSx77 <td> The training corpus is reviews.
<tr align="center"><td> Reviews+BertEncoder(large)+MlmTarget <td> https://share.weiyun.com/hn7kp9bs <td> The training corpus is reviews. The configuration file is bert_large_config.json
<tr align="center"><td> MixedCorpus+BertEncoder(large)+BertTarget <td> https://share.weiyun.com/5G90sMJ <td> Pre-trained on mixed large Chinese corpus. The configuration file is bert_large_config.json
<tr align="center"><td> MixedCorpus+BertEncoder(base)+BertTarget <td> https://share.weiyun.com/5QOzPqq <td> Pre-trained on mixed large Chinese corpus. The configuration file is bert_base_config.json
<tr align="center"><td> MixedCorpus+BertEncoder(small)+BertTarget <td> https://share.weiyun.com/fhcUanfy <td> Pre-trained on mixed large Chinese corpus. The configuration file is bert_small_config.json
<tr align="center"><td> MixedCorpus+BertEncoder(tiny)+BertTarget <td> https://share.weiyun.com/yXx0lfUg <td> Pre-trained on mixed large Chinese corpus. The configuration file is bert_tiny_config.json
<tr align="center"><td> MixedCorpus+GptEncoder+LmTarget <td> https://share.weiyun.com/51nTP8V <td> Pre-trained on mixed large Chinese corpus. Training steps: 500,000 (with sequence lenght of 128) + 100,000 (with sequence length of 512)
<tr align="center"><td> Reviews+LstmEncoder+LmTarget <td> https://share.weiyun.com/57dZhqo  <td> The training corpus is amazon reviews + JDbinary reviews + dainping reviews (11.4M reviews in total). Language model target is used. It is suitable for datasets related with reviews. It achieves over 5 percent improvements on some review datasets compared with random initialization. Set hidden_size in models/rnn_config.json to 512 before using it. Training steps: 200,000; Sequence length: 128;
<tr align="center"><td> (MixedCorpus & Amazon reviews)+LstmEncoder+(LmTarget & ClsTarget) <td> https://share.weiyun.com/5B671Ik  <td> Firstly pre-trained on mixed large Chinese corpus with LM target. And then is pre-trained on Amazon reviews with lm target and cls target. It is suitable for datasets related with reviews. It can achieve comparable results with BERT on some review datasets. Training steps: 500,000 + 100,000; Sequence length: 128
<tr align="center"><td> IfengNews+BertEncoder+BertTarget <td> https://share.weiyun.com/5HVcUWO <td> The training corpus is news data from Ifeng website. We use news title to predict news abstract. Training steps: 100,000; Sequence length: 128
<tr align="center"><td> jdbinary+BertEncoder+ClsTarget <td> https://share.weiyun.com/596k2bu <td> The training corpus is review data from JD (jingdong). CLS target is used for pre-training. It is suitable for datasets related with shopping reviews. Training steps: 50,000; Sequence length: 128
<tr align="center"><td> jdfull+BertEncoder+MlmTarget <td> https://share.weiyun.com/5L6EkUF <td> The training corpus is review data from JD (jingdong). MLM target is used for pre-training. Training steps: 50,000; Sequence length: 128
<tr align="center"><td> Amazonreview+BertEncoder+ClsTarget <td> https://share.weiyun.com/5XuxtFA <td> The training corpus is review data from Amazon (including book reviews, movie reviews, and etc.). Classification target is used for pre-training. It is suitable for datasets related with reviews, e.g. accuracy is improved on Douban book review datasets from 87.6 to 88.5 (compared with Google BERT). Training steps: 20,000; Sequence length: 128
<tr align="center"><td> XNLI+BertEncoder+ClsTarget <td> https://share.weiyun.com/5oXPugA <td> Infersent with BertEncoder
</table>
MixedCorpus contains baidubaike, Wikizh, WebQA, RenMinRiBao, literature, and reviews.

<br/>