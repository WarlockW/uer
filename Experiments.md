## Experiments
### Speed
```
GPU：Tesla P40

CPU：Intel(R) Xeon(R) CPU E5-2699 v4 @ 2.20GHz

```
We use BERT to test the speed of distributed training mode. Google BERT is trained for 1 million steps and each step contains 128,000 tokens. It takes around 18 days to reproduce the experiments by UER-py on 3 GPU machines (24 GPU in total).

<table>
<tr align="center"><th> #(machine) <th> #(GPU)/machine <th> tokens/second
<tr align="center"><td> 1 <td> 0 <td> 276
<tr align="center"><td> 1 <td> 1 <td> 7050
<tr align="center"><td> 1 <td> 2 <td> 13071
<tr align="center"><td> 1 <td> 4 <td> 24695
<tr align="center"><td> 1 <td> 8 <td> 44300
<tr align="center"><td> 3 <td> 8 <td> 84386
</table>
 
### Qualitative evaluation
We qualitatively evaluate pre-trained models by finding words' near neighbours.

#### Character-based model
Evaluation of context-independent word embedding:

<table>
<tr align="center"><td> Target word: 苹 <td> <td> Target word: 吃 <td>  <td> Target word: 水 <td> 
<tr align="center"><td> 蘋 <td> 0.762  <td> 喝 <td> 0.539 <td> 河 <td> 0.286 
<tr align="center"><td> apple <td> 0.447  <td> 食 <td> 0.475 <td> 海 <td> 0.278
<tr align="center"><td> iphone <td> 0.400 <td> 啃 <td> 0.340 <td> water <td> 0.276
<tr align="center"><td> 柠 <td> 0.347  <td> 煮 <td> 0.324 <td> 油 <td> 0.266
<tr align="center"><td> ios <td> 0.317  <td> 嚐 <td> 0.322 <td> 雨 <td> 0.259
</table>


Evaluation of context-dependent word embedding:

Target sentence: 其冲积而形成小平原沙土层厚而肥沃，盛产苹果、大樱桃、梨和葡萄。
<table>
<tr align="center"><td> Target word: 苹 <td> 
<tr align="center"><td> 蘋 <td> 0.822
<tr align="center"><td> 莓 <td> 0.714 
<tr align="center"><td> 芒 <td> 0.706
<tr align="center"><td> 柠 <td> 0.704
<tr align="center"><td> 樱 <td> 0.696 
</table>


Target sentence: 苹果削减了台式Mac产品线上的众多产品。
<table>
<tr align="center"><td> Target word: 苹 <td> 
<tr align="center"><td> 蘋 <td> 0.892
<tr align="center"><td> apple <td> 0.788 
<tr align="center"><td> iphone <td> 0.743
<tr align="center"><td> ios <td> 0.720
<tr align="center"><td> ipad <td> 0.706 
</table>

#### Word-based model
Evaluation of context-independent word embedding:

<table>
<tr align="center"><td> Target word: 苹果 <td> <td> Target word: 腾讯 <td>  <td>  Target word: 吉利 <td> 
<tr align="center"><td> 苹果公司 <td> 0.419  <td> 新浪 <td> 0.357 <td> 沃尔沃 <td> 0.277
<tr align="center"><td> apple <td> 0.415  <td> 网易 <td> 0.356 <td> 伊利 <td> 0.243
<tr align="center"><td> 苹果电脑 <td> 0.349 <td> 搜狐 <td> 0.356 <td> 长荣 <td> 0.235
<tr align="center"><td> 微软 <td> 0.320  <td> 百度 <td> 0.341 <td> 天安 <td> 0.224
<tr align="center"><td> mac <td> 0.298  <td> 乐视 <td> 0.332 <td> 哈达 <td> 0.220
</table>


Evaluation of context-dependent word embedding:

Target sentence: 其冲积而形成小平原沙土层厚而肥沃，盛产苹果、大樱桃、梨和葡萄。
<table>
<tr align="center"><td> Target word: 苹果 <td> 
<tr align="center"><td> 柠檬 <td> 0.734
<tr align="center"><td> 草莓 <td> 0.725 
<tr align="center"><td> 荔枝 <td> 0.719
<tr align="center"><td> 树林 <td> 0.697
<tr align="center"><td> 牡丹 <td> 0.686 
</table>

 
Target sentence: 苹果削减了台式Mac产品线上的众多产品
<table>
<tr align="center"><td> Target word: 苹果 <td> 
<tr align="center"><td> 苹果公司 <td> 0.836
<tr align="center"><td> apple <td> 0.829
<tr align="center"><td> 福特 <td> 0.796
<tr align="center"><td> 微软 <td> 0.777
<tr align="center"><td> 苹果电脑 <td> 0.773 
</table>

 
Target sentence: 讨吉利是通过做民间习俗的吉祥事，或重现过去曾经得到好结果的行为，以求得好兆头。
<table>
<tr align="center"><td> Target word: 吉利 <td> 
<tr align="center"><td> 仁德 <td> 0.749
<tr align="center"><td> 光彩 <td> 0.743
<tr align="center"><td> 愉快 <td> 0.736
<tr align="center"><td> 永元 <td> 0.736
<tr align="center"><td> 仁和 <td> 0.732 
</table>

 
Target sentence: 2010年6月2日福特汽车公司宣布出售旗下高端汽车沃尔沃予中国浙江省的吉利汽车，同时将于2010年第四季停止旗下中阶房车品牌所有业务
<table>
<tr align="center"><td> Target word: 吉利 <td> 
<tr align="center"><td> 沃尔沃 <td> 0.771
<tr align="center"><td> 卡比 <td> 0.751
<tr align="center"><td> 永利 <td> 0.745
<tr align="center"><td> 天安 <td> 0.741
<tr align="center"><td> 仁和 <td> 0.741 
</table>

 
Target sentence: 主要演员有扎克·布拉夫、萨拉·朝克、唐纳德·费森、尼尔·弗林、肯·詹金斯、约翰·麦吉利、朱迪·雷耶斯、迈克尔·莫斯利等。
<table>
<tr align="center"><td> Target word: 吉利 <td> 
<tr align="center"><td> 玛利 <td> 0.791
<tr align="center"><td> 米格 <td> 0.768
<tr align="center"><td> 韦利 <td> 0.767
<tr align="center"><td> 马力 <td> 0.764
<tr align="center"><td> 安吉 <td> 0.761 
</table>

### Quantitative evaluation
We use a range of Chinese datasets to evaluate the performance of UER-py. Douban book review, ChnSentiCorp, Shopping, and Tencentnews are sentence-level small-scale sentiment classification datasets. MSRA-NER is a sequence labeling dataset. These datasets are included in this project. Dianping, JDfull, JDbinary, Ifeng, and Chinanews are large-scale classification datasets. They are collected in [glyph](https://arxiv.org/pdf/1708.02657.pdf) and can be downloaded at [glyph's github project](https://github.com/zhangxiangxiao/glyph). These five datasets don't contain validation set. We use 10% instances in trainset for validation.

Most pre-training models consist of 2 stages: pre-training on general-domain corpus and fine-tuning on downstream dataset. We recommend 3-stage mode: 1)Pre-training on general-domain corpus; 2)Pre-training on downstream dataset; 3)Fine-tuning on downstream dataset. Stage 2 enables models to get familiar with distributions of downstream tasks. It is sometimes known as semi-supervised fune-tuning.

Hyper-parameter settings are as follows:
- Stage 1: We train with batch size of 256 sequences and each sequence contains 256 tokens. We load Google's pretrained models and train upon it for 500,000 steps. The learning rate is 2e-5 and other optimizer settings are identical with Google BERT. BERT tokenizer is used.
- Stage 2: We train with batch size of 256 sequences. For classification datasets, the sequence length is 128. For sequence labeling datasets, the sequence length is 256. We train upon Google's pretrained model for 20,000 steps. Optimizer settings and tokenizer are identical with stage 1.
- Stage 3: For classification datasets, the training batch size and epochs are 64 and 3. For sequence labeling datasets, the training batch size and epochs are 32 and 5. Optimizer settings and tokenizer are identical with stage 1.

We provide the pre-trained models (using BERT target) on different downstream datasets: [book_review_model.bin](https://share.weiyun.com/52BEFs2); [chnsenticorp_model.bin](https://share.weiyun.com/53WDBeJ); [shopping_model.bin](https://share.weiyun.com/5HaxwAf); [msra_model.bin](https://share.weiyun.com/5E6XpEt). Tencentnews dataset and its pretrained model will be publicly available after data desensitization. 

<table>
<tr align="center"><th> Model/Dataset              <th> Douban book review <th> ChnSentiCorp <th> Shopping <th> MSRA-NER <th> Tencentnews review
<tr align="center"><td> BERT                       <td> 87.5               <td> 94.3         <td> 96.3     <td> 93.0/92.4/92.7  <td> 84.2
<tr align="center"><td> BERT+semi_BertTarget       <td> 88.1               <td> 95.6         <td> 97.0     <td> 94.3/92.6/93.4  <td> 85.1
<tr align="center"><td> BERT+semi_MlmTarget        <td> 87.9               <td> 95.5         <td> 97.1     <td>   <td> 85.1
</table>

Pre-training is also important for other encoders and targets. We pre-train a 2-layer LSTM on 1.9G review corpus with language model target. Embedding size and hidden size are 512. The model is much more efficient than BERT in pre-training and fine-tuning stages. We show that pre-training brings significant improvements and achieves competitive results (the differences are not big compared with the results of BERT).

<table>
<tr align="center"><th> Model/Dataset              <th> Douban book review <th> ChnSentiCorp <th> Shopping 
<tr align="center"><td> BERT                       <td> 87.5               <td> 94.3         <td> 96.3     
<tr align="center"><td> LSTM                       <td> 80.2               <td> 88.3         <td> 94.4     
<tr align="center"><td> LSTM+pre-training          <td> 86.6(+6.4)         <td> 94.5(+6.2)   <td> 96.5(+2.1)     
</table>


It requires tremendous computional resources to fine-tune on large-scale datasets. For Ifeng, Chinanews, Dianping, JDbinary, and JDfull datasets, we provide their classification models (see Chinese model zoo). Classification models on large-scale datasets allow users to reproduce the results without training. Besides that, classification models could be used for improving other related tasks. More experimental results will come soon. 

Ifeng and Chinanews datasets contain news' titles and abstracts. In stage 2, we use title to predict abstract. 

<table>
<tr align="center"><th> Model/Dataset              <th> Ifeng     <th> Chinanews <th> Dianping <th> JDbinary <th> JDfull
<tr align="center"><td> pre-SOTA (Glyph & Glyce)   <td> 85.76     <td> 91.88     <td> 78.46    <td> 91.76    <td> 54.24 
<tr align="center"><td> BERT                       <td> 87.50     <td> 93.37     <td>          <td> 92.37    <td> 54.79
<tr align="center"><td> BERT+semi+BertTarget       <td> 87.65     <td>           <td>          <td>          <td>
</table>

We also provide the pre-trained models on different corpora, encoders, and targets (see Chinese model zoo). Selecting proper pre-training models is beneficial to the performance of downstream tasks.

<table>
<tr align="center"><th> Model/Dataset                     <th> MSRA-NER
<tr align="center"><td> Wikizh corpus (Google)            <td> 93.0/92.4/92.7
<tr align="center"><td> Renminribao corpus                <td> 94.4/94.4/94.4
</table>

<br/>