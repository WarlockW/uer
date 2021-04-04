## Features
UER-py has the following features:
- __Reproducibility.__ UER-py has been tested on many datasets and should match the performances of the original pre-training model implementations such as BERT, GPT, ELMo, and T5.
- __Multi-GPU.__ UER-py supports CPU mode, single GPU mode, and distributed training mode. 
- __Model modularity.__ UER-py is divided into multiple components: embedding, encoder, target, and downstream task fine-tuning. Ample modules are implemented in each component. Clear and robust interface allows users to combine modules with as few restrictions as possible.
- __Efficiency.__ UER-py refines its pre-processing, pre-training, and fine-tuning stages, which largely improves speed and needs less memory.
- __Model zoo.__ With the help of UER-py, we pre-trained models with different corpora, encoders, and targets. Proper selection of pre-trained models is important to the downstream task performances.
- __SOTA results.__ UER-py supports comprehensive downstream tasks (e.g. classification and machine reading comprehension) and has been used in winning solutions of many NLP competitions.