UER-py has the following features:
- __Reproducibility__ UER-py has been tested on many datasets and should match the performances of the original pre-training model implementations such as BERT, GPT-2, ELMo, and T5.
- __Multi-GPU__ UER-py supports CPU mode, single GPU mode, and distributed training mode. 
- __Model modularity__ UER-py is divided into multiple components: embedding, encoder, and target. Ample modules are implemented in each component. Clear and robust interface allows users to combine modules to construct pre-training models with as few restrictions as possible.
- __Efficiency__ UER-py refines its pre-processing, pre-training, fine-tuning, and inference stages, which largely improves speed while requires less memory and disk space.
- __Model zoo__ With the help of UER-py, we pre-trained models with different corpora, encoders, and targets. Proper selection of pre-trained models is important to the performances of downstream tasks.
- __SOTA results__ UER-py supports comprehensive downstream tasks (e.g. classification and machine reading comprehension) and provides winning solutions of many NLP competitions.
- __Abundant functions__ UER-py provides abundant functions related with pre-training, such as feature extractor and mixed precision training.