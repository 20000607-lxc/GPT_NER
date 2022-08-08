## English Named Entity Recognition using GPT2 


### dataset list

1. conll2003_bio: datasets/conll2003_bio
   **note**: it's the BIO version of conll2003 dataset. collected from https://github.com/Alibaba-NLP/CLNER

2. ontonote: datasets/ontonotes
   
3. other datasets:
   1. cner: datasets/cner
   2. CLUENER: datasets/cluener
   **note**: the official link is  https://github.com/CLUEbenchmark/CLUENER 
   3. conll2003: datasets/conll2003
   **note**: it's the BIESO version of conll2003 dataset, use it to test if this model is adaptable to other labeling style. 
      

**note**: please write another DataProcessor in `ner_seq.py`  to use new datasets.
### model list

1. GPT2 + continuous prompt.
2. GPT2 + discrete prompt (human construct questions).

### transformers

1.I use transformers 4.6.0  which is in `models.transformers_master `


### requirement

1. PyTorch == 1.7.0
2. cuda=9.0
3. python3.6+
4. transformers >= 4.6.0

### input format

Input format (prefer BIOS tag scheme), with each character its label for one line. Sentences are splited with a null line.
The cner dataset labels are transferred into BIOS scheme in the DataProcessor.
```text
美	B-LOC
国	I-LOC
的	O
华	B-PER
莱	I-PER
士	I-PER

我	O
跟	O
他	O
```
### promot format
if template = (m,n,0), then the sequence fed into gpt2 is prompt1(length=m) + input + prompt2(length=n)

### run the code

1. Modify the configuration information in `run_ner.py`
2. Modify the prompt template by setting `TEMPLATE_CLASSES` in `run_ner.py` .


