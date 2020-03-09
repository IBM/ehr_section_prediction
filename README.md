
# EHR Section Prediction

This repository can be used to generate the data and run the experiments described in ["Leveraging Medical Literature for Section Prediction in Electronic Health Records"](https://www.aclweb.org/anthology/D19-1492/) Rosenthal et al. (EMNLP 2019) on Wikipedia and i2b2 data. Results on these experiments using Wikipedia only can be found in supplemental material. 

### Installation

**Prerequisites**

  * The following packages are needed:
    [python3](https://www.python.org/download/releases/3.0/),
    [pytorch](https://pytorch.org/), 
    [NLTK](https://anaconda.org/anaconda/nltk),
    [spacy](https://spacy.io/usage/),
    [sklearn](https://anaconda.org/anaconda/scikit-learn),
    [pytorch-pretrained-bert](https://github.com/huggingface/transformers)
    
    ```
    $ conda install pytorch torchvision -c pytorch
    $ conda install -c anaconda nltk
    $ conda install -c conda-forge spacy
    $ python -m spacy download en
    $ conda install -c anaconda scikit-learn
    $ pip install pytorch-pretrained-bert
         (version: 0.6.2)
    ```
  * The following i2b2 resources are needed:
    1. i2b2 Unstructured Notes from n2c2 NLP Research https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/: 
    2. i2b2 section boundaries from Dai et al. https://www.hindawi.com/journals/bmri/2015/873012/ (Contact Author)
    3. i2b2 labels for headers. Due to the privacy restrictions of i2b2 we cannot release our labels for the headers publicly. These can be easily created manually for the 1321 headers. Please contact us for more details. We keep the ability to run to i2b2 in the code, but show usage for MedLit only.

### Data Formatting

Before running the experiments *.pkl files need to be generated from the data for train/dev/test. 

Note: Set the output_dir to the same location for MedLit and i2b2 data. 

##### MedLit Data

These *_ref files are available. To recreate them for the EHR headers.

```
python header_util.py --output_dir <output dir> --data_dir <directory of MedLit data> --sources MedLit
```


##### i2b2 Data

These files cannot be sent directly. The i2b2 resource needs to be downloaded separately. First create *_ref.txt files with the EHR headers, then create the pkl files. 

```
python header_util.py --output_dir <output dir> --data_dir <directory of i2b2 data> --sources i2b2 --annotation_dir <directory of i2b2 annotations from Dai et al>
```  

### *.pkl files

Create the *.pkl files to load data easily:

```
python data_util.py --output_dir <output dir> --ref_data_dir <output dir from header_util> (--split_by_section) --vocab_dir <output dir>
```

Note: keep output_dir and vocab_dir to the same dir
Note: you can split by sentence or section (sentence is default)
Note: If you don't want to create the *pkl files for MedLit, set --textbook_data_sets to None
  
### Model training and test

##### GRU-RNN model
```
$ python main.py --data_dir <location of output from data_util.py> 
--vocab_dir <location of vocab from data_util.py> 
--output_dir <experiment output dir> 
--num_train_epochs 10 --do_train
--tuning_set MedLit --do_eval --test_set MedLit
```

To continue training add the following argument
```
--reload_model_file <stored *.pt file>
```

For more details,
```
$ python main.py --help
```

##### BERT model

* NOTE: This is the HuggingFace BERT v0.6.2 implementation. More recent implementations (e.g. transformers) will not be compatible with this code, but this code can be adapted to work with transformers.

*Example:*<br>
Similar to running the deep learning model, main.py, but uses BERT
```
python bert_main.py --data_dir <location of output from data_util.py> 
--vocab_dir <location of vocab from data_util.py> 
--output_dir <experiment output dir> 
--bert_model <eg bert_base_uncased> --num_train_epochs 10 --do_lower_case 
--train_data <ALL|MedLit|i2b2> --tuning_set <ALL|MedLit|i2b2> --test_set <ALL|MedLit|i2b2>
--do_train --do_eval 
```
To run with local generated model file, (e.g. biobert or to continue tuning), point to directory with the following:
```
bert_config.json
vocab.txt
pytorch_model.bin
```
For more details,
```
$ python bert_main.py --help
```
