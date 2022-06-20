# -*- coding: utf-8 -*-
from os.path import join

BASE_WORD_REGEX = r'(?:[\w]+(?:[’\'])?)|[!"#$%&\'’\(\)*+,-./:;<=>?@\[\]^_`{|}~]'
BASE_SENTENCE_REGEX = r"((?:\s*\n)+\s*|(?:(?<=[a-z0-9)]\.)\s+))(?=[A-Z-])"

class Config:
    def __init__(self):
        # Dataset name
        self.dataset_name = 'QUAERO_restrict_proc_chem_devi_diso_v2'    # any name, just for logging
        
        # where to store produced data and models (not labeled output, but for example fasttext embeddings or final model)
        self.datadir = './data/'      
        # output directory (for labeled test data)
        self.output_dir = 'out'
        
        #######
        ## DATA
        #######
        # Train, val, test directories,
        # containing Brat datasets : should be .ann and .txt files
        # self.train_dir = '/export/home/cse200093/Expe_Pheno/data/test_SOSY_DISO_train_val'
        # self.train_dir = '/export/home/cse200093/brat_data/QUAERO_FrenchMed/corpus/train_dev_medline_emea'
        # self.train_dir = '/export/home/cse200093/brat_data/n2c2_2019/train/brat_with_UMLS_stys/brat_files/train'
        # self.train_dir = '/export/home/cse200093/brat_data/expe_translation_Gold_after_conversion'
        self.train_dir = '/export/home/cse200093/brat_data/QUAERO_FrenchMed/corpus/train_dev_medline_emea_restrict_proc_chem_devi_diso'
        # self.val_dir = '/export/home/cse200093/Expe_Pheno/data/trainsosy_diso_all_attributes_MERGED/'
        self.val_dir   = None
        # '/export/home/cse200093/brat_data/QUAERO_FrenchMed/corpus/dev/MEDLINE' 
        # self.val_dir = '/export/home/cse200093/Expe_Pheno/data/final_val/' # if None, validation set will be extracted from the train
        # self.test_dir  = '/export/home/cse200093/brat_data/QUAERO_FrenchMed/corpus/test/EMEA'
        self.test_dir  = None
        # '/export/home/cse200093/brat_data/n2c2_2019/train/brat_with_UMLS_stys/brat_files/test'
        # '/export/home/cse200088/brat_data/train_bidon'      # if None, no test will be performed
        
        
        # Does the data contains nested entities?
        # Edit `nested` to fit your data:
        # - True: no constraint (entities can overlap, nest with no restriction)
        # - False: no entity overlapping at all
        # - no_same_label: entities can be nested, but not if the have the same label
        self.nested = True
        assert self.nested in [True, False, "no_same_label"]
        
        #######
        ## Carbon Tracker
        #######
        # Run Carbon tracker
        self.track_carbon = False
        
        #######
        ## MODELS
        #######
        ## BERT
        # Name or path to your BERT-like model
        self.bert_model = '/export/home/cse200093/test-mlm-2' # camembert_large
        # self.bert_model = 'camembert/camembert-large'
        # self.bert_model = "/export/home/cse200093/camembert-large/"
        # self.bert_model = "/export/home/opt/data/camembert/v0/camembert-base/"
        # self.bert_model = '/export/home/cse200093/clinicalBERT'
        # self.bert_model = "/export/home/cse200093/camembert-base/"
        # self.bert_model = 'bert-large-uncased'
        
        ## FASTTEXT
        self.fasttext_embeddings = join(self.datadir, f'{self.dataset_name}_fasttext.txt')
        self.fasttext_lang = 'fr'               # en for English
        # self.fasttext_model = 'cc.fr.300.bin'   # cc.en.300.bin for English data, 
        #                                        # or your own model if you have one
        self.fasttext_model = '/export/home/cse200093/output/output.bin'
        #self.fasttext_model = '/export/home/cse200093/BioWordVec_PubMed_MIMICIII_d200.bin'
        ## Model main hyper-parameters
        self.max_epochs = 20        # Maximum number of epochs
        self.patience = 5          # Early stopping patience
        self.finetune_bert = True   # False = freeze BERT embeddings during training 
        self.batch_size = 8
        
        #######
        ## Logging
        #######
        self.logging_tensorboard = False   # Use tensorboard logging (into "logs" directory)
        self.logging_richlogger = True    # Use rich-logger logging (table in console with all metrics)
        
        
        #######
        ## MISC
        #######
        # Regex to split Words
        self.word_regex = BASE_WORD_REGEX 
        # Regex to split sentences
        self.sentence_regex = BASE_SENTENCE_REGEX
        
        # Seeds
        # The training will run for each seed, so there will be
        # len(seeds) training.
        # Choose only one seed when developping, but 
        # give more to obtain robust results to evaluate your system
        self.seeds = (42, )   # the training will run for each seed
        #self.seeds = (42, 1558, 555, 123, 456, 789)   # the training will run for each seed

