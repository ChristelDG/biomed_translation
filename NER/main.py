from os.path import isfile, isdir, join, dirname
from os import makedirs
import logging
import operator
import fasttext
import fasttext.util

from nlstruct.datasets.brat import load_from_brat
from nlstruct.data_utils import regex_tokenize

from pyner_utils import run_pyner

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Load config 
    logger.info(f'   Load configuration')
    from config import Config
    conf = Config()
    
    # Sanity checks
    error = False
    if not isdir(conf.train_dir):
        logger.error(f'Train dir {conf.train_dir} does not exist')
        error = True
    if conf.val_dir is not None:
        if not isdir(conf.val_dir):
            logger.error(f'Validation dir {conf.val_dir} does not exist')
            error = True
    if conf.test_dir is not None:
        if not isdir(conf.test_dir):
            logger.error(f'Test dir {conf.test_dir} does not exist')
            error = True
    if conf.nested not in [True, False, "no_same_label"]:
        logger.error(f'Unknown value {conf.nested} for configuration parameter `nested`')
        error = True
    try:
        ft_dir = dirname(conf.fasttext_embeddings)
        if not isdir(ft_dir):
            makedirs(ft_dir)
    except Exception as err:
        logging.error(f'Could not create fasttext directory {conf.fasttext_embeddings}')
        raise err

    if error:
        raise Exception('Sanity check has failed, read the error messages')    
        
    word_split_regex = conf.word_regex    
        
    ##########
    ## Create FASTTEXT embeddings
    ##########
    if isfile(conf.fasttext_embeddings):
        logger.info(f'   Fasttext embeddings already created ({conf.fasttext_embeddings}). Remove this file to compute them from scratch')
    else:
        logger.info(f"   Let's build fasttext embeddings for this dataset (will be saved to {conf.fasttext_embeddings})")
        logger.info(f'   Load fasttext {conf.fasttext_model} ({conf.fasttext_lang} -- can be long, but it\'s only once)')
        if not isfile(conf.fasttext_model):
            fasttext.util.download_model(conf.fasttext_lang, if_exists='ignore')  
        ft = fasttext.load_model(conf.fasttext_model)
        # Load corpora
        # Note that we also load test dataset here for simplicity.
        #   It could be done at evaluation time with the exact same result.
        logger.info(f'   Load datasets')
        datasets = [load_from_brat(conf.train_dir)]
        if conf.val_dir is not None:
            datasets.append(load_from_brat(conf.val_dir))
        if conf.test_dir is not None:
            datasets.append(load_from_brat(conf.test_dir))
            
        # Word-tokenize corpora
        logger.info('   Tokenize datasets')

        # Collect words and rank them by frequency
        all_words = {}
        for dataset in datasets:
            for doc in dataset:
                words = regex_tokenize(doc['text'], reg=word_split_regex)
                for word in words['text']:
                    n = all_words.get(word, 0)
                    all_words[word] = n+1

        logger.info(f'   Build fasttext embeddings for the {len(all_words)} found words')
        with open(conf.fasttext_embeddings, 'w') as ft_out:
            for word, _ in sorted(all_words.items(), key=operator.itemgetter(1), reverse=True):
                emb = ft.get_word_vector(word)
                ft_out.write(word + ' ' + ' '.join(['{:.6f}'.format(e) for e in emb]) + '\n')
        logger.info(f'   fasttext embeddings written to {conf.fasttext_embeddings}')
            
    ##########
    ## Train pyner
    ##########
    if len(conf.seeds) > 1:
        logger.info(f'Will run the training {len(conf.seeds)} times...')
    for i, seed in enumerate(conf.seeds):
        logger.info(f'{i+1}. Train {conf.dataset_name} with seed {seed} (nested={conf.nested})')
        if conf.output_dir:
            out_brat_dir = join(conf.output_dir, f'{i+1}')
            result_file = join(conf.output_dir, 'results.csv')
            if not isdir(out_brat_dir):
                makedirs(out_brat_dir)
        else:
            out_brat_dir = None

        model, result_filename = run_pyner(train=conf.train_dir, 
                                            val=conf.val_dir, 
                                            test=conf.test_dir, 
                                            seed=seed,
                                            outpath=out_brat_dir, 
                                            word_regex=word_split_regex,
                                            sentence_split_regex=conf.sentence_regex,
                                            run_name=f'{conf.dataset_name}/{i}',
                                            bert_name=conf.bert_model,
                                            fasttext_file=conf.fasttext_embeddings,   
                                            nested=conf.nested,
                                            finetune_bert=conf.finetune_bert,
                                            max_epochs=conf.max_epochs,
                                            batch_size=conf.batch_size,
                                            logging_tensorboard=conf.logging_tensorboard,
                                            logging_richlogger=conf.logging_richlogger,
                                            logger=logger,
                                            track_carbon=conf.track_carbon
                                 ) #config.sentence_split_regex['brat'])
        # Save model if there is only one seed
        if len(conf.seeds) == 1:
            model_out_file = join(conf.output_dir, f'{conf.dataset_name}-model.pt')
            model.save_pretrained(model_out_file)
            logger.info(f'Trained model saved to {model_out_file}')
            logger.info(f'To use this model: ')
            logger.info(f'   ner = load_pretrained("{model_out_file}")')
            logger.info(f'   export_to_brat(ner.predict(load_from_brat("path/to/brat/test")), filename_prefix="path/to/exported_brat")')
            logger.info('')
        
        if out_brat_dir is not None:
            logger.info(f'Brat results written to {out_brat_dir}')
        with open(result_file, 'a') as f_out:
            f_out.write(f'{conf.dataset_name}\t{seed}\t{result_filename}\n')
        logger.info(f'   Train pyner DONE for {conf.dataset_name}/{i}, results written to {result_filename}')
    logger.info(f'DONE')
