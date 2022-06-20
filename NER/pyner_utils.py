# -*- coding: utf-8 -*-

"""
from main import main
seeds = (42, 1558, 555, 123, 456, 789)
for dataset in ("deft",): # or deft:dev
    for seed in seeds:
        main(dataset, seed=seed, do_tagging="full", do_biaffine=True, finetune_bert=True)
        main(dataset, seed=seed, do_tagging="full", do_biaffine=True, finetune_bert=False)
"""

import argparse
import gc
import json
import string
import torch
import logging

import pandas as pd

from nlstruct.base import *
from nlstruct.checkpoint import *
from nlstruct.datasets import *
from rich_logger import RichTableLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers

from carbontracker.tracker import CarbonTracker
from carbontracker import parser

from os.path import dirname, isdir, join
from os import makedirs

if "display" not in globals():
    display = print

shared_cache = {}

BASE_WORD_REGEX = r'(?:[\w]+(?:[’\'])?)|[!"#$%&\'’\(\)*+,-./:;<=>?@\[\]^_`{|}~]'
BASE_SENTENCE_REGEX = r"((?:\s*\n)+\s*|(?:(?<=[\w0-9]{2,}\.|[)]\.)\s+))(?=[[:upper:]]|•|\n)"

class CarbonTrackerCallback(Callback):
    """
    Callback for carbon tracking at each epoch
    """
    def __init__(self, max_epochs, log_dir='ct_logs'):
        super().__init__()
        self.log_dir = log_dir
        self.tracker = CarbonTracker(epochs=max_epochs, log_dir=log_dir, components="gpu")
        # Get log file name
        for handler in self.tracker.logger.logger_output.handlers:
            if type(handler) == logging.FileHandler:
                self.log_file_name = handler.baseFilename
                #print("+++++++++++++++++" + self.log_file_name + "++++++++++++++++++++")
        
        
    def on_train_epoch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', unused: 'Optional' = None):
        self.tracker.epoch_start()

    def on_train_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', unused: 'Optional' = None):
        self.tracker.epoch_end()
        
    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', unused: 'Optional' = None):
        self.tracker.stop()
        
    def get_infos(self):
        logs = parser.parse_all_logs(log_dir=self.log_dir)
        log_dicts = [l for l in logs if f['output_filename']]
        assert len(log_dicts) == 1
        log_dict = log_dicts[0]
        return {
            'consumption': log_dict['actual'],
            'components': log_dict['components']['gpu']['devices']
        }




def run_pyner(
    train,
    val,
    test,
    bert_name,
    seed,
    word_regex=BASE_WORD_REGEX,
    sentence_split_regex=BASE_SENTENCE_REGEX,
    outpath=None,
    run_name=None,    
    nested=True,
    do_char=True,
    do_biaffine=True,
    do_tagging="full",
    doc_context=True,
    finetune_bert=False,
    bert_lower=False,
    n_bert_layers=4,
    biaffine_size=64,
    bert_proj_size=None,
    biaffine_loss_weight=1.,
    hidden_size=400,
    #max_steps=None,
    max_epochs=20,
    resources="",
    fasttext_file="",  # set to "" to disable
    unique_label=False,
    norm_bert=False,
    dropout_p=0.1,
    batch_size=32,
    patience=3,
    lr=1e-3,
    use_lr_schedules=True,
    word_pooler_mode="mean",
    bert_size=None,
    hf_resources="",
    predict_kwargs=None,
    gpus=1,
    logging_tensorboard=True,
    logging_richlogger=False,
    logger=None,
    track_carbon=False
):
    if logger is None:
        logging.basicConfig(
           format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
           datefmt="%m/%d/%Y %H:%M:%S",
           level=logging.INFO,
        )
    logger = logging.getLogger(__name__)

    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    if predict_kwargs is None:
        predict_kwargs = {}
    if run_name is None:
        run_name = dataset_name

    for name, value in locals().items():
        print(name.ljust(40), value)

    if nested == "no_same_label":
        filter_predictions = "no-crossing-same-label"
    elif nested == False:
        filter_predictions = "no-overlapping"
    else:        
        filter_predictions = False
        
    dataset = BRATDataset(
        train=train,
        val=val if val is not None else 0.2,
        test=test
    )
    metrics = {
        "exact": dict(module="dem", binarize_tag_threshold=1., binarize_label_threshold=1., word_regex=word_regex),
        "half_word": dict(module="dem", binarize_tag_threshold=0.5, binarize_label_threshold=1., word_regex=word_regex),
        "any_word": dict(module="dem", binarize_tag_threshold=1e-5, binarize_label_threshold=1., word_regex=word_regex),
    }


    if filter_predictions is not False:
        predict_kwargs["filter_predictions"] = filter_predictions

    display(dataset.describe())

    model = InformationExtractor(
        seed=seed,
        preprocessor=dict(
            module="ner_preprocessor",
            bert_name=bert_name,  # transformer name
            bert_lower=bert_lower,
            split_into_multiple_samples=True,
            sentence_split_regex=sentence_split_regex,  # regex to use to split sentences (must not contain consuming patterns)
            sentence_balance_chars=(),  # try to avoid splitting between parentheses
            sentence_entity_overlap="split",  # raise when an entity spans more than one sentence
            word_regex=word_regex,  # regex to use to extract words (will be aligned with bert tokens), leave to None to use wordpieces as is
            substitutions=(),  # Apply these regex substitutions on sentences before tokenizing
            keep_bert_special_tokens=False,
            min_tokens=0,
            doc_context=doc_context,
            join_small_sentence_rate=0.,
            max_tokens=256,  # split when sentences contain more than 512 tokens
            large_sentences="equal-split",  # for these large sentences, split them in equal sub sentences < 512 tokens
            empty_entities="raise",  # when an entity cannot be mapped to any word, raise
            vocabularies={
                **{  # vocabularies to use, call .train() before initializing to fill/complete them automatically from training data
                    "entity_label": dict(module="vocabulary", values=sorted(dataset.labels()), with_unk=False, with_pad=False),
                },
                **({
                       "char": dict(module="vocabulary", values=string.ascii_letters + string.digits + string.punctuation, with_unk=True, with_pad=False),
                   } if do_char else {})
            },
            fragment_label_is_entity_label=True,
            multi_label=False,
            filter_entities=None,  # "entity_type_score_density", "entity_type_score_lesion"),
        ),
        dynamic_preprocessing=False,

        # Text encoders
        encoder=dict(
            module="concat",
            dropout_p=0.5,
            encoders=[
                dict(
                    module="bert",
                    path=bert_name,
                    n_layers=n_bert_layers,
                    freeze_n_layers=0 if finetune_bert is not False else -1,  # freeze 0 layer (including the first embedding layer)
                    bert_dropout_p=None if finetune_bert else 0.,
                    token_dropout_p=0.,
                    proj_size=bert_proj_size,
                    output_lm_embeds=False,
                    combine_mode="scaled_softmax" if not norm_bert else "softmax",
                    do_norm=norm_bert,
                    do_cache=not finetune_bert,
                    word_pooler=dict(module="pooler", mode=word_pooler_mode),
                ),
                *([dict(
                    module="char_cnn",
                    in_channels=8,
                    out_channels=50,
                    kernel_sizes=(3, 4, 5),
                )] if do_char else []),
                *([dict(
                    module="word_embeddings",
                    filename=fasttext_file,
                )] if fasttext_file else [])
            ],
        ),
        decoder=dict(
            module="contiguous_entity_decoder",
            contextualizer=dict(
                module="lstm",
                num_layers=3,
                gate=dict(module="sigmoid_gate", init_value=0., proj=True),
                bidirectional=True,
                hidden_size=hidden_size,
                dropout_p=0.4,
                gate_reference="last",
            ),
            span_scorer=dict(
                module="bitag",
                do_biaffine=do_biaffine,
                do_tagging=do_tagging,
                do_length=False,

                threshold=0.5,
                max_fragments_count=200,
                max_length=40,
                hidden_size=biaffine_size,
                allow_overlap=True,
                dropout_p=dropout_p,
                tag_loss_weight=1.,
                biaffine_loss_weight=biaffine_loss_weight,
                eps=1e-14,
            ),
            intermediate_loss_slice=slice(-1, None),
        ),

        _predict_kwargs=predict_kwargs,
        batch_size=batch_size,

        # Use learning rate schedules (linearly decay with warmup)
        use_lr_schedules=use_lr_schedules,
        warmup_rate=0.1,

        gradient_clip_val=10.,
        _size_factor=0.001,

        # Learning rates
        main_lr=lr,
        fast_lr=lr,
        bert_lr=5e-5,

        # Optimizer, can be class or str
        optimizer_cls="transformers.AdamW",
        metrics=metrics,
    ).train()

    model.encoder.encoders[0].cache = shared_cache

    try:
        callbacks = []
        if track_carbon:
            carbontracker = CarbonTrackerCallback(max_epochs)
            callbacks.append(carbontracker)
            
        early_stop_callback = EarlyStopping(
            monitor="val_exact_f1", 
            min_delta=0.00, 
            patience=patience, 
            verbose=False, 
            mode="max")
        callbacks.append(early_stop_callback)
        
        loggers = []
        if logging_tensorboard:
            tb_logger = pl_loggers.TensorBoardLogger("logs/")
            loggers.append(tb_logger) 
        if logging_richlogger:
            loggers.append(RichTableLogger(key="epoch", fields={
                    "epoch": {},
                    "step": {},
                    "(.*)_?loss": {"goal": "lower_is_better", "format": "{:.4f}"},
                    "(.*)_precision": False, #{"goal": "higher_is_better", "format": "{:.4f}", "name": r"\1_p"},
                    "(.*)_recall": False, #{"goal": "higher_is_better", "format": "{:.4f}", "name": r"\1_r"},
                    "(.*)_tp": False,
                    "(.*)_f1": {"goal": "higher_is_better", "format": "{:.4f}", "name": r"\1_f1"},

                    ".*_lr|max_grad": {"format": "{:.2e}"},
                    "duration": {"format": "{:.0f}", "name": "dur(s)"},
                }))

        
        chk_path = join('checkpoints', run_name + '-{hashkey}-{global_step:05d}')
        chk_dir = dirname(chk_path)
        if not isdir(chk_dir):
            makedirs(chk_dir)
        
        model_checkpoint_callback = ModelCheckpoint(chk_path)
        callbacks.append(model_checkpoint_callback)
        
        trainer = pl.Trainer(
            gpus=gpus,
            #progress_bar_refresh_rate=False,
            checkpoint_callback=False,  # do not make checkpoints since it slows down the training a lot
            callbacks=callbacks,
            logger=loggers,
            #val_check_interval=max_steps // 10,
            #max_steps=max_steps)
            max_epochs=max_epochs)
        #print(dataset.test_data)
        trainer.fit(model, dataset)
        trainer.logger[0].finalize(True)
        logger.info('Training done')

        result_output_filename = "checkpoints/" + run_name + "-{}.json".format(model_checkpoint_callback.hashkey)
        if not os.path.exists(result_output_filename):
            model.cuda();
            val_results = model.metrics(list(model.predict(dataset.val_data)), dataset.val_data)
            test_predictions = list(model.predict(dataset.test_data))
            last_inputs = model.last_inputs

            torch.save(last_inputs, '_last_inputs.pkl')
            logger.info('written last_inputs to _last_inputs.pkl')
            
            test_results = model.metrics(test_predictions, dataset.test_data)
            if outpath is not None:
                export_to_brat(test_predictions, filename_prefix=outpath)
       
           
            logger.info('Validation results:')
            display(pd.DataFrame(val_results).T)
            logger.info('Test results')
            display(pd.DataFrame(test_results).T)            

            def json_default(o):
                if isinstance(o, slice):
                    return str(o)
                raise

            with open(result_output_filename, 'w') as json_file:
                res = {
                    "config": {**get_config(model), 
                        "max_epochs": max_epochs, 
                        "early_stopping": early_stop_callback.stopped_epoch if early_stop_callback.stopped_epoch > 0 else max_epochs, 
                        "patience":patience},
                    "results": {
                        "val": val_results,
                        "test": test_results
                    },
                }
                if track_carbon:
                    res['carbontracker'] = carbontracker.get_infos()
                json.dump(res, json_file, default=json_default)
    except AlreadyRunningException as e:
        model = None
        logger.warning("Experiment was already running")
        logger.warning(str(e))
        result_output_filename = None

    return model, result_output_filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', help='seed', type=int, default=42)

    parser.add_argument('--do_char', help='do_char', action="store_true", default=False)
    parser.add_argument('--do_biaffine', help='do_biaffine', action="store_true", default=False)
    parser.add_argument('--do_tagging', help='do_tagging', default=False, const="full", nargs="?")
    parser.add_argument('--doc_context', help='doc_context', action="store_true", default=False)

    parser.add_argument('--finetune_bert', help='finetune_bert', action="store_true", default=False)
    parser.add_argument('--bert_lower', help='bert_lower', action="store_true", default=False)

    parser.add_argument('--n_bert_layers', help='n_bert_layers', type=int, default=4)
    parser.add_argument('--bert_proj_size', help='bert_proj_size', default=None)
    parser.add_argument('--unique_label', help='unique_label', action="store_true", default=False)
    parser.add_argument('--hidden_size', help='hidden_size', type=int, default=400)

    parser.add_argument('--biaffine_size', help='biaffine_size', type=int, default=None)
    parser.add_argument('--biaffine_loss_weight', help='biaffine_loss_weight', type=float, default=1.)
    parser.add_argument('--max_epochs', help='max_epochs', type=int, default=20)
    parser.add_argument('--bert_name', help='bert_name', required=True, default=None)
    parser.add_argument('--fasttext_file', help='fasttext_file', default=None)
    parser.add_argument('--norm_bert', help='norm_bert', default=False)

    parser.add_argument('--resources', help='resources', default="")

    args = parser.parse_args()

    main(**vars(args))
