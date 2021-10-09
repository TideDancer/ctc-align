import logging
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Union
import datasets
import numpy as np
import librosa
import torch
from torch import nn
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    trainer_utils,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
)
logger = logging.getLogger(__name__)
from model import Wav2Vec2ForPCTC, Wav2Vec2CTCTokenizerPho
import json
import util
from util import DataCollatorCTCWithPadding, CTCTrainer
ph_61_to_39_map = json.load(open('vocabs/ph_61_to_39.map','r'))

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_feature_extractor: Optional[bool] = field(
        default=True, metadata={"help": "Whether to freeze the feature extractor layers of the model."}
    )
    verbose_logging: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to log verbose messages or not."},
    )
    ctc_type : Optional[str] = field(default='standard', metadata={"help": "standard, no-skip"})
    prior_type : Optional[str] = field(default=None, metadata={"help": "no, entropy, uniform, posterior"})

def configure_logger(model_args: ModelArguments, training_args: TrainingArguments):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging_level = logging.WARNING
    if model_args.verbose_logging:
        logging_level = logging.DEBUG
    elif trainer_utils.is_main_process(training_args.local_rank):
        logging_level = logging.INFO
    logger.setLevel(logging_level)

@dataclass
class DataTrainingArguments:
    dataset_name: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_split_name: Optional[str] = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    validation_split_name: Optional[str] = field(
        default="validation",
        metadata={
            "help": "The name of the validation data set split to use (via the datasets library). Defaults to 'validation'"
        },
    )
    target_text_column: Optional[str] = field(
        default="phonetic_detail",
        metadata={"help": "Column in the dataset that contains label (target phonemes). Defaults to 'phonetic_detail'"},
    )
    speech_file_column: Optional[str] = field(
        default="file",
        metadata={"help": "Column in the dataset that contains speech file path. Defaults to 'file'"},
    )
    target_feature_extractor_sampling_rate: Optional[bool] = field(
        default=False,
        metadata={"help": "Resample loaded audio to target feature extractor's sampling rate or not."},
    )
    max_duration_in_seconds: Optional[float] = field(
        default=None,
        metadata={"help": "Filters out examples longer than specified. Defaults to no filtering."},
    )
    orthography: Optional[str] = field(
        default="librispeech",
        metadata={
            "help": "Orthography used for normalization and tokenization: 'librispeech' (default), 'timit', or 'buckwalter'."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    vocab: Optional[str] = field(default="regular", metadata={"help": "regular, splitter, s-boundary, t-boundary"})
    debug_mode: Optional[bool] = field(default=False)
    eval_train: Optional[bool] = field(default=False)

@dataclass
class Orthography:
    do_lower_case: bool = False
    vocab_file: Optional[str] = None
    # word_delimiter_token: Optional[str] = "|"
    translation_table: Optional[Dict[str, str]] = field(default_factory=dict)
    words_to_remove: Optional[Set[str]] = field(default_factory=set)
    untransliterator: Optional[Callable[[str], str]] = None

    @classmethod
    def from_name(cls, name: str):
        return cls()

    def preprocess_for_training(self, pho_list: list, vocab_type='regular') -> list:
        # perform 61 -> 39 mapping
        utt = list(map(lambda e: ph_61_to_39_map[e] if e in ph_61_to_39_map else e, pho_list['utterance']))
        # process end
        tmp = pho_list['stop'] + [-100]
        tmp_utt = utt + ['dummy']
        i = 0
        stops = []
        while i < len(tmp)-1:
            if tmp_utt[i] != tmp_utt[i+1] and len(tmp_utt[i]) > 0:
                stops.append(tmp[i])
            i += 1
        
        # process phos
        tmp_utt = utt + ['dummy']
        i, phos = 0, []
        while i < len(tmp_utt) - 1:
            if tmp_utt[i] != tmp_utt[i+1] and len(tmp_utt[i]) > 0:
                phos.append(tmp_utt[i])
            i += 1

        # handle s-boundary and t-boundary
        if vocab_type == 's-boundary' or vocab_type == 'splitter' or vocab_type == 't-boundary':
            tmp_phos, tmp_starts, tmp_stops = [], [], []
            for i in range(len(phos)-1):
                tmp_phos.append(phos[i])
                if vocab_type == 's-boundary': tmp_phos.append(phos[i]+'_')
                if vocab_type == 'splitter': tmp_phos.append('_')
                if vocab_type == 't-boundary': tmp_phos.append(phos[i]+'_'+phos[i+1])
                tmp_stops += [stops[i], stops[i]]
            tmp_phos.append(phos[-1])
            tmp_stops.append(stops[-1])
            phos, stops = tmp_phos, tmp_stops
        
        assert len(phos) == len(stops)
        return phos, stops # a list of phonemes

    def create_processor(self, model_args: ModelArguments) -> Wav2Vec2Processor:
        if model_args.model_name_or_path == 'facebook/wav2vec2-large':
            model_name = 'facebook/wav2vec2-base'
        else:
            model_name = model_args.model_name_or_path
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name, cache_dir=model_args.cache_dir
        )
        if self.vocab_file:
            tokenizer = Wav2Vec2CTCTokenizerPho(
                self.vocab_file,
                cache_dir=model_args.cache_dir,
                do_lower_case=self.do_lower_case,
                # word_delimiter_token=self.word_delimiter_token,
            )
        else:
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
                model_name,
                cache_dir=model_args.cache_dir,
                do_lower_case=self.do_lower_case,
                # word_delimiter_token=self.word_delimiter_token,
            )
        return Wav2Vec2Processor(feature_extractor, tokenizer)

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    configure_logger(model_args, training_args)

    orthography = Orthography.from_name(data_args.orthography.lower())
    orthography.vocab_file = 'vocabs/timit.39.' + data_args.vocab 
    processor = orthography.create_processor(model_args)
    model = Wav2Vec2ForPCTC.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        gradient_checkpointing=training_args.gradient_checkpointing,
        vocab_size=len(processor.tokenizer),
        ctc_type=model_args.ctc_type,
        prior_type=model_args.prior_type,
    )

    # proc data
    train_dataset = datasets.load_dataset(data_args.dataset_name, data_args.dataset_config_name, split=data_args.train_split_name)
    if data_args.eval_train:
        val_dataset = datasets.load_dataset(data_args.dataset_name, data_args.dataset_config_name, split=data_args.train_split_name)
    else: 
        val_dataset = datasets.load_dataset(data_args.dataset_name, data_args.dataset_config_name, split=data_args.validation_split_name)

    # debug
    if data_args.debug_mode:
        train_dataset = train_dataset.select(list(range(50)))
        val_dataset = val_dataset.select(list(range(50)))

    wer_metric = datasets.load_metric("wer")
    target_sr = processor.feature_extractor.sampling_rate if data_args.target_feature_extractor_sampling_rate else None
    
    def prepare_example(example):  # TODO(elgeish) make use of multiprocessing?
        example["speech"], example["sampling_rate"] = librosa.load(example[data_args.speech_file_column], sr=target_sr)
        if data_args.max_duration_in_seconds is not None:
            example["duration_in_seconds"] = len(example["speech"]) / example["sampling_rate"]
        # Normalize and clean up text; order matters!
        example[data_args.target_text_column], example['stops'] = orthography.preprocess_for_training(example[data_args.target_text_column], vocab_type=data_args.vocab)
        return example

    train_dataset = train_dataset.map(prepare_example, keep_in_memory=False, load_from_cache_file=True, remove_columns=[data_args.speech_file_column])
    val_dataset = val_dataset.map(prepare_example, remove_columns=[data_args.speech_file_column])

    if data_args.max_duration_in_seconds is not None:
        def filter_by_max_duration(example):
            return example["duration_in_seconds"] <= data_args.max_duration_in_seconds
        old_train_size = len(train_dataset)
        old_val_size = len(val_dataset)
        train_dataset = train_dataset.filter(filter_by_max_duration, remove_columns=["duration_in_seconds"])
        val_dataset = val_dataset.filter(filter_by_max_duration, remove_columns=["duration_in_seconds"])
        if len(train_dataset) > old_train_size:
            logger.warning(f"Filtered out {len(train_dataset) - old_train_size} train example(s) longer than {data_args.max_duration_in_seconds} second(s).")
        if len(val_dataset) > old_val_size:
            logger.warning(f"Filtered out {len(val_dataset) - old_val_size} validation example(s) longer than {data_args.max_duration_in_seconds} second(s).")
    logger.info(f"Split sizes: {len(train_dataset)} train and {len(val_dataset)} validation.")

    def prepare_dataset(batch):
        # check that all files have the correct sampling rate
        assert (
            len(set(batch["sampling_rate"])) == 1
        ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."
        batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
        batch["labels"] = processor.tokenizer(batch[data_args.target_text_column], is_split_into_words=True).input_ids # batch x list x phones
        return batch

    train_dataset = train_dataset.map(
        prepare_dataset,
        batch_size=training_args.per_device_train_batch_size,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        keep_in_memory=True,
        load_from_cache_file=False
    )
    val_dataset = val_dataset.map(
        prepare_dataset,
        batch_size=training_args.per_device_train_batch_size,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        for dtw_type in ['l2', 'l1']:
            if dtw_type == 'l2':
                dist_func = lambda x,y: (x-y)**2 # squared distance
            if dtw_type == 'l1':
                dist_func = lambda x,y: abs(x-y) # absolute distance

            if data_args.vocab == 'regular':
                boundary_list = []
                dist, acc, bias = util.dist_regular(pred_ids, val_dataset, dist_func)
            elif data_args.vocab == 'splitter' or data_args.vocab == 's-boundary' or data_args.vocab == 't-boundary':
                if data_args.vocab == 'splitter':
                    boundary_list = ['_']
                else:
                    boundary_list = list(filter(lambda e: '_' in e, list(processor.tokenizer.get_vocab().keys())))
                dist, acc, bias = util.dist_boundary(pred_ids, val_dataset, processor, boundary_list, dist_func, mid_boundary=True)
            
            if dtw_type == 'l2':
                l2_dist, l2_acc, l2_bias = dist, acc, bias
            if dtw_type == 'l1':
                l1_dist, l1_acc, l1_bias = dist, acc, bias

        pred_str = processor.batch_decode(pred_ids, spaces_between_special_tokens=True) # spaces_between_special_tokens=True to use ' ' to join list of tokens
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False, spaces_between_special_tokens=True) 
        pred_str, label_str = util.clean_str(pred_str, boundary_list), util.clean_str(label_str, boundary_list)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer":wer, "mse":l2_dist, "l2_acc":l2_acc, "l2_bias":l2_bias, "mae": l1_dist, "l1_acc": l1_acc, "l1_bias":l1_bias}

    if model_args.freeze_feature_extractor:
        model.freeze_feature_extractor()

    trainer = CTCTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.feature_extractor,
    )

    if training_args.do_train:
        trainer.train()
    if training_args.do_eval:
        trainer.evaluate()

if __name__ == "__main__":
    main()
