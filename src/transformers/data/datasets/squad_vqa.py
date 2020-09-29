import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union

import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset

from ...modeling_auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING
from ...tokenization_utils import PreTrainedTokenizer
from ..processors.squad_vqa import SquadFeatures, SquadV1Processor, SquadV2Processor, squad_convert_examples_to_features


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class SquadDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    model_type: str = field(
        default=None, metadata={"help": "Model type selected in the list: " + ", ".join(MODEL_TYPES)}
    )
    data_dir: str = field(
        default=None, metadata={"help": "The input data dir. Should contain the .json files for the SQuAD task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    max_query_length: int = field(
        default=64,
        metadata={
            "help": "The maximum number of tokens for the question. Questions longer than this will "
            "be truncated to this length."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, the SQuAD examples contain some that do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0, metadata={"help": "If null_score - best_non_null is greater than the threshold predict null."}
    )
    n_best_size: int = field(
        default=20, metadata={"help": "If null_score - best_non_null is greater than the threshold predict null."}
    )
    lang_id: int = field(
        default=0,
        metadata={
            "help": "language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)"
        },
    )
    scene_file: str = field(
        default=None, metadata={"help": "The input scene graph file. Should contain the .json files for the SQuAD task."}
    )
    cached_embedding: str = field(
        default=None, metadata={"help": "The scene graph vocab embedding file."}
    )
    emb_file: str = field(
        default=None, metadata={"help": "The input embedding file (e.g. Glove), this adds time to the total processing."}
    )
    emb_dim: int = field(
        default=300,
        metadata={
            "help": "Embedding dimension, when raw file is provided. If cached embedding is provided, this is not needed."
        },
    )

    threads: int = field(default=1, metadata={"help": "multiple threads for converting example to features"})


class Split(Enum):
    train = "train"
    dev = "dev"


class SquadDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    args: SquadDataTrainingArguments
    features: List[SquadFeatures]
    mode: Split
    is_language_sensitive: bool

    def __init__(
        self,
        args: SquadDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        is_language_sensitive: Optional[bool] = False,
        cache_dir: Optional[str] = None,
    ):
        self.args = args
        self.is_language_sensitive = is_language_sensitive
        self.processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        self.mode = mode
        # Load data features from cache or dataset file
        version_tag = "v2" if args.version_2_with_negative else "v1"
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode.value, tokenizer.__class__.__name__, str(args.max_seq_length), version_tag,
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if args.version_2_with_negative:
                ans_to_ix, ix_to_ans = self.processor.create_dicts(args.data_dir)
            else:
                ans_to_ix, ix_to_ans = {}, {}
            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                if mode == Split.dev:
                    examples = self.processor.get_dev_examples(args.data_dir, to_ix_dict=ans_to_ix)
                else:
                    examples = self.processor.get_train_examples(args.data_dir, to_ix_dict=ans_to_ix)

                self.features = squad_convert_examples_to_features(
                    examples=examples,
                    tokenizer=tokenizer,
                    max_seq_length=args.max_seq_length,
                    doc_stride=args.doc_stride,
                    max_query_length=args.max_query_length,
                    is_training=mode == Split.train,
                    threads=args.threads,
                )

                start = time.time()
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # Convert to Tensors and build dataset
        feature = self.features[i]

        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        attention_mask = torch.tensor(feature.attention_mask, dtype=torch.long)
        token_type_ids = torch.tensor(feature.token_type_ids, dtype=torch.long)
        cls_index = torch.tensor(feature.cls_index, dtype=torch.long)
        p_mask = torch.tensor(feature.p_mask, dtype=torch.float)
        is_impossible = torch.tensor(feature.is_impossible, dtype=torch.float)
        #orig_ans = torch.tensor(feature.orig_ans, dtype=torch.float)
        if self.args.version_2_with_negative:
            titles = torch.tensor(feature.title, dtype=torch.long)
        else:
            titles = torch.tensor(feature.title)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "titles": titles
        }

        if self.args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
            del inputs["token_type_ids"]

        if self.args.model_type in ["xlnet", "xlm"]:
            inputs.update({"cls_index": cls_index, "p_mask": p_mask})
            if self.args.version_2_with_negative:
                inputs.update({"is_impossible": is_impossible})
            if self.is_language_sensitive:
                inputs.update({"langs": (torch.ones(input_ids.shape, dtype=torch.int64) * self.args.lang_id)})

        if self.mode == Split.train:
            start_positions = torch.tensor(feature.start_position, dtype=torch.long)
            end_positions = torch.tensor(feature.end_position, dtype=torch.long)
            orig_answers = torch.tensor(feature.orig_ans, dtype=torch.long)
            inputs.update({"start_positions": start_positions, "end_positions": end_positions})
            inputs.update({"orig_answers": orig_answers})
            if self.args.version_2_with_negative:
                inputs.update({"is_impossibles": int(is_impossible)})

        return inputs
