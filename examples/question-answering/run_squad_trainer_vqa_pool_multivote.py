# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for question-answering."""


import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch
from tqdm.auto import tqdm, trange
import json

from transformers import AutoConfig, AutoModelForQuestionAnsweringVQAPool_MultiVote as AutoModelForQuestionAnswering, AutoTokenizer, HfArgumentParser, SquadDataset
from transformers import SquadDataTrainingArguments as DataTrainingArguments
from transformers import Trainer, TrainingArguments
from transformers import squad_convert_examples_to_features
from transformers.data.processors.squad_vqa import SquadResult, SquadV1Processor, SquadV2Processor, SquadProcessor, initializeWordEmbeddings
from transformers import SymbolDict, initEmbRandom

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    predict_file: Optional[str] = field(
        default='dev-v2.0.json', metadata={"help": "File to Evaluate"}
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Prepare Question-Answering task
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    is_language_sensitive = hasattr(model.config, "lang2id")
    train_dataset = (
        SquadDataset(
            data_args, tokenizer=tokenizer, is_language_sensitive=is_language_sensitive, cache_dir=model_args.cache_dir
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        SquadDataset(
            data_args,
            tokenizer=tokenizer,
            mode="dev",
            is_language_sensitive=is_language_sensitive,
            cache_dir=model_args.cache_dir,
        )
        if training_args.do_eval
        else None
    )

    weights_filename = os.path.join(data_args.data_dir, "ans_weights.json")
    if os.path.exists(weights_filename):
        with open(weights_filename, "r", encoding="utf-8") as reader:
            weights_list = json.load(reader)
    else:
        weights_list = None

    #Load Scene graph, if provided
    if data_args.scene_file:
        scene_file_path = os.path.join(data_args.data_dir, data_args.scene_file)
        with open(scene_file_path, "r", encoding="utf-8") as reader:
            scene_dataset = json.load(reader)

        sceneDict = SymbolDict()
        for scene in tqdm(scene_dataset.values(), desc="Creating Scene Dictionary"):
            for obj in scene["objects"].values():
                sceneDict.addSymbols(obj["name"])
                sceneDict.addSymbols(obj["attributes"])
                for rel in obj["relations"]:
                   sceneDict.addSymbols(rel["name"])
        #create vocab           
        sceneDict.createVocab(minCount=0)

        if data_args.cached_embedding:
            import numpy as np
            embedding = np.load(os.path.join(data_args.data_dir, data_args.cached_embedding))
        elif data_args.emb_file:
            embedding = initializeWordEmbeddings(data_args.emb_dim, 
                                                wordsDict=sceneDict, 
                                                random=False,
                                                filename=os.path.join(data_args.data_dir, data_args.emb_file))
        else:
            embedding = initEmbRandom(sceneDict.getNumSymbols(), data_args.emb_dim)
    else:
        scene_dataset = None
        sceneDict = None
        embedding = None

    # Initialize our Trainer
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, scene_dataset=scene_dataset, scene_dict = sceneDict, embedding = embedding, weights_list = weights_list)

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        evaluate(eval_dataset, trainer)


    return eval_results

def evaluate(eval_dataset, trainer):
    eval_dataloader = trainer.get_eval_dataloader(eval_dataset)
    batch_size = eval_dataloader.batch_size
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", batch_size)

    all_results = []
    model = trainer.model
    cnt = 0
    for inputs in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        print(inputs.keys())
        inputs = trainer._prepare_inputs(inputs, model)
        cnt +=1

        with torch.no_grad():
            outputs = model(**inputs)
            print('outputs')
            print(outputs)
        if cnt > 0:
            break


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
