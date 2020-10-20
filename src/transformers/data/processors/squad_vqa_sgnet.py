import json
import logging
import os
from functools import partial
from multiprocessing import Pool, cpu_count
import spacy

import numpy as np
from tqdm import tqdm

from ...file_utils import is_tf_available, is_torch_available
from ...tokenization_bert import whitespace_tokenize
from ...tokenization_utils_base import TruncationStrategy
from .utils import DataProcessor


# Store the tokenizers which insert 2 separators tokens
MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart"}


if is_torch_available():
    import torch
    from torch.utils.data import TensorDataset

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

class SimpleNlp(object):
    def __init__(self):
        self.nlp = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat'])
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
        # self.nlp = nltk.data.load('tokenizers/punkt/english.pickle').tokenize

    def nlp(self, text):
        return self.nlp(text)

def squad_convert_example_to_features(
    example, max_seq_length, doc_stride, max_query_length, padding_strategy, is_training
):
    features = []

    que_span = example.que_span
    org_que_token = example.token_que
    org_doc_token = example.token_doc
    all_doc_span = example.doc_span

    query_tokens = tokenizer.tokenize(example.question_text)

    que_tokens = []
    prev_is_whitespace = True
    for c in example.question_text:
        if _is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                que_tokens.append(c)
            else:
                que_tokens[-1] += c
            prev_is_whitespace = False

    sub_que_span = []

    que_org_to_split_map = {}
    pre_tok_len = 0
    for idx, que_token in enumerate(que_tokens):
        sub_que_tok = tokenizer.tokenize(que_token)
        que_org_to_split_map[idx] = (pre_tok_len, len(sub_que_tok) + pre_tok_len - 1)
        pre_tok_len += len(sub_que_tok)

    for idx, (start_ix, end_ix) in enumerate(que_span):
        head_start, head_end = que_org_to_split_map[idx]

        # sub_start_idx and sub_end_idx of children of head node
        head_spans = [(que_org_to_split_map[start_ix-1][0], que_org_to_split_map[end_ix-1][1])]
        # all other head sub_tok point to first head sub_tok
        if head_start != head_end:
            head_spans.append((head_start+1, head_end))
            sub_que_span.append(head_spans)

            for i in range(head_start+1, head_end+1):
                sub_que_span.append([(i, i)])
        else:
            sub_que_span.append(head_spans)

    #assert len(sub_que_span) == len(query_tokens)

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []

    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    doc_org_to_split_map = {}
    pre_tok_len = 0
    for idx, doc_token in enumerate(example.doc_tokens):
        sub_doc_tok = tokenizer.tokenize(doc_token)
        doc_org_to_split_map[idx] = (pre_tok_len, len(sub_doc_tok) + pre_tok_len - 1)
        pre_tok_len += len(sub_doc_tok)

    cnt_span = 0
    #for sen_idx, sen_span in enumerate(all_doc_span):
    #    for idx, (start_ix, end_ix) in enumerate(sen_span):
    #        assert (start_ix <= len(sen_span) and end_ix <= len(sen_span))
    #        cnt_span += 1

    #assert cnt_span == len(example.doc_tokens)

    sub_doc_span = []
    pre_sen_len = 0
    for sen_idx, sen_span in enumerate(all_doc_span):
        sen_offset = pre_sen_len
        pre_sen_len += len(sen_span)
        for idx, (start_ix, end_ix) in enumerate(sen_span):
            head_start, head_end = doc_org_to_split_map[sen_offset+idx]
            # sub_start_idx and sub_end_idx of children of head node
            head_spans = [(doc_org_to_split_map[sen_offset+start_ix-1][0], doc_org_to_split_map[sen_offset+end_ix-1][1])]
            # all other head sub_tok point to first head sub_tok
            if head_start != head_end:
                head_spans.append((head_start + 1, head_end))
                sub_doc_span.append(head_spans)

                for i in range(head_start + 1, head_end + 1):
                    sub_doc_span.append([(i, i)])
            else:
                sub_doc_span.append(head_spans)

    #assert len(sub_doc_span) == len(all_doc_tokens)

    # making masks
    que_span_mask = np.zeros((len(sub_que_span), len(sub_que_span)))
    for idx, span_list in enumerate(sub_que_span):
        for (start_ix, end_ix) in span_list:
            if start_ix != end_ix:
                que_span_mask[start_ix:end_ix + 1, idx] = 1

    doc_span_mask = np.zeros((len(sub_doc_span), len(sub_doc_span)))

    for idx, span_list in enumerate(sub_doc_span):
        for (start_ix, end_ix) in span_list:
            if start_ix != end_ix:
                doc_span_mask[start_ix:end_ix + 1, idx] = 1

    if len(query_tokens) > max_query_length:
        query_tokens = query_tokens[0:max_query_length]
        que_span_mask = que_span_mask[:max_query_length,:max_query_length]


    if is_training and not example.is_impossible:
        # Get start and end position
        start_position = example.start_position
        end_position = example.end_position

        # If the answer cannot be found in the text, then skip this example.
        actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)])
        cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
        if actual_text.find(cleaned_answer_text) == -1:
            logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
            return []

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
        )

    spans = []

    truncated_query = tokenizer.encode(
        example.question_text, add_special_tokens=False, truncation=True, max_length=max_query_length
    )

    # Tokenizers who insert 2 SEP tokens in-between <context> & <question> need to have special handling
    # in the way they compute mask of added tokens.
    tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
    sequence_added_tokens = (
        tokenizer.max_len - tokenizer.max_len_single_sentence + 1
        if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
        else tokenizer.max_len - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair

    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):

        # Define the side we want to truncate / pad and the text/pair sorting
        if tokenizer.padding_side == "right":
            texts = truncated_query
            pairs = span_doc_tokens
            truncation = TruncationStrategy.ONLY_SECOND.value
        else:
            texts = span_doc_tokens
            pairs = truncated_query
            truncation = TruncationStrategy.ONLY_FIRST.value

        encoded_dict = tokenizer.encode_plus(  # TODO(thom) update this logic
            texts,
            pairs,
            truncation=truncation,
            padding=padding_strategy,
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
            return_token_type_ids=True,
        )

        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
        )

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            if tokenizer.padding_side == "right":
                non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                last_padding_id_position = (
                    len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id)
                )
                non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1 :]

        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len

        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict or (
            "overflowing_tokens" in encoded_dict and len(encoded_dict["overflowing_tokens"]) == 0
        ):
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0)
        p_mask = np.ones_like(span["token_type_ids"])
        if tokenizer.padding_side == "right":
            p_mask[len(truncated_query) + sequence_added_tokens :] = 0
        else:
            p_mask[-len(span["tokens"]) : -(len(truncated_query) + sequence_added_tokens)] = 0

        pad_token_indices = np.where(span["input_ids"] == tokenizer.pad_token_id)
        special_token_indices = np.asarray(
            tokenizer.get_special_tokens_mask(span["input_ids"], already_has_special_tokens=True)
        ).nonzero()

        p_mask[pad_token_indices] = 1
        p_mask[special_token_indices] = 1

        # Set the cls index to 0: the CLS index can be used for impossible answers
        p_mask[cls_index] = 0

        #Input mask based on head span
        start_doc_ix = span["start"]
        end_doc_ix = span["start"] + span["length"] - 1
        select_doc_len = span['length']
        select_que_len = span['truncated_query_with_special_tokens_length'] - 2
        #assert len(head_select_idx) == select_doc_len

        input_span_mask = np.zeros((max_seq_length, max_seq_length))
        # 0 count for [CLS] and select_doc_len+1 count for [SEP]
        input_span_mask[1:select_que_len + 1, 1:select_que_len + 1] = que_span_mask
   #input_span_mask[1:select_doc_len + 1, 1:select_doc_len + 1] = doc_span_mask[start_doc_ix:end_doc_ix + 1,
    #                                                                    start_doc_ix:end_doc_ix + 1]
        input_span_mask[select_que_len + 2:select_que_len + select_doc_len + 2,
        select_que_len + 2:select_que_len + select_doc_len + 2] = doc_span_mask[start_doc_ix:end_doc_ix + 1, 
                                                                    start_doc_ix:end_doc_ix + 1]
        #input_span_mask[select_doc_len + 2:select_doc_len + select_que_len + 2,
        #select_doc_len + 2:select_doc_len + select_que_len + 2] = que_span_mask
        record_mask = []
        for i in range(max_seq_length):
            i_mask = []
            for j in range(max_seq_length):
                if input_span_mask[i, j] == 1:
                    i_mask.append(j)
            record_mask.append(i_mask)

        span_is_impossible = example.is_impossible
        orig_ans = example.orig_ans
        title = example.title
        start_position = 0
        end_position = 0
        if is_training and not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    doc_offset = len(truncated_query) + sequence_added_tokens

                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        features.append(
            SquadFeatures(
                span["input_ids"],
                span["attention_mask"],
                span["token_type_ids"],
                cls_index,
                p_mask.tolist(),
                example_index=0,  # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                start_position=start_position,
                end_position=end_position,
                is_impossible=span_is_impossible,
                orig_ans=orig_ans,
                title=title,
                input_span_mask=record_mask,
                qas_id=example.qas_id,
            )
        )
    return features


def squad_convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def squad_convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    is_training,
    padding_strategy="max_length",
    return_dataset=False,
    threads=1,
    tqdm_enabled=True,
):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model.
    It is model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.

    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        padding_strategy: Default to "max_length". Which padding strategy to use
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset,
            if 'tf': returns a tf.data.Dataset
        threads: multiple processing threadsa-smi


    Returns:
        list of :class:`~transformers.data.processors.squad.SquadFeatures`

    Example::

        processor = SquadV2Processor()
        examples = processor.get_dev_examples(data_dir)

        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
        )
    """

    # Defining helper methods
    features = []
    threads = min(threads, cpu_count())
    with Pool(threads, initializer=squad_convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            squad_convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            padding_strategy=padding_strategy,
            is_training=is_training,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert squad examples to features",
                disable=not tqdm_enabled,
            )
        )
    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(
        features, total=len(features), desc="add example index and unique id", disable=not tqdm_enabled
    ):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features
    if return_dataset == "pt":
        if not is_torch_available():
            raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.long)
        all_orig_answers = torch.tensor([f.orig_ans for f in features], dtype=torch.long)
        all_title = torch.tensor([int(f.title) for f in features], dtype=torch.long) #f.title
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

        if not is_training:
            all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, 
                all_attention_masks,
                all_token_type_ids, 
                all_feature_index, 
                all_cls_index, 
                all_p_mask, 
                all_title,
                all_example_index
            )
        else:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_start_positions,
                all_end_positions,
                all_cls_index,
                all_p_mask,
                all_is_impossible,
                all_orig_answers,
                all_title,
                all_example_index
            )

        return features, dataset
    elif return_dataset == "tf":
        if not is_tf_available():
            raise RuntimeError("TensorFlow must be installed to return a TensorFlow dataset.")

        def gen():
            for i, ex in enumerate(features):
                if ex.token_type_ids is None:
                    yield (
                        {
                            "input_ids": ex.input_ids,
                            "attention_mask": ex.attention_mask,
                            "feature_index": i,
                            "qas_id": ex.qas_id,
                        },
                        {
                            "start_positions": ex.start_position,
                            "end_positions": ex.end_position,
                            "cls_index": ex.cls_index,
                            "p_mask": ex.p_mask,
                            "is_impossible": ex.is_impossible,
                        },
                    )
                else:
                    yield (
                        {
                            "input_ids": ex.input_ids,
                            "attention_mask": ex.attention_mask,
                            "token_type_ids": ex.token_type_ids,
                            "feature_index": i,
                            "qas_id": ex.qas_id,
                        },
                        {
                            "start_positions": ex.start_position,
                            "end_positions": ex.end_position,
                            "cls_index": ex.cls_index,
                            "p_mask": ex.p_mask,
                            "is_impossible": ex.is_impossible,
                        },
                    )

        # Why have we split the batch into a tuple? PyTorch just has a list of tensors.
        if "token_type_ids" in tokenizer.model_input_names:
            train_types = (
                {
                    "input_ids": tf.int32,
                    "attention_mask": tf.int32,
                    "token_type_ids": tf.int32,
                    "feature_index": tf.int64,
                    "qas_id": tf.string,
                },
                {
                    "start_positions": tf.int64,
                    "end_positions": tf.int64,
                    "cls_index": tf.int64,
                    "p_mask": tf.int32,
                    "is_impossible": tf.int32,
                },
            )

            train_shapes = (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                    "feature_index": tf.TensorShape([]),
                    "qas_id": tf.TensorShape([]),
                },
                {
                    "start_positions": tf.TensorShape([]),
                    "end_positions": tf.TensorShape([]),
                    "cls_index": tf.TensorShape([]),
                    "p_mask": tf.TensorShape([None]),
                    "is_impossible": tf.TensorShape([]),
                },
            )
        else:
            train_types = (
                {"input_ids": tf.int32, "attention_mask": tf.int32, "feature_index": tf.int64, "qas_id": tf.string},
                {
                    "start_positions": tf.int64,
                    "end_positions": tf.int64,
                    "cls_index": tf.int64,
                    "p_mask": tf.int32,
                    "is_impossible": tf.int32,
                },
            )

            train_shapes = (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "feature_index": tf.TensorShape([]),
                    "qas_id": tf.TensorShape([]),
                },
                {
                    "start_positions": tf.TensorShape([]),
                    "end_positions": tf.TensorShape([]),
                    "cls_index": tf.TensorShape([]),
                    "p_mask": tf.TensorShape([None]),
                    "is_impossible": tf.TensorShape([]),
                },
            )

        return tf.data.Dataset.from_generator(gen, train_types, train_shapes)
    else:
        return features


class SquadProcessor(DataProcessor):
    """
    Processor for the SQuAD data set.
    Overriden by SquadV1Processor and SquadV2Processor, used by the version 1.1 and version 2.0 of SQuAD, respectively.
    """

    train_file = None
    dev_file = None

    def _get_example_from_tensor_dict(self, tensor_dict, evaluate=False):
        if not evaluate:
            answer = tensor_dict["answers"]["text"][0].numpy().decode("utf-8")
            answer_start = tensor_dict["answers"]["answer_start"][0].numpy()
            answers = []
        else:
            answers = [
                {"answer_start": start.numpy(), "text": text.numpy().decode("utf-8")}
                for start, text in zip(tensor_dict["answers"]["answer_start"], tensor_dict["answers"]["text"])
            ]

            answer = None
            answer_start = None

        return SquadExample(
            qas_id=tensor_dict["id"].numpy().decode("utf-8"),
            question_text=tensor_dict["question"].numpy().decode("utf-8"),
            context_text=tensor_dict["context"].numpy().decode("utf-8"),
            answer_text=answer,
            start_position_character=answer_start,
            title=tensor_dict["title"].numpy().decode("utf-8"),
            answers=answers,
        )

    def get_examples_from_dataset(self, dataset, evaluate=False):
        """
        Creates a list of :class:`~transformers.data.processors.squad.SquadExample` using a TFDS dataset.

        Args:
            dataset: The tfds dataset loaded from `tensorflow_datasets.load("squad")`
            evaluate: boolean specifying if in evaluation mode or in training mode

        Returns:
            List of SquadExample

        Examples::

            >>> import tensorflow_datasets as tfds
            >>> dataset = tfds.load("squad")

            >>> training_examples = get_examples_from_dataset(dataset, evaluate=False)
            >>> evaluation_examples = get_examples_from_dataset(dataset, evaluate=True)
        """

        if evaluate:
            dataset = dataset["validation"]
        else:
            dataset = dataset["train"]

        examples = []
        for tensor_dict in tqdm(dataset):
            examples.append(self._get_example_from_tensor_dict(tensor_dict, evaluate=evaluate))

        return examples

    def get_train_examples(self, data_dir, filename=None, input_tag_file=None, to_ix_dict=None):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.

        """
        if data_dir is None:
            data_dir = ""

        if self.train_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(
            os.path.join(data_dir, self.train_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]

        input_tag_data = []
        with open(input_tag_file, "r", encoding='utf-8') as reader:
            for line in reader:
                input_tag_data.append(json.loads(line))

        return self._create_examples(input_data, input_tag_data, "train", to_ix_dict)

    def get_dev_examples(self, data_dir, filename=None, input_tag_file=None, to_ix_dict=None):
        """
        Returns the evaluation example from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the evaluation file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.
        """
        if data_dir is None:
            data_dir = ""

        if self.dev_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(
            os.path.join(data_dir, self.dev_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]

        input_tag_data = []
        with open(input_tag_file, "r", encoding='utf-8') as reader:
            for line in reader:
                input_tag_data.append(json.loads(line))

        return self._create_examples(input_data, input_tag_data, "dev", to_ix_dict)

    def _create_examples(self, input_data, input_tag_data, set_type, to_ix_dict):
        simple_nlp = SimpleNlp()
        is_training = set_type == "train"

        qas_id_to_tag_idx_map = {}
        all_dqtag_data = []
        for idx, tag_data in enumerate(tqdm(input_tag_data,ncols=50)):
            qas_id = tag_data["qas_id"]
            qas_id_to_tag_idx_map[qas_id] = idx
            tag_rep = tag_data["tag_rep"]
            dqtag_data = {
                "qas_id": qas_id,
                #"head_que": [int(i) for i in tag_rep["pred_head_que"]],
                "span_que": [eval(i) for i in tag_rep["hpsg_list_que"]],
                #"type_que": tag_rep["pred_type_que"],
                "span_doc": [eval(i) for sen_span in tag_rep["hpsg_list_doc"] for i in sen_span],
                #"type_doc": [i for sen in tag_rep["pred_type_doc"] for i in sen],
                #"head_doc": [int(i) for sen_head in tag_rep["pred_head_doc"] for i in sen_head],
                "token_doc": [token for sen_token in tag_rep['doc_tokens'] for token in sen_token],
                "token_que": tag_rep['que_tokens']
            }
            all_dqtag_data.append(dqtag_data)

        examples = []
        for entry in tqdm(input_data, desc='span mask creation'):
            title = entry["title"]
            for paragraph in entry["paragraphs"]:
                context_text = paragraph["context"]

                sen_texts = simple_nlp.nlp(context_text)
                sen_list = []

                for sen_ix, sent in enumerate(sen_texts.sents):
                    sent_tokens = []
                    prev_is_whitespace = True
                    for c in sent.string:
                        if _is_whitespace(c):
                            prev_is_whitespace = True
                        else:
                            if prev_is_whitespace:
                                sent_tokens.append(c)
                            else:
                                sent_tokens[-1] += c
                            prev_is_whitespace = False
                    sen_list.append((sen_ix, sent_tokens))

                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in context_text:
                    if _is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                cnt_token = 0
                new_sen_list = []
                flag = False
                tmp_token = ""
                for sen_ix, sent_tokens in sen_list:
                    new_sent_tokens = []
                    for tok_ix, token in enumerate(sent_tokens):
                        if tok_ix == 0 and flag:
                            token = tmp_token+token
                            flag = False
                            tmp_token = ""
                            assert token == doc_tokens[cnt_token]

                        if token != doc_tokens[cnt_token]:
                            assert tok_ix == len(sent_tokens) - 1
                            tmp_token = token
                            flag = True
                        else:
                            assert token == doc_tokens[cnt_token]
                            new_sent_tokens.append(token)

                            cnt_token += 1
                    new_sen_list.append(new_sent_tokens)

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    dqtag = all_dqtag_data[qas_id_to_tag_idx_map[qas_id]]
                    assert dqtag["qas_id"] == qas_id

                    span_doc = dqtag["span_doc"]
                    #head_doc = dqtag["head_doc"]
                    #type_doc = dqtag["type_doc"]
                    #assert len(span_doc) == len(head_doc) == len(type_doc) == cnt_token, qas_id
                    # reconstruct into sentences
                    new_span_doc = []
                    #new_head_doc = []
                    #new_type_doc = []
                    cnt = 0
                    for sent_tokens in new_sen_list:
                        new_span_sen = []
                        #new_head_sen = []
                        #new_type_sen = []
                        for _ in sent_tokens:
                            new_span_sen.append(span_doc[cnt])
                            #new_head_sen.append(head_doc[cnt])
                            #new_type_sen.append(type_doc[cnt])
                            cnt += 1
                        new_span_doc.append(new_span_sen)
                        #new_head_doc.append(new_head_sen)
                        #new_type_doc.append(new_type_sen)
                        
                    question_text = qa["question"]
                    start_position_character = None
                    answer_text = None
                    answers = []

                    is_impossible = qa.get("is_impossible", False)
                    orig_ans = to_ix_dict.get(qa.get("orig_ans", "unknown"), 0)
                    if not is_impossible:
                        if is_training:
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            start_position_character = answer["answer_start"]
                        else:
                            answers = qa["answers"]

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        orig_ans=orig_ans,
                        answers=answers,
                        que_span=dqtag["span_que"],
                        token_que=dqtag["token_que"],
                        doc_span=new_span_doc,
                        token_doc=dqtag["token_doc"],
                    )
                    examples.append(example)
        return examples

    def create_dicts(self, data_dir, filename=None):
        '''
        Return Answer to ID and ID to Answer dictionaries
        '''
        with open(
            os.path.join(data_dir, "ans_to_ix.json"), "r", encoding="utf-8"
        ) as reader:
            ans_to_ix = json.load(reader)

        with open(
            os.path.join(data_dir, "ix_to_ans.json"), "r", encoding="utf-8"
        ) as reader:
            ix_to_ans = json.load(reader)

        return ans_to_ix, ix_to_ans

    def load_scene_graph(self, data_dir, filename=None):
        '''
        Return Answer to ID and ID to Answer dictionaries
        '''
        with open(
            os.path.join(data_dir, "vg_scene_graph.json"), "r", encoding="utf-8"
        ) as reader:
            sceneDict = json.load(reader)

        return sceneDict



class SquadV1Processor(SquadProcessor):
    train_file = "train-v1.1.json"
    dev_file = "dev-v1.1.json"


class SquadV2Processor(SquadProcessor):
    train_file = "train-v2.0.json"
    dev_file = "dev-v2.0.json"


class SquadExample:
    """
    A single training/test example for the Squad dataset, as loaded from disk.

    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
        self,
        qas_id,
        question_text,
        context_text,
        answer_text,
        start_position_character,
        title,
        answers=[],
        is_impossible=False,
        orig_ans=0,
        que_span=None,
        token_que=None,
        doc_span=None,
        token_doc=None,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers
        self.orig_ans = orig_ans
        self.que_span = que_span
        self.token_que = token_que
        self.doc_span = doc_span
        self.token_doc = token_doc

        self.start_position, self.end_position = 0, 0

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start and end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]


class SquadFeatures:
    """
    Single squad example features to be fed to a model.
    Those features are model-specific and can be crafted from :class:`~transformers.data.processors.squad.SquadExample`
    using the :method:`~transformers.data.processors.squad.squad_convert_examples_to_features` method.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        cls_index: the index of the CLS token.
        p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
            Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
        example_index: the index of the example
        unique_id: The unique Feature identifier
        paragraph_len: The length of the context
        token_is_max_context: List of booleans identifying which tokens have their maximum context in this feature object.
            If a token does not have their maximum context in this feature object, it means that another feature object
            has more information related to that token and should be prioritized over this feature for that token.
        tokens: list of tokens corresponding to the input ids
        token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.
        start_position: start of the answer token index
        end_position: end of the answer token index
    """

    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        cls_index,
        p_mask,
        example_index,
        unique_id,
        paragraph_len,
        token_is_max_context,
        tokens,
        token_to_orig_map,
        start_position,
        end_position,
        is_impossible,
        orig_ans,
        title,
        input_span_mask=None,
        qas_id: str = None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.orig_ans = orig_ans
        self.title = title
        self.input_span_mask = input_span_mask
        self.qas_id = qas_id


class SquadResult:
    """
    Constructs a SquadResult which can be used to evaluate a model's output on the SQuAD dataset.

    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(self, unique_id, start_logits, end_logits, choice_logits=None, start_top_index=None, end_top_index=None, cls_logits=None):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.unique_id = unique_id
        if choice_logits:
            self.choice_logits = choice_logits

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
            self.cls_logits = cls_logits


class SymbolDict(object):
    def __init__(self, empty=False):
        self.padding = "<PAD>"
        self.unknown = "<UNK>"
        self.start = "<START>"
        self.end = "<END>"

        self.invalidSymbols = [self.padding, self.unknown, self.start, self.end]

        if empty:
            self.sym2id = {self.padding: 0}
            self.id2sym = [self.padding]
        else:
            self.sym2id = {self.padding: 0, self.unknown: 1, self.start: 2, self.end: 3}
            self.id2sym = [self.padding, self.unknown, self.start, self.end]
        self.allSeqs = []

    def getNumSymbols(self):
        return len(self.sym2id)

    def isValid(self, enc):
        return enc not in self.invalidSymbols

    def resetSeqs(self):
        self.allSeqs = []

    def addSymbols(self, seq):
        if type(seq) is not list:
            seq = [seq]
        self.allSeqs += seq

    # Call to create the words-to-integers vocabulary after (reading word sequences with addSymbols). 
    def addToVocab(self, symbol):
        if symbol not in self.sym2id:
            self.sym2id[symbol] = self.getNumSymbols()
            self.id2sym.append(symbol)

    # create vocab only if not existing..?
    def createVocab(self, minCount=0, top=0, addUnk=False, weights=False):
        counter = {}
        for symbol in self.allSeqs:
            counter[symbol] = counter.get(symbol, 0) + 1

        isTop = lambda symbol: True
        if top > 0:
            topItems = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:top]
            tops = [k for k, v in topItems]
            isTop = lambda symbol: symbol in tops

        if addUnk:
            self.addToVocab(self.unknown)

        for symbol in counter:
            if counter[symbol] > minCount and isTop(symbol):
                self.addToVocab(symbol)

        self.counter = counter

        self.counts = np.array([counter.get(sym, 0) for sym in self.id2sym])

        if weights:
            self.weights = np.array([1.0 for sym in self.id2sym])
            if config.ansWeighting:
                weight = lambda w: 1.0 / float(w) if w > 0 else 0.0
                self.weights = np.array([weight(counter.get(sym, 0)) for sym in self.id2sym])
                totalWeight = np.sum(self.weights)
                self.weights /= totalWeight
                self.weights *= len(self.id2sym)
            elif config.ansWeightingRoot:
                weight = lambda w: 1.0 / math.sqrt(float(w)) if w > 0 else 0
                self.weights = np.array([weight(counter.get(sym, 0)) for sym in self.id2sym])
                totalWeight = np.sum(self.weights)
                self.weights /= totalWeight
                self.weights *= len(self.id2sym)

    # Encodes a symbol. Returns the matching integer.
    def encodeSym(self, symbol):
        if symbol not in self.sym2id:
            symbol = self.unknown
        return self.sym2id[symbol]  # self.sym2id.get(symbol, None) # # -1 VQA MAKE SURE IT DOESNT CAUSE BUGS

    '''
    Encodes a sequence of symbols.
    Optionally add start, or end symbols. 
    Optionally reverse sequence 
    '''

    def encodeSeq(self, decoded, addStart=False, addEnd=False, reverse=False):
        if reverse:
            decoded.reverse()
        if addStart:
            decoded = [self.start] + decoded
        if addEnd:
            decoded = decoded + [self.end]
        encoded = [self.encodeSym(symbol) for symbol in decoded]
        return encoded

    # Decodes an integer into its symbol 
    def decodeId(self, enc):
        return self.id2sym[enc] if enc < self.getNumSymbols() else self.unknown

    '''
    Decodes a sequence of integers into their symbols.
    If delim is given, joins the symbols using delim,
    Optionally reverse the resulted sequence 
    '''

    def decodeSeq(self, encoded, delim=None, reverse=False, stopAtInvalid=True):
        length = 0
        for i in range(len(encoded)):
            if not self.isValid(self.decodeId(encoded[i])) and stopAtInvalid:
                # if not self.isValid(encoded[i]) and stopAtInvalid:
                break
            length += 1
        encoded = encoded[:length]

        decoded = [self.decodeId(enc) for enc in encoded]
        if reverse:
            decoded.reverse()

        if delim is not None:
            return delim.join(decoded)

        return decoded

def initEmbRandom(num, dim):
    # uniform initialization
    
    lowInit = -1.0
    highInit = 1.0
    embeddings = np.random.uniform(low=lowInit, high=highInit,
                                    size=(num, dim))
    return embeddings

def sentenceEmb(sentence, wordVectors, dim):
        words = sentence.split(" ")
        wordEmbs = initEmbRandom(len(words), dim)
        for idx, word in enumerate(words):
            if word in wordVectors:
                wordEmbs[idx] = wordVectors[word]
        sentenceEmb = np.mean(wordEmbs, axis=0)
        return sentenceEmb
        
def initializeWordEmbeddings(dim, wordsDict=None, random=False, filename=None):

    embeddings = initEmbRandom(wordsDict.getNumSymbols(), dim)

    # if wrdEmbRandom = False, use GloVE
    counter = 0
    if not random:
        wordVectors = {}
        with open(filename, 'r') as inFile:  
            for line in tqdm(inFile):
                line = line.strip().split()
                word = line[0].lower()
                vector = np.array([float(x) for x in line[1:]])
                wordVectors[word] = vector

            for sym, idx in tqdm(wordsDict.sym2id.items()):
                if " " in sym:
                    symEmb = sentenceEmb(sym, wordVectors, 300)
                    embeddings[idx] = symEmb
                else:
                    if sym in wordVectors:
                        embeddings[idx] = wordVectors[sym]
                        counter += 1


    return embeddings