""" Very heavily inspired by the official evaluation script for SQuAD version 2.0 which was
modified by XLNet authors to update `find_best_threshold` scripts for SQuAD V2.0

In addition to basic functionality, we also compute additional statistics and
plot precision-recall curves if an additional na_prob.json file is provided.
This file is expected to map question ID's to the model's predicted probability
that a question is unanswerable.
"""


import collections
import json
import logging
import math
import re
import string

from transformers.tokenization_bert import BasicTokenizer


logger = logging.getLogger(__name__)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_raw_scores(examples, preds):
    """
    Computes the exact and f1 scores from the examples and the model predictions
    """
    exact_scores = {}
    f1_scores = {}

    for example in examples:
        qas_id = example.qas_id
        gold_answers = [answer["text"] for answer in example.answers if normalize_answer(answer["text"])]

        if not gold_answers:
            # For unanswerable questions, only correct answer is empty string
            gold_answers = [""]

        if qas_id not in preds:
            print("Missing prediction for %s" % qas_id)
            continue

        prediction = preds[qas_id]
        exact_scores[qas_id] = max(compute_exact(a, prediction) for a in gold_answers)
        f1_scores[qas_id] = max(compute_f1(a, prediction) for a in gold_answers)

    return exact_scores, f1_scores


def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    new_scores = {}
    for qid, s in scores.items():
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            new_scores[qid] = s
    return new_scores


def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores.values()) / total),
                ("f1", 100.0 * sum(f1_scores.values()) / total),
                ("total", total),
            ]
        )
    else:
        total = len(qid_list)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
                ("f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
                ("total", total),
            ]
        )


def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval["%s_%s" % (prefix, k)] = new_eval[k]


def find_best_thresh_v2(preds, scores, na_probs, qid_to_has_ans):
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for i, qid in enumerate(qid_list):
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            if preds[qid]:
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]

    has_ans_score, has_ans_cnt = 0, 0
    for qid in qid_list:
        if not qid_to_has_ans[qid]:
            continue
        has_ans_cnt += 1

        if qid not in scores:
            continue
        has_ans_score += scores[qid]

    return 100.0 * best_score / len(scores), best_thresh, 1.0 * has_ans_score / has_ans_cnt


def find_all_best_thresh_v2(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    best_exact, exact_thresh, has_ans_exact = find_best_thresh_v2(preds, exact_raw, na_probs, qid_to_has_ans)
    best_f1, f1_thresh, has_ans_f1 = find_best_thresh_v2(preds, f1_raw, na_probs, qid_to_has_ans)
    main_eval["best_exact"] = best_exact
    main_eval["best_exact_thresh"] = exact_thresh
    main_eval["best_f1"] = best_f1
    main_eval["best_f1_thresh"] = f1_thresh
    main_eval["has_ans_exact"] = has_ans_exact
    main_eval["has_ans_f1"] = has_ans_f1


def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for _, qid in enumerate(qid_list):
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            if preds[qid]:
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]
    return 100.0 * best_score / len(scores), best_thresh


def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    best_exact, exact_thresh = find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
    best_f1, f1_thresh = find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)

    main_eval["best_exact"] = best_exact
    main_eval["best_exact_thresh"] = exact_thresh
    main_eval["best_f1"] = best_f1
    main_eval["best_f1_thresh"] = f1_thresh


def squad_evaluate(examples, preds, no_answer_probs=None, no_answer_probability_threshold=1.0):
    qas_id_to_has_answer = {example.qas_id: bool(example.answers) for example in examples}
    has_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if has_answer]
    no_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if not has_answer]

    if no_answer_probs is None:
        no_answer_probs = {k: 0.0 for k in preds}

    exact, f1 = get_raw_scores(examples, preds)

    exact_threshold = apply_no_ans_threshold(
        exact, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold
    )
    f1_threshold = apply_no_ans_threshold(f1, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold)

    evaluation = make_eval_dict(exact_threshold, f1_threshold)

    if has_answer_qids:
        has_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=has_answer_qids)
        merge_eval(evaluation, has_ans_eval, "HasAns")

    if no_answer_qids:
        no_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=no_answer_qids)
        merge_eval(evaluation, no_ans_eval, "NoAns")

    if no_answer_probs:
        find_all_best_thresh(evaluation, preds, exact, f1, no_answer_probs, qas_id_to_has_answer)

    return evaluation


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'", orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position : (orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def compute_predictions_logits(
    all_examples,
    all_features,
    all_results,
    n_best_size,
    max_answer_length,
    do_lower_case,
    output_prediction_file,
    output_nbest_file,
    output_null_log_odds_file,
    verbose_logging,
    version_2_with_negative,
    null_score_diff_threshold,
    tokenizer,
):
    """Write final predictions to the json file and log-odds of null if needed."""
    if output_prediction_file:
        logger.info(f"Writing predictions to: {output_prediction_file}")
    if output_nbest_file:
        logger.info(f"Writing nbest to: {output_nbest_file}")
    if output_null_log_odds_file and version_2_with_negative:
        logger.info(f"Writing null_log_odds to: {output_null_log_odds_file}")

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"]
    )

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                        )
                    )
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                )
            )
        prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"]
        )

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index : (pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start : (orig_doc_end + 1)]

                tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

                # tok_text = " ".join(tok_tokens)
                #
                # # De-tokenize WordPieces that have been split off.
                # tok_text = tok_text.replace(" ##", "")
                # tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(_NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(_NbestPrediction(text="", start_logit=null_start_logit, end_logit=null_end_logit))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1, "No valid predictions"

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1, "No valid predictions"

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json

    if output_prediction_file:
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")

    if output_nbest_file:
        with open(output_nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if output_null_log_odds_file and version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions

def compute_predictions_logits_vqa(
    all_examples,
    all_features,
    all_results,
    n_best_size,
    max_answer_length,
    do_lower_case,
    output_prediction_file,
    output_nbest_file,
    output_null_log_odds_file,
    output_nbest_choice_file,
    verbose_logging,
    version_2_with_negative,
    null_score_diff_threshold,
    tokenizer,
    ix_to_ans_dict,
):
    """Write final predictions to the json file and log-odds of null if needed."""
    if output_prediction_file:
        logger.info(f"Writing predictions to: {output_prediction_file}")
    if output_nbest_file:
        logger.info(f"Writing nbest to: {output_nbest_file}")
    if output_null_log_odds_file and version_2_with_negative:
        logger.info(f"Writing null_log_odds to: {output_null_log_odds_file}")

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"]
    )

    _PrelimPrediction_choice = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction_choice", ["feature_index", "choice_index", "choice_logit"]
    )

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()
    all_nbest_choice_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        prelim_choices = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            choice_indexes = _get_best_indexes(result.choice_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for choice_index in choice_indexes:
                prelim_choices.append(
                        _PrelimPrediction_choice(
                            feature_index=feature_index,
                            choice_index=choice_index,
                            choice_logit=result.choice_logits[choice_index],
                        )
                    )
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                        )
                    )
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                )
            )
        prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)
        prelim_choices = sorted(prelim_choices, key=lambda x: (x.choice_logit), reverse=True)

        #Choice calculations
        _NbestPrediction_choice = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction_choice", ["text", "choice_logit"]
        )

        nbest_choice = []
        for pred in prelim_choices:
            nbest_choice.append(_NbestPrediction_choice(text=ix_to_ans_dict.get(str(pred.choice_index)), choice_logit=pred.choice_logit))
        total_choice_scores = []
        best_choice_entry = None
        for entry in nbest_choice:
            total_choice_scores.append(entry.choice_logit)
            if not best_choice_entry:
                if entry.text:
                    best_choice_entry = entry

        choice_probs = _compute_softmax(total_choice_scores)

        nbest_choice_json = []
        for (i, entry) in enumerate(nbest_choice):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = choice_probs[i]
            output["choice_logit"] = entry.choice_logit
            nbest_choice_json.append(output)

        all_nbest_choice_json[example.qas_id] = nbest_choice_json

        #Span calculations
        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"]
        )

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index : (pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start : (orig_doc_end + 1)]

                tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

                # tok_text = " ".join(tok_tokens)
                #
                # # De-tokenize WordPieces that have been split off.
                # tok_text = tok_text.replace(" ##", "")
                # tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(_NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(_NbestPrediction(text="", start_logit=null_start_logit, end_logit=null_end_logit))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1, "No valid predictions"

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1, "No valid predictions"

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                #all_predictions[example.qas_id] = ""
                all_predictions[example.qas_id] = remove_plural(best_choice_entry.text)
            else:
                if nbest_json[0]["text"] != "" and nbest_json[0]["probability"] > nbest_choice_json[0]["probability"]:
                    all_predictions[example.qas_id] = remove_plural(best_non_null_entry.text)
                else:
                    all_predictions[example.qas_id] = remove_plural(best_choice_entry.text)
        all_nbest_json[example.qas_id] = nbest_json

    if output_prediction_file:
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")

    if output_nbest_file:
        with open(output_nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if output_nbest_choice_file:
        with open(output_nbest_choice_file, "w") as writer:
            writer.write(json.dumps(all_nbest_choice_json, indent=4) + "\n")

    if output_null_log_odds_file and version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions

def compute_predictions_log_probs(
    all_examples,
    all_features,
    all_results,
    n_best_size,
    max_answer_length,
    output_prediction_file,
    output_nbest_file,
    output_null_log_odds_file,
    start_n_top,
    end_n_top,
    version_2_with_negative,
    tokenizer,
    verbose_logging,
):
    """ XLNet write prediction logic (more complex than Bert's).
        Write final predictions to the json file and log-odds of null if needed.

        Requires utils_squad_evaluate.py
    """
    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_log_prob", "end_log_prob"]
    )

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_log_prob", "end_log_prob"]
    )

    logger.info("Writing predictions to: %s", output_prediction_file)
    # logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive

        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            cur_null_score = result.cls_logits

            # if we could have irrelevant answers, get the min score of irrelevant
            score_null = min(score_null, cur_null_score)

            for i in range(start_n_top):
                for j in range(end_n_top):
                    start_log_prob = result.start_logits[i]
                    start_index = result.start_top_index[i]

                    j_index = i * end_n_top + j

                    end_log_prob = result.end_logits[j_index]
                    end_index = result.end_top_index[j_index]

                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= feature.paragraph_len - 1:
                        continue
                    if end_index >= feature.paragraph_len - 1:
                        continue

                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue

                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_log_prob=start_log_prob,
                            end_log_prob=end_log_prob,
                        )
                    )

        prelim_predictions = sorted(
            prelim_predictions, key=lambda x: (x.start_log_prob + x.end_log_prob), reverse=True
        )

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            # XLNet un-tokenizer
            # Let's keep it simple for now and see if we need all this later.
            #
            # tok_start_to_orig_index = feature.tok_start_to_orig_index
            # tok_end_to_orig_index = feature.tok_end_to_orig_index
            # start_orig_pos = tok_start_to_orig_index[pred.start_index]
            # end_orig_pos = tok_end_to_orig_index[pred.end_index]
            # paragraph_text = example.paragraph_text
            # final_text = paragraph_text[start_orig_pos: end_orig_pos + 1].strip()

            # Previously used Bert untokenizer
            tok_tokens = feature.tokens[pred.start_index : (pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start : (orig_doc_end + 1)]
            tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            if hasattr(tokenizer, "do_lower_case"):
                do_lower_case = tokenizer.do_lower_case
            else:
                do_lower_case = tokenizer.do_lowercase_and_remove_accent

            final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)

            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(text=final_text, start_log_prob=pred.start_log_prob, end_log_prob=pred.end_log_prob)
            )

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="", start_log_prob=-1e6, end_log_prob=-1e6))

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_log_prob + entry.end_log_prob)
            if not best_non_null_entry:
                best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_log_prob"] = entry.start_log_prob
            output["end_log_prob"] = entry.end_log_prob
            nbest_json.append(output)

        assert len(nbest_json) >= 1, "No valid predictions"
        assert best_non_null_entry is not None, "No valid predictions"

        score_diff = score_null
        scores_diff_json[example.qas_id] = score_diff
        # note(zhiliny): always predict best_non_null_entry
        # and the evaluation script will search for the best threshold
        all_predictions[example.qas_id] = best_non_null_entry.text

        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions


def remove_plural(prediction_text):

    replace_dict = {'advertisements': 'advertisement',
    'airplanes': 'airplane',
    'animals': 'animal',
    'apples': 'apple',
    'arches': 'arch',
    'arms': 'arm',
    'arrows': 'arrow',
    'backpacks': 'backpack',
    'bagels': 'bagel',
    'bags': 'bag',
    'balconies': 'balcony',
    'balloons': 'balloon',
    'balls': 'ball',
    'bananas': 'banana',
    'bars': 'bar',
    'baskets': 'basket',
    'beans': 'bean',
    'bears': 'bear',
    'beds': 'bed',
    'benches': 'bench',
    'berries': 'berry',
    'bicycles': 'bicycle',
    'bikers': 'biker',
    'bikes': 'bike',
    'birds': 'bird',
    'blankets': 'blanket',
    'bleachers': 'bleacher',
    'blinds': 'blind',
    'boards': 'board',
    'boats': 'boat',
    'bolts': 'bolt',
    'books': 'book',
    'boots': 'boot',
    'bottles': 'bottle',
    'boulders': 'boulder',
    'bowls': 'bowl',
    'boxes': 'box',
    'boys': 'boy',
    'bracelets': 'bracelet',
    'branches': 'branch',
    'bricks': 'brick',
    'bubbles': 'bubble',
    'buckets': 'bucket',
    'buildings': 'building',
    'bulls': 'bull',
    'bunches': 'bunch',
    'buns': 'bun',
    'buses': 'bus',
    'bushes': 'bush',
    'buttons': 'button',
    'cabinets': 'cabinet',
    'cables': 'cable',
    'cakes': 'cake',
    'candles': 'candle',
    'caps': 'cap',
    'carrots': 'carrot',
    'cars': 'car',
    'cats': 'cat',
    'chains': 'chain',
    'chairs': 'chair',
    'cherries': 'cherry',
    'children': 'child',
    'chips': 'chip',
    'chopsticks': 'chopstick',
    'circles': 'circle',
    'claws': 'claw',
    'cleats': 'cleat',
    'clocks': 'clock',
    'clouds': 'cloud',
    'coats': 'coat',
    'computers': 'computer',
    'condiments': 'condiment',
    'cones': 'cone',
    'containers': 'container',
    'controllers': 'controller',
    'cookies': 'cookie',
    'cords': 'cord',
    'cows': 'cow',
    'cracks': 'crack',
    'crates': 'crate',
    'crumbs': 'crumb',
    'cucumbers': 'cucumber',
    'cupcakes': 'cupcake',
    'cups': 'cup',
    'curtains': 'curtain',
    'customers': 'customer',
    'decorations': 'decoration',
    'desserts': 'dessert',
    'diamonds': 'diamond',
    'directions': 'direction',
    'dishes': 'dish',
    'dogs': 'dog',
    'donuts': 'donut',
    'doors': 'door',
    'dots': 'dot',
    'doughnuts': 'doughnut',
    'drapes': 'drape',
    'drawers': 'drawer',
    'dresses': 'dress',
    'drinks': 'drink',
    'drivers': 'driver',
    'ducks': 'duck',
    'earphones': 'earphone',
    'earrings': 'earring',
    'ears': 'ear',
    'eggs': 'egg',
    'elephants': 'elephant',
    'engines': 'engine',
    'evergreens': 'evergreen',
    'eyeglasses': 'eyeglass',
    'eyes': 'eye',
    'faces': 'face',
    'fans': 'fan',
    'feathers': 'feather',
    'feet': 'foot',
    'fences': 'fence',
    'fingers': 'finger',
    'firefighters': 'firefighter',
    'firemen': 'fireman',
    'fixtures': 'fixture',
    'flags': 'flag',
    'flames': 'flame',
    'floors': 'floor',
    'flops': 'flop',
    'flowers': 'flower',
    'footprints': 'footprint',
    'forks': 'fork',
    'frisbees': 'frisbee',
    'fruits': 'fruit',
    'games': 'game',
    'giants': 'giant',
    'giraffes': 'giraffe',
    'girls': 'girl',
    'glasses': 'glass',
    'gloves': 'glove',
    'goats': 'goat',
    'goggles': 'goggle',
    'grapes': 'grape',
    'greens': 'green',
    'handlebars': 'handlebar',
    'hands': 'hand',
    'hats': 'hat',
    'headlights': 'headlight',
    'headphones': 'headphone',
    'heads': 'head',
    'hearts': 'heart',
    'hedges': 'hedge',
    'helmets': 'helmet',
    'herbs': 'herb',
    'hills': 'hill',
    'holes': 'hole',
    'homes': 'home',
    'hooves': 'hoove',
    'horns': 'horn',
    'horses': 'horse',
    'hotdogs': 'hotdog',
    'hours': 'hour',
    'houses': 'house',
    'humans': 'human',
    'jackets': 'jacket',
    'jets': 'jet',
    'keys': 'key',
    'kids': 'kid',
    'kites': 'kite',
    'knees': 'knee',
    'knives': 'knife',
    'knobs': 'knob',
    'lambs': 'lamb',
    'lamps': 'lamp',
    'laptops': 'laptop',
    'leaves': 'leaf',
    'legos': 'lego',
    'legs': 'leg',
    'lemons': 'lemon',
    'letters': 'letter',
    'lights': 'light',
    'lines': 'line',
    'logs': 'log',
    'magazines': 'magazine',
    'magnets': 'magnet',
    'males': 'male',
    'men': 'man',
    'meters': 'meter',
    'mirrors': 'mirror',
    'motorbikes': 'motorbike',
    'motorcycles': 'motorbike',
    'mountains': 'mountain',
    'muffins': 'muffin',
    'mushrooms': 'mushroom',
    'nails': 'nail',
    'napkins': 'napkin',
    'noodles': 'noodle',
    'numbers': 'number',
    'numerals': 'numeral',
    'nuts': 'nut',
    'oars': 'oar',
    'officers': 'officer',
    'olives': 'olive',
    'onions': 'onion',
    'oranges': 'orange',
    'overalls': 'overall',
    'pads': 'pad',
    'paintings': 'painting',
    'pajamas': 'pajama',
    'pancakes': 'pancake',
    'passengers': 'passenger',
    'pastries': 'pastry',
    'paws': 'paw',
    'pebbles': 'pebble',
    'pedestrians': 'pedestrian',
    'pens': 'pen',
    'peppers': 'pepper',
    'phones': 'phone',
    'photos': 'photo',
    'pickles': 'pickle',
    'pictures': 'picture',
    'pigeons': 'pigeon',
    'pillars': 'pillar',
    'pillows': 'pillow',
    'pilots': 'pilot',
    'pipes': 'pipe',
    'pizzas': 'pizza',
    'planes': 'plane',
    'plants': 'plant',
    'plates': 'plate',
    'players': 'player',
    'pockets': 'pocket',
    'poles': 'pole',
    'posters': 'poster',
    'posts': 'post',
    'potatoes': 'potatoe',
    'pots': 'pot',
    'powerlines': 'powerline',
    'propellers': 'propeller',
    'rackets': 'racket',
    'railings': 'railing',
    'rectangles': 'rectangle',
    'reflections': 'reflection',
    'reins': 'rein',
    'remotes': 'remote',
    'riders': 'rider',
    'rings': 'ring',
    'ripples': 'ripple',
    'rocks': 'rock',
    'ropes': 'rope',
    'roses': 'rose',
    'rows': 'row',
    'saddles': 'saddle',
    'sailboats': 'sailboat',
    'sails': 'sail',
    'sandals': 'sandal',
    'sandwiches': 'sandwich',
    'screens': 'screen',
    'screws': 'screw',
    'seagulls': 'seagull',
    'seats': 'seat',
    'seeds': 'seed',
    'shades': 'shade',
    'shadows': 'shadow',
    'sheets': 'sheet',
    'shelves': 'shelf',
    'shingles': 'shingle',
    'shirts': 'shirt',
    'shoes': 'shoe',
    'shorts': 'short',
    'shoulders': 'shoulder',
    'shrubs': 'shrub',
    'shutters': 'shutter',
    'sides': 'side',
    'sideways': 'sideway',
    'signs': 'sign',
    'sinks': 'sink',
    'skateboarders': 'skater',
    'skateboards': 'skateboard',
    'skaters': 'skater',
    'skiers': 'skier',
    'skies': 'sky',
    'skyscrapers': 'skyscraper',
    'sleeves': 'sleeve',
    'slices': 'slice',
    'slopes': 'slope',
    'sneakers': 'sneaker',
    'snowboards': 'snowboard',
    'snowflakes': 'snowflake',
    'socks': 'sock',
    'speakers': 'speaker',
    'spectators': 'spectator',
    'spices': 'spice',
    'spokes': 'spoke',
    'spoons': 'spoon',
    'spots': 'spot',
    'sprinkles': 'sprinkle',
    'squares': 'square',
    'stairs': 'stair',
    'starbucks': 'starbuck',
    'stars': 'star',
    'statues': 'statue',
    'stems': 'stem',
    'steps': 'step',
    'stickers': 'sticker',
    'sticks': 'stick',
    'stirrups': 'stirrup',
    'stones': 'stone',
    'stools': 'stool',
    'straps': 'strap',
    'strawberries': 'strawberry',
    'streets': 'street',
    'strings': 'string',
    'stripes': 'stripe',
    'students': 'student',
    'suitcases': 'suitcase',
    'suits': 'suit',
    'sunglasses': 'sunglass',
    'surfboards': 'surfboard',
    'surfers': 'surfer',
    'swans': 'swan',
    'tables': 'table',
    'tails': 'tail',
    'tattoos': 'tattoo',
    'teeth': 'tooth',
    'tents': 'tent',
    'ties': 'tie',
    'tiles': 'tile',
    'tires': 'tire',
    'toilets': 'toilet',
    'tomatoes': 'tomato',
    'tongs': 'tong',
    'toothbrushes': 'toothbrush',
    'toothpicks': 'toothpick',
    'toppings': 'topping',
    'tourists': 'tourist',
    'towels': 'towel',
    'toys': 'toy',
    'tracks': 'track',
    'trains': 'train',
    'travelers': 'traveler',
    'trees': 'tree',
    'triangles': 'triangle',
    'tricks': 'trick',
    'trucks': 'truck',
    'trunks': 'trunk',
    'tulips': 'tulip',
    'tusks': 'tusk',
    'twigs': 'twig',
    'umbrellas': 'umbrella',
    'uniforms': 'uniform',
    'urinals': 'urinal',
    'utensils': 'utensil',
    'vans': 'van',
    'vases': 'vase',
    'vegetables': 'vegetable',
    'veggies': 'veggy',
    'vehicles': 'vehicle',
    'vines': 'vine',
    'walls': 'wall',
    'waves': 'wafe',
    'weeds': 'weed',
    'wetsuits': 'wetsuit',
    'wheels': 'wheel',
    'whiskers': 'whisker',
    'windows': 'window',
    'wings': 'wing',
    'wipers': 'wiper',
    'wires': 'wire',
    'women': 'woman',
    'woods': 'wood',
    'words': 'word',
    'wrinkles': 'wrinkle',
    'wristbands': 'wristband',
    'zebras': 'zebra'}

    replaced_prediction = [replace_dict.get(word, word) for word in prediction_text.split()]
    replaced_prediction = ' '.join(replaced_prediction)

    return replaced_prediction