"""
This script prepares data in the format for the testing
algorithms to run
The script is expanded to the resource attribute
"""

from __future__ import division
import copy
import itertools
import pdb
import re
from pathlib import Path
from typing import Dict

import numpy as np
import pm4py
import pandas as pd
from keras_nlp.src.layers import SinePositionEncoding
from src.commons import shared_variables as shared
from src.commons.log_utils import LogData

def prepare_encoded_data(log_data: LogData, resource: bool):
    """
    Get all possible symbols for activities and resources and annotate them with integers.
    """
    act_name_key = log_data.act_name_key
    act_chars = log_data.log[act_name_key].unique().tolist()
    act_chars.sort()
    target_act_chars = copy.copy(act_chars)
    target_act_chars.append('!')
    #target_act_chars.sort()

    act_to_int = dict((c, i+1) for i, c in enumerate(act_chars))
    target_act_to_int = dict((c, i+1) for i, c in enumerate(target_act_chars))
    target_int_to_act = dict((i+1, c) for i, c in enumerate(target_act_chars))

    if resource:
        res_name_key = log_data.res_name_key
        res_chars = list(log_data.log[res_name_key].unique())
        res_chars.sort()
        target_res_chars = copy.copy(res_chars)
        target_res_chars.append('!')
        #target_res_chars.sort()
        res_to_int = dict((c, i+1) for i, c in enumerate(res_chars))
        target_res_to_int = dict((c, i+1) for i, c in enumerate(target_res_chars))
        target_int_to_res = dict((i+1, c) for i, c in enumerate(target_res_chars))
    else:
        res_to_int = None
        target_res_to_int = None
        target_int_to_res = None
    return act_chars,res_chars, act_to_int, target_act_to_int, target_int_to_act, res_to_int, target_res_to_int, target_int_to_res


def select_petrinet_compliant_traces(log_data: LogData,  method_fitness: str, traces: pd.DataFrame, path_to_pn_model_file: Path):
    """
    Select traces compliant to a Petri Net at least in a certain percentage specified as compliance_th
    """

    compliant_trace_ids = []
    if (method_fitness == "fitness_alignments") or  (method_fitness == "conformance_diagnostics_alignments_prefix"):
        method_fitness = "conformance_diagnostics_alignments"
    elif method_fitness == "fitness_token_based_replay":
        method_fitness = "conformance_diagnostics_token_based_replay"

    for trace_id, fitness in get_pn_fitness(path_to_pn_model_file, method_fitness, traces, log_data).items():
        if fitness >= log_data.compliance_th:
            compliant_trace_ids.append(trace_id)

    compliant_traces = traces[traces[log_data.case_name_key].isin(compliant_trace_ids)]
    return compliant_traces


def get_pn_fitness(bk_file: Path, method_fitness: str, log: pd.DataFrame, log_data: LogData) -> Dict[str, float]:
    # Decode traces for feeding them to the Petri net
    dec_log = log.replace(to_replace={
        log_data.act_name_key: log_data.act_enc_mapping,
    })

    dec_log[log_data.timestamp_key] = pd.to_datetime(log_data.log[log_data.timestamp_key], unit='s') 

    if 'bpmn' in str(bk_file):
        bpmn = pm4py.read_bpmn(str(bk_file))
        net, initial_marking, final_marking = pm4py.convert_to_petri_net(bpmn)
    else:
        net, initial_marking, final_marking = pm4py.read_pnml(str(bk_file))

    if method_fitness == "conformance_diagnostics_alignments":
        alignments = pm4py.conformance_diagnostics_alignments(dec_log,  net, initial_marking, final_marking,
                                                            activity_key=log_data.act_name_key,
                                                            case_id_key=log_data.case_name_key,
                                                            timestamp_key= log_data.timestamp_key)
        trace_fitnesses = [a['fitness'] for a in alignments]
        
    elif method_fitness == "fitness_alignments":
        alignments = pm4py.fitness_alignments(dec_log,  net, initial_marking, final_marking,
                                                          activity_key=log_data.act_name_key,
                                                          case_id_key=log_data.case_name_key,
                                                          timestamp_key= log_data.timestamp_key)
                                                          
        trace_fitnesses = [alignments['log_fitness']]
    
    elif method_fitness == "conformance_diagnostics_token_based_replay":
        alignments = pm4py.conformance_diagnostics_token_based_replay(dec_log,  net, initial_marking, final_marking,
                                                          activity_key=log_data.act_name_key,
                                                          case_id_key=log_data.case_name_key,
                                                          timestamp_key= log_data.timestamp_key)
        trace_fitnesses = [a['trace_fitness'] for a in alignments]
        
    elif method_fitness == "fitness_token_based_replay":
        alignments = pm4py.fitness_token_based_replay(dec_log,  net, initial_marking, final_marking,
                                                          activity_key=log_data.act_name_key,
                                                          case_id_key=log_data.case_name_key,
                                                          timestamp_key= log_data.timestamp_key)
        trace_fitnesses = [alignments['log_fitness']]
        
        
    trace_ids = list(log[log_data.case_name_key].unique())
    trace_ids_with_fitness = dict(zip(trace_ids, trace_fitnesses))
    return trace_ids_with_fitness


# === Helper functions ===


def encode(crop_trace: pd.DataFrame, log_data: LogData,  maxlen: int, char_indices: Dict[str, int],
                      char_indices_group: Dict[str, int], resource: bool) -> np.ndarray:
    """
    Onehot encoding of an ongoing trace (control-flow + resource)
    """
    chars = list(char_indices.keys())
    if resource:
        sentence = ''.join(crop_trace[log_data.act_name_key].tolist())
        sentence_group = ''.join(crop_trace[log_data.res_name_key].tolist())
        chars_group = list(char_indices_group.keys())
        if shared.use_One_hot_encoding:
            num_features = len(chars) + len(chars_group) + 1
            x = np.zeros((1, maxlen, num_features), dtype=np.float32)
            leftpad = maxlen - len(sentence)
            for t, char in enumerate(sentence):
                for c in chars:
                    if c == char:
                        x[0, t + leftpad, char_indices[c] - 1] = 1
                if t < len(sentence_group):
                    if sentence_group[t] in chars_group:
                        x[0, t + leftpad, len(char_indices) + char_indices_group[sentence_group[t]] - 1] = 1
                x[0, t + leftpad, len(chars) + len(chars_group)] = t + 1
        else:
            if shared.combined_Act_res:
                result_list = [x + y for x, y in itertools.product(chars, chars_group)]
                target_to_int = dict((c, i + 1) for i, c in enumerate(result_list))
                num_features = maxlen
                x = np.zeros((1, num_features), dtype=np.float32)
                for t, char in enumerate(sentence):
                    x[0, t] = target_to_int[char + sentence_group[t]]
            else:
                num_features = maxlen * 2
                counter_act = 0
                counter_res = 1
                x = np.zeros((1, num_features), dtype=np.float32)
                for t, char in enumerate(sentence):
                    x[0, counter_act] = char_indices[char]
                    if t < len(sentence_group):
                        x[0, counter_res] = char_indices_group[sentence_group[t]]
                    counter_act += 2
                    counter_res += 2
    else:
        sentence = ''.join(crop_trace[log_data.act_name_key].tolist())
        if shared.use_One_hot_encoding:
            num_features = len(chars) + 1
            x = np.zeros((1, maxlen, num_features), dtype=np.float32)
            leftpad = maxlen - len(sentence)
            for t, char in enumerate(sentence):
                for c in chars:
                    if c == char:
                        x[0, t + leftpad, char_indices[c]] = 1
                x[0, t + leftpad, len(chars)] = t + 1
        else:
            num_features = maxlen
            x = np.zeros((1, num_features), dtype=np.float32)
            for t, char in enumerate(sentence):
                x[0, t] = char_indices[char]
    return x

def repetitions(seq: str):
    r = re.compile(r"(.+?)\1+")
    for match in r.finditer(seq):
        yield match.group(1), len(match.group(0)) / len(match.group(1)) #, indices


def reduce_loop_probability(prefix_seq):
    tmp = dict()
    list_of_rep = list(repetitions(prefix_seq))
    if list_of_rep:
        str_rep = list_of_rep[-1][0]
        if prefix_seq.endswith(str_rep):
            return np.math.exp(list_of_rep[-1][-1]), list_of_rep[-1][0][0]
        else:
            return 1, list_of_rep[-1][0][0]
    return 1, " "


def apply_reduction_probability(act_seq, res_seq, pred_act, pred_res, target_act_to_ind, target_res_to_ind, resource=False):
    stop_symbol_probability_amplifier_current, start_of_the_cycle_symbol=reduce_loop_probability(act_seq)
    if start_of_the_cycle_symbol in target_act_to_ind:
        place_of_starting_symbol = target_act_to_ind[start_of_the_cycle_symbol]
        pred_act[place_of_starting_symbol-1] = pred_act[place_of_starting_symbol-1] / stop_symbol_probability_amplifier_current
    stop_symbol_probability_amplifier_current_res, start_of_the_cycle_symbol_res = reduce_loop_probability(res_seq)
    if resource:
        if start_of_the_cycle_symbol_res in target_res_to_ind:
            place_of_starting_symbol_res = target_res_to_ind[start_of_the_cycle_symbol_res]
            pred_res[place_of_starting_symbol_res-1] = pred_res[
                                                     place_of_starting_symbol_res-1] / stop_symbol_probability_amplifier_current_res
    return pred_act, pred_res

def get_beam_size(self, NodePrediction, current_prediction_premis, prefix_trace, prefix_trace_df,
                  prediction, res_prediction, y_char, fitness, act_ground_truth_org,
                  char_indices, target_ind_to_act, target_act_to_ind, target_ind_to_res, target_res_to_ind, step, 
                  log_data, resource, beam_size):
    record = []
    act_prefix = prefix_trace.cropped_line
    res_prefix = prefix_trace.cropped_line_group if resource else None
    print(f'Beam size: {beam_size}, act_prefix: {act_prefix}',f'res_prefix: {res_prefix}' if resource else '')
    if shared.useProb_reduction:
        prediction, res_prediction =  apply_reduction_probability(act_prefix, res_prefix, prediction, res_prediction, target_act_to_ind, target_res_to_ind,
                               resource)
    if resource:
        # create probability matrix
        prob_matrix = np.log(prediction) + np.log(res_prediction[:, np.newaxis])
        sorted_prob_matrix = np.argsort(prob_matrix, axis=None)[::-1]

    for j in range(beam_size):
        prefix_trace = prefix_trace.cropped_trace if isinstance(prefix_trace, NodePrediction) else prefix_trace
        print(f'Iteration: {j}')
        if resource:
            res_pred_idx, act_pred_idx = np.unravel_index(sorted_prob_matrix[j], prob_matrix.shape)
            temp_prediction = target_ind_to_act[act_pred_idx + 1]
            temp_res_prediction = target_ind_to_res[res_pred_idx + 1]
            probability_this = prob_matrix[res_pred_idx][act_pred_idx]
        else:
            pred_idx = np.argsort(prediction)[len(prediction) - j - 1]
            temp_prediction = target_ind_to_act[pred_idx + 1]
            temp_res_prediction = None
            probability_this = np.log(np.sort(prediction)[len(prediction) - 1 - j])
        predicted_row = prefix_trace_df.tail(1).copy()
        predicted_row.loc[:, log_data.act_name_key] = temp_prediction
        predicted_row.loc[:, log_data.res_name_key] = temp_res_prediction
        temp_cropped_trace_next = pd.concat([prefix_trace, predicted_row], axis=0)
        probability_of = current_prediction_premis.probability_of + probability_this
        print(f'Temp prediction: {temp_prediction}, Temp res prediction: {temp_res_prediction}, Probability:{probability_of}')
        temp = NodePrediction(temp_cropped_trace_next,probability_of)
        self.put(temp)
        trace_org = [log_data.act_enc_mapping[i] if i != "!" else "" for i in temp_cropped_trace_next[log_data.act_name_key].tolist()]
        if len(fitness) > 0:
            fitness_sorted = np.array(fitness)[np.argsort(prediction)]
            fitness_this = fitness_sorted[len(fitness_sorted) - 1 - j]
            y_char_sorted = np.array(y_char)[np.argsort(prediction)]
            y_char_this = y_char_sorted[len(y_char_sorted) - 1 - j]

            record.append(str(
                "trace_org = " + '>>'.join(trace_org) +
                "// previous = " + str(round(current_prediction_premis.probability_of, 3)) +
                "// current = " + str(round(current_prediction_premis.probability_of + np.log(probability_this), 3)) +
                "// rnn = " + str(round(y_char_this, 3)) +
                "// fitness = " + str(round(fitness_this, 3))) +
                          "&"
                          )
    return self, record
