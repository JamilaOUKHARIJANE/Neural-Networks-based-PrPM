
from __future__ import division
import csv
from pathlib import Path
from queue import PriorityQueue
import distance
import numpy as np
import pandas as pd
import keras
from jellyfish import damerau_levenshtein_distance

from src.commons import shared_variables as shared
from src.commons.log_utils import LogData
from src.evaluation.prepare_data import get_beam_size, encode, get_pn_fitness, compliance_checking
from src.training.train_common import CustomTransformer
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
from tqdm import tqdm


def run_experiments(log_data: LogData, compliant_traces: pd.DataFrame, maxlen, predict_size, char_indices,
                    target_char_indices, target_indices_char, char_indices_group, target_char_indices_group,
                    target_indices_char_group, model_file: Path, output_file: Path, bk_file: Path,
                    method_fitness: str, resource: bool, outcome: bool, weight: list, bk_model):

    # Load model, set this to the model generated by train.py
    model = keras.models.load_model(model_file, custom_objects={'CustomTransformer': CustomTransformer})
    class NodePrediction:
        def __init__(self, crop_trace: pd.DataFrame, probability_of=0):
            self.cropped_trace = crop_trace
            self.cropped_line = ''.join(crop_trace[log_data.act_name_key].tolist())
            if resource:
                self.cropped_line_group = ''.join(crop_trace[log_data.res_name_key].tolist())
            if (not resource and self.cropped_line[-1] != '!') or (resource and self.cropped_line_group [-1] != '!' and self.cropped_line[-1] != '!') :
                self.model_input = encode(crop_trace, log_data, maxlen, char_indices, char_indices_group, resource)
            self.probability_of = probability_of

        def __str__(self):
            return f"Prefix: {self.cropped_line}, prob. {self.probability_of}"

        def __lt__(self, other):
            return -self.probability_of < -other.probability_of
        def get_cropped_trace(self):
            return self.cropped_trace

    class CacheFitness:
        def __init__(self):
            self.trace = {}
            
        def add(self, crop_trace:str, fitness: float):
            self.trace[crop_trace] = fitness
            
        def get(self, crop_trace:str):
            if crop_trace not in self.trace.keys():
                return None
            else:
                return self.trace[crop_trace]

    class CacheTrace:
        def __init__(self):
            self.trace = {}
            
        def add(self, crop_trace:str, output: list):
            self.trace[crop_trace] = output
            
        def get(self, crop_trace:str):
            if crop_trace not in self.trace.keys():
                return None
            else:
                return self.trace[crop_trace]

    def ensure_node_prediction(prefix, log_data, maxlen, char_indices, char_indices_group, resource):
        if isinstance(prefix, NodePrediction):
            return prefix
        else:
            if isinstance(prefix, dict):
                crop_trace = prefix.get('crop_trace', pd.DataFrame())
                probability_of = prefix.get('probability_of', 0)
                return NodePrediction(crop_trace, probability_of)
            elif isinstance(prefix, str):
                crop_trace = pd.DataFrame({log_data.act_name_key: list(prefix)})
                if resource:
                    crop_trace[log_data.res_name_key] = [''] * len(crop_trace)  # Add empty 'Resource' column
                return NodePrediction(crop_trace, 0)
            elif isinstance(prefix, pd.DataFrame):
                return NodePrediction(prefix, 0)
            else:
                raise ValueError(f"Cannot convert {type(prefix)} to NodePrediction")
            
    def apply_trace(trace, prefix_size, log_data, predict_size, bk_file, target_indices_char, target_char_indices,
                    target_indices_char_group, target_char_indices_group, method_fitness, resource, outcome, weight, bk_model):

        if len(trace) > prefix_size:
            
            trace_name = trace[log_data.case_name_key].iloc[0]
            trace_prefix = trace.head(prefix_size)

            # Concatenate activities and resources in the trace prefix
            trace_prefix_act = ''.join(trace_prefix[log_data.act_name_key].tolist())
            trace_prefix_res = ''.join(trace_prefix[log_data.res_name_key].tolist()) if resource else None

            act_prefix = ''.join(trace_prefix[log_data.act_name_key].tolist()) + "_" + str(weight)
            res_prefix = ''.join(trace_prefix[log_data.res_name_key].tolist()) if resource else None
            check_prefix = cache_trace.get(act_prefix+""+res_prefix) if resource else cache_trace.get(act_prefix)
            if check_prefix == None:

                trace_ground_truth = trace.tail(trace.shape[0] - prefix_size)
                act_ground_truth = ''.join(trace_ground_truth[log_data.act_name_key].tolist())
                act_ground_truth_org = [log_data.act_enc_mapping[i] if i != "!" else "" for i in act_ground_truth]

                if resource:
                    res_ground_truth = ''.join(trace_ground_truth[log_data.res_name_key].tolist())
                if outcome:
                    outcome_ground_truth = trace[log_data.label_name_key].iloc[0]

                # Initialize queue for beam search, put root of the tree inside
                visited_nodes: PriorityQueue[NodePrediction] = PriorityQueue()
                visited_nodes.put(NodePrediction(trace_prefix))
                frontier_nodes: PriorityQueue[NodePrediction] = PriorityQueue()

                child_node = None
                record_update = []
                is_violated = False
                for i in range(predict_size - prefix_size): #copia al posto di ground truth
                    if visited_nodes.empty():
                            break
                    violated_nodes = {}
                    for k in range(min(shared.beam_size, len(visited_nodes.queue))):
                        child_node = visited_nodes.get()
                        temp_cropped_trace = child_node.cropped_trace
                        if child_node.cropped_line[-1] == "!" or (resource and child_node.cropped_line_group[-1] == "!"):
                            if shared.BK_end:
                                prefix_trace = ensure_node_prediction(temp_cropped_trace, log_data, maxlen,
                                                                  char_indices, char_indices_group,
                                                                  resource)
                                prefix_trace = prefix_trace.cropped_trace if isinstance(prefix_trace,
                                                                                    NodePrediction) else prefix_trace
                                prefix_trace = prefix_trace[:-1]
                                if resource:
                                    BK_res = compliance_checking(log_data, child_node.cropped_line[-1],
                                                         child_node.cropped_line_group[-1], bk_model, prefix_trace,resource)
                                else:
                                    BK_res = compliance_checking(log_data, child_node.cropped_line[-1],
                                                         None,bk_model, prefix_trace,resource)
                            if k == 0:
                                if shared.BK_end and BK_res == np.NINF:  # violated: continue the search
                                    violated_nodes[k] = child_node
                                    continue
                                else: # satisfied or not using BK
                                    visited_nodes = PriorityQueue()
                                    break
                            else:
                                if shared.BK_end:
                                    if BK_res == np.NINF:  # violated: continue the search
                                        if all(violated_nodes.get(i) for i in range(k)) and k == min(shared.beam_size, len(visited_nodes.queue)) -1:
                                            visited_nodes = PriorityQueue()
                                            child_node = violated_nodes.get(0)
                                            is_violated = True
                                            break
                                        else:
                                            violated_nodes[k] = child_node
                                            continue
                                    else: # satisfied
                                        if all(violated_nodes.get(i) for i in range(k)):
                                            visited_nodes = PriorityQueue()
                                            break
                                        else:
                                            continue
                                else:
                                    continue
                        enc = child_node.model_input
                        if shared.use_modulator:
                            y = model.predict([enc["x_act"], enc["x_group"]], verbose=0)
                        else:
                            y = model.predict(enc, verbose=0) # make predictions
                        
                        if  not resource and not outcome:
                            y_char = y[0]
                            y_group = None
                            y_o = None
                        elif not resource and outcome:
                            y_char = y[0][0]
                            y_group = None
                            y_o = y[1][0][0]
                        elif resource and not outcome:
                            y_char = y[0][0]
                            y_group = y[1][0]
                            y_o = None
                        elif resource and outcome:
                            y_char = y[0][0]
                            y_group = y[1][0]
                            y_o = y[2][0][0]
                        
                        # combine with fitness
                        if method_fitness is None or weight == 0:
                            fitness = []
                            fitness_temp = []
                            score = y_char
                            
                        else:
                            fitness = [] 
                            fitness_temp = []
                            
                            for f in range(1,len(target_indices_char)+1):
                                
                                if f <= len(target_indices_char):
                                    temp_prediction = target_indices_char[f]               
                                else:
                                    temp_prediction = log_data.new_chars[f-len(target_indices_char)]
                                    
                                predicted_row = temp_cropped_trace.tail(1).copy()
                                predicted_row.loc[:, log_data.act_name_key] = temp_prediction
                                temp_cropped_trace_next= pd.concat([temp_cropped_trace, predicted_row])                        
                                temp_cropped_line_next = ''.join(temp_cropped_trace_next[log_data.act_name_key].tolist()) 
                                                                
                                check_cache = cache_fitness.get(temp_cropped_line_next )
                                if check_cache == None:
                                    fitness_current = get_pn_fitness(bk_file, method_fitness, temp_cropped_trace_next, log_data)[trace_name]
                                    cache_fitness.add(temp_cropped_line_next, fitness_current)
                                else:
                                    fitness_current = check_cache
                                    
                                fitness = fitness +  [np.exp(fitness_current) ]
                                fitness_temp = fitness_temp +  [fitness_current]
                                                            
                            if sum(fitness) > 0:
                                fitness = [f/sum(fitness) for f in fitness] 
                            else:
                                fitness = np.repeat(1/len(fitness),len(fitness)).tolist()
                                fitness_temp = np.repeat(1/len(fitness_temp),len(fitness_temp)).tolist()
                                
                            '''if len(log_data.new_chars) > 0:
                                y_char = y_char + min(y_char)*len(log_data.new_chars)'''
                            score = [pow(a,1-weight)*pow(b,weight) for a,b in zip(y_char, fitness)]
                        temp_cropped_trace = ensure_node_prediction(temp_cropped_trace, log_data, maxlen, char_indices, char_indices_group, resource)
                        # put top 3 based on score
                        frontier_nodes, record = get_beam_size(frontier_nodes, NodePrediction, child_node, bk_model,weight, temp_cropped_trace,
                                                               score,y_group, y_char, fitness_temp,
                                                               target_indices_char, target_char_indices, target_indices_char_group,
                                                               target_char_indices_group,log_data, resource, beam_size = shared.beam_size)
                        record_update = record_update + record
                        
                    visited_nodes = frontier_nodes
                    frontier_nodes = PriorityQueue()
                predicted = child_node.cropped_line[prefix_size:-1]
                if resource:
                    predicted_group = child_node.cropped_line_group[prefix_size:-1]
                
                if outcome:
                    predicted_outcome = '1' if y_o >= 0.5 else '0'

                output = []
                if len(act_ground_truth) > 0:
                    output.append(trace_name)
                    output.append(prefix_size)
                    output.append(trace_prefix_act)
                    output.append(act_ground_truth)
                    output.append(predicted)
                    dls = 1 - \
                        (damerau_levenshtein_distance(predicted, act_ground_truth) / max(len(predicted), len(act_ground_truth)))
                    if dls < 0:
                        dls = 0
                    output.append(dls)
                    output.append(1 - distance.jaccard(predicted, act_ground_truth))
                    
                    if resource:
                        output.append(trace_prefix_res)
                        output.append(res_ground_truth)
                        output.append(predicted_group)
                        dls_res = 1 - \
                            (damerau_levenshtein_distance(predicted_group, res_ground_truth)
                                / max(len(predicted_group), len(res_ground_truth)))
                        if dls_res < 0:
                            dls_res = 0
                        output.append(dls_res)
                        output.append(1 - distance.jaccard(predicted_group, res_ground_truth))
                        # Combine activity and resource strings for combined evaluation
                        combined_ground_truth = ''.join([a + r for a, r in zip(act_ground_truth, res_ground_truth)])
                        combined_predicted = ''.join([a + r for a, r in zip(predicted, predicted_group)])

                        dls_combined = 1 - (
                                    damerau_levenshtein_distance(combined_predicted, combined_ground_truth) / max(
                                len(combined_predicted), len(combined_ground_truth)))
                        if dls_combined < 0:
                            dls_combined = 0
                        output.append(dls_combined)
                    
                    if outcome:
                        output.append(outcome_ground_truth)
                        output.append(predicted_outcome)
                        output.append('1' if outcome_ground_truth == predicted_outcome else '0')
                    output.append(weight)
                    output.append('Violated' if is_violated else 'Satisfied')
                    cache_trace.add(act_prefix+""+res_prefix, output) if resource else cache_trace.add(act_prefix, output)
            else:
                print('check_prefix:', check_prefix)
                trace_ground_truth = trace.tail(trace.shape[0] - prefix_size)
                act_ground_truth = ''.join(trace_ground_truth[log_data.act_name_key].tolist())
                output = []
                
                output.append(trace_name)
                output.append(prefix_size)
                output.append(check_prefix[2])
                output.append(act_ground_truth)
                predicted = check_prefix[4]
                output.append(predicted)
                dls = 1 - \
                    (damerau_levenshtein_distance(predicted, act_ground_truth) / max(len(predicted), len(act_ground_truth)))
                if dls < 0:
                    dls = 0
                output.append(dls)
                output.append(1 - distance.jaccard(predicted, act_ground_truth))
                if resource:
                    trace_prefix_res = ''.join(trace_prefix[log_data.res_name_key].tolist())
                    res_ground_truth = ''.join(trace_ground_truth[log_data.res_name_key].tolist())
                    predicted_group = check_prefix[9]
                    output.append(trace_prefix_res)
                    output.append(res_ground_truth)
                    output.append(predicted_group)

                    dls_res = 1 - (damerau_levenshtein_distance(predicted_group, res_ground_truth) / max(len(predicted_group), len(res_ground_truth)))
                    dls_res = max(dls_res, 0)  # Ensure non-negative
                    output.append(dls_res)
                    output.append(1 - distance.jaccard(predicted_group, res_ground_truth))
                    # Combine activity and resource strings for combined evaluation
                    combined_ground_truth = ''.join([a + r for a, r in zip(act_ground_truth, res_ground_truth)])
                    combined_predicted = ''.join([a + r for a, r in zip(predicted, predicted_group)])

                    dls_combined = 1 - (
                            damerau_levenshtein_distance(combined_predicted, combined_ground_truth) / max(
                        len(combined_predicted), len(combined_ground_truth)))
                    if dls_combined < 0:
                        dls_combined = 0
                    output.append(dls_combined)
                output.append(check_prefix[-2])  # weight
                output.append(check_prefix[-1]) #compliance satisfied or violated

            if output:
                with open(output_file, 'a', encoding='utf-8', newline='') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    spamwriter.writerow(output)


##############################################################

    with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # Headers for the new file
        if not resource and not outcome:
            spamwriter.writerow(["Case ID", "Prefix length","Trace Prefix Act", "Ground truth", "Predicted",
                 "Damerau-Levenshtein", "Jaccard", "Weight", "Compliance"])
        elif not resource and outcome:
            spamwriter.writerow(["Case ID", "Prefix length", "Trace Prefix Act", "Ground truth",
                                 "Predicted", "Damerau-Levenshtein", "Jaccard", "Ground truth outcome", "Predicted outcome", "Outcome diff.", "Weight", "Compliance"])
        elif resource and not outcome:
            spamwriter.writerow(["Case ID", "Prefix length",
                 "Trace Prefix Act", "Ground truth", "Predicted Acts", "Damerau-Levenshtein Acts", "Jaccard Acts",
                 "Trace Prefix Res", "Ground Truth Resources", "Predicted Resources", "Damerau-Levenshtein Resources",
                 "Jaccard Resources", "Damerau-Levenshtein Combined","Weight", "Compliance"])
        elif resource and outcome:
            spamwriter.writerow(["Case ID", "Prefix length", "Trace Prefix Act", "Ground truth", "Predicted", "Damerau-Levenshtein",
                                 "Jaccard", "Trace Prefix Res", "Ground Truth Group", "Predicted Group", "Damerau-Levenshtein Resource",
                                 "Ground truth outcome", "Predicted outcome", "Outcome diff.", "Weight", "Compliance"])
            
    cache_fitness = CacheFitness()
    cache_trace = CacheTrace()
    for prefix_size in range(log_data.evaluation_prefix_start, log_data.evaluation_prefix_end+1):
        print(prefix_size)
        compliant_traces = compliant_traces.reset_index(drop=True) 
        for w in weight: 
            tqdm.pandas()
            compliant_traces.groupby(log_data.case_name_key).progress_apply(lambda x: apply_trace(x, prefix_size, log_data,
                                                                                    predict_size, bk_file,
                                                                                   target_indices_char, target_char_indices,
                                                                                    target_indices_char_group, target_char_indices_group,
                                                                                    method_fitness, resource, outcome, w, bk_model))
