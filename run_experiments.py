import argparse
import tensorflow as tf
import statistics as stat

from src.commons import log_utils, shared_variables as shared
from src.evaluation import evaluation
from src.training import train_model


class ExperimentRunner:
    def __init__(self, train, evaluate, use_variant_split):
        self._train = train
        self._evaluate = evaluate
        self._use_variant_split = use_variant_split

        print('Perform training:', self._train)
        print('Perform evaluation:', self._evaluate)
        print('Use variant-based split:', self._use_variant_split)

    def run_experiments(self, log_list, alg, method_fitness, weight, resource, timestamp, outcome, model_folder):
        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4,
                                          allow_soft_placement=True)
        session = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_epsilon(session)

        for log_name in log_list:
            log_path = shared.log_folder / log_name
            self._run_single_experiment(log_path, alg, method_fitness, weight, resource, timestamp, outcome,model_folder)

    def _run_single_experiment(self, log_path, alg, method_fitness, weight, resource, timestamp, outcome,model_folder):
        log_data = log_utils.LogData(log_path,self._use_variant_split)
        log_data.encode_log(resource, timestamp, outcome)
        
        trace_sizes = list(log_data.log.value_counts(subset=[log_data.case_name_key], sort=False))

        print('Log name:', log_data.log_name.value + log_data.log_ext.value)
        print('Log size:', len(trace_sizes))
        print('Trace size avg.:', stat.mean(trace_sizes))
        print('Trace size stddev.:', stat.stdev(trace_sizes))
        print('Trace size min.:', min(trace_sizes))
        print('Trace size max.:', max(trace_sizes))
        print(f'Evaluation prefix range: [{log_data.evaluation_prefix_start}, {log_data.evaluation_prefix_end}]')
        log_data.maxlen = max(trace_sizes)
        if self._train:
            train_model.train(log_data, model_folder, resource, outcome) #keras_trans or LSTM
        
        if self._evaluate and shared.usePosEncoding:
            evaluation.evaluate_all(log_data, model_folder+"_PosEncoding", alg, method_fitness, weight, resource, timestamp, outcome) #keras_trans or LSTM
        else:
            evaluation.evaluate_all(log_data, model_folder, alg, method_fitness, weight, resource, timestamp,
                                    outcome)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default=None, help='input log')
    #parser.add_argument('--use_Pos_enc', default=False, action='store_true', help='use positional encoding')
    parser.add_argument('--algo', default="beamsearch", help='use baseline or beamsearch', type=str)
    parser.add_argument('--model',default="keras_trans", help='use LSTM or Keras_trans', type=str)
    parser.add_argument('--use_Prob_reduction', default=False, action='store_true',help='use probability reduction')
    parser.add_argument('--use_variant_split', default=False, action='store_true', help='Use variant-based split for train/test')
    # encoding configuration
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--one_hot_encoding', default=False, action='store_true', help='use one-hot encoding')
    group.add_argument('--combine_Act_res', default=False, action='store_true', help='encode all activities resources combination')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', default=True, action='store_true', help='train without evaluating')
    group.add_argument('--evaluate', default=False, action='store_true', help='evaluate without training')
    group.add_argument('--full_run', default=False, action='store_true', help='train and evaluate model')
    group.add_argument('--weight', default=0, action='store_true', help='train and evaluate model')



    args = parser.parse_args()

    logs = [args.log.strip()] if args.log else shared.log_list
    w = [float(args.weight)]
    #shared.usePosEncoding=args.use_Pos_enc
    shared.use_One_hot_encoding=args.one_hot_encoding
    shared.combined_Act_res = args.combine_Act_res
    shared.useProb_reduction = args.use_Prob_reduction

    if args.full_run:
        args.train = True
        args.evaluate = True
    elif args.evaluate:
        args.train = False
        args.evaluate = True
    elif args.train:
        args.train = True
        args.evaluate = False        
    

    ExperimentRunner(train=args.train,
                     evaluate=args.evaluate,
                     use_variant_split=args.use_variant_split) \
        .run_experiments(log_list=logs,
                         alg = args.algo,
                         method_fitness = None,  
                         weight = w,
                         resource = True,
                         timestamp = False,
                         outcome = False,
                         model_folder= args.model)
