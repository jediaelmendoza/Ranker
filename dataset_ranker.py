import pandas as pd
import ranker as ranker
import os

def ranking(args):
    i = 0
    while i < 25:
        i = i + 1
        dataset_path = args.output_path / f"dataset{i}.csv"
        if not dataset_path.exists():
            print(f"No dataset{i}.csv found in the output directory.")
            continue
        print(f"##### Ranking dataset{i}.csv #####")
        dataset_df = pd.read_csv(dataset_path)
        ranked_df = None
        ranking_method = ""
        ranking_model = ""
        if args.ranker == 0:
            ranked_df = dataset_df
            ranking_method = "None"
        elif args.ranker == 1:
            if args.model == 1:
                ranked_df = ranker.train_test_pointwise(dataset_df,1)
                ranking_method= "PointwiseDecisionTree"
            elif args.model == 2:
                ranked_df = ranker.train_test_pointwise(dataset_df,2)
                ranking_method= "PointwiseNeuralNet"
        elif args.ranker == 2:
            if args.model == 1:
                ranked_df = ranker.train_test_pairwise(dataset_df,1)
                ranking_method= "PairwiseDecisionTree"
            elif args.model == 2:
                ranked_df = ranker.train_test_pairwise(dataset_df,2)
                ranking_method= "PairwiseNeuralNet"
        else:
            print(f"Invalid ranking method.")
            break
        results_path = args.output_path / ranking_method
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        new_dataset = results_path / f"dataset{i}.csv"
        ranked_df.to_csv(new_dataset, index=False)
    ranker.compute_apfdc_loop(results_path)
    result = "results"
    apfdc_path = results_path / result
    ranker.collect_apfdc_loop(apfdc_path)

def compute_apfdc(args):
    i = 0
    while i < 25:
        i = i + 1
        dataset_path = args.output_path / f"dataset{i}.csv"
        if not dataset_path.exists():
            print(f"No dataset{i}.csv found in the output directory.")
            continue
        print(f"##### Evaluating dataset{i}.csv #####")
        dataset_df = pd.read_csv(dataset_path)
        apfdc = ranker.compute_apfdc(dataset_df)
        results = 'results'
        results_path = args.output_path / results
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        new_dataset = results_path / f"result{i}.csv"
        apfdc.to_csv(new_dataset, index=False)

def collect_apfdc(args):
    i = 0
    apfdcs = pd.DataFrame(columns=['Dataset','Mean_APFDc'])
    while i < 25:
        i = i + 1
        dataset_path = args.output_path / f"result{i}.csv"
        if not dataset_path.exists():
            print(f"No result{i}.csv found in the output directory.")
            continue
        dataset_df = pd.read_csv(dataset_path)
        mean = ranker.mean_apfdc(dataset_df)
        apfdcs = apfdcs.append({'Dataset': f"result{i}", 'Mean_APFDc': mean}, ignore_index=True)
    results_path = args.output_path
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    new_dataset = results_path / f"collect_apfdc.csv"
    apfdcs.to_csv(new_dataset, index=False)