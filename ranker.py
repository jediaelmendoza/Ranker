import os

import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers import Dense
from optuna.multi_objective import trial
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
import datetime
import keras
import optuna


def create_idealrank(dataset):
    dataset_df = dataset.copy()
    builds = dataset_df['Build'].unique().tolist()
    ranked = pd.DataFrame()
    for build in builds:
        build_df = dataset_df.loc[dataset_df['Build'] == build].copy()
        build_df.sort_values(
            ["Verdict", 'Duration'],
            ascending=[False, True],
            inplace=True,
            ignore_index=True,
        )
        build_df["Rank"] = build_df.index
        ranked = ranked.append(build_df,ignore_index=True)
    #ranked.to_csv('ranked.csv', index=False)
    return ranked

def normalize(dataset):
        dont_normalize = [
            "Build",
            "Test",
            "Verdict",
            "Duration",
            "Rank"
        ]
        feature_dataset = dataset.drop(dont_normalize, axis=1)
        scaler = MinMaxScaler()
        scaler.fit(feature_dataset)
        normalized_dataset = pd.DataFrame(
            scaler.transform(feature_dataset),
            columns=feature_dataset.columns,
        )
        for col in dont_normalize:
            normalized_dataset[col] = dataset[col]
        ranks = normalized_dataset.loc[:,normalized_dataset.columns.isin(['Rank'])]
        scaler.fit(ranks)
        #ranks.to_csv("ranks.csv",index=False)
        normalized_dataset["NormRank"] = scaler.transform(ranks)
        #normalized_dataset.to_csv("norms.csv")
        return normalized_dataset

def create_training_set(dataset):
    df = dataset.copy()
    ranked = normalize(create_idealrank(df))
    #ranked.to_csv("ranked.csv")
    #print(ranked.head())

    x_train_index, x_test_index = next(GroupShuffleSplit(test_size=0.10, random_state=42).split(ranked,groups=ranked['Build']))

    #Training set
    train_set = ranked.iloc[x_train_index]
    x_train = train_set.loc[:, ~train_set.columns.isin(['Rank','NormRank','Verdict','Duration','Build','Test'])]
    y_train = train_set.loc[:, train_set.columns.isin(['NormRank'])]

    #Test set
    test_set = ranked.iloc[x_test_index]
    x_test = test_set.loc[:, ~test_set.columns.isin(['Rank','NormRank','Verdict','Duration'])]
    y_test = test_set.loc[:, test_set.columns.isin(['NormRank'])]

    comparison = test_set.loc[:, test_set.columns.isin(['Build','Test','Rank','NormRank','Verdict','Duration'])]
    #x_train.to_csv("x_train.csv",index=False)
    #y_train.to_csv("y_train.csv", index=False)
    #x_test.to_csv("x_test.csv",index=False)
    return x_train, y_train, x_test, y_test, comparison

def create_pairs(dataset):
    df = dataset.copy()
    df['Verdict'] = np.where((df['Verdict'] == 2) | (df['Verdict'] == 1), 1, 0)

    #delete builds with only one test case
    singleTests = df['Build'].value_counts()
    singleBuild = singleTests[singleTests == 1].index
    #print(singleBuild)
    df.drop(df[df['Build'].isin(singleBuild)].index, inplace=True)

    builds = df['Build'].unique().tolist()
    #print(len(builds))
    paired = pd.DataFrame()
    for build in builds:
        build_df = df.loc[df['Build'] == build].copy()
        tests = build_df['Test'].unique().tolist()
        for test in tests:
            current_test = pd.DataFrame()
            other_tests = pd.DataFrame()
            current_test = pd.concat([build_df.loc[build_df['Test'] == test]] * (len(tests)-1), ignore_index=True)
            current_test = current_test.add_suffix('_A')
            other_tests = pd.concat([build_df.loc[build_df['Test'] != test]], ignore_index=True)
            other_tests = other_tests.drop('Build', axis=1)
            other_tests = other_tests.add_suffix('_B')
            current_test = current_test.join(other_tests)
            paired = paired.append(current_test, ignore_index=True)
            #print(current_test.head())
    #paired.to_csv('paired.csv', index=False)
    return paired

def class_pairs(dataset):
    df = dataset.copy()
    df['Class'] = np.where((df['Verdict_A'] > df['Verdict_B']) | ((df['Verdict_A'] == df['Verdict_B']) & (df['Duration_A'] < df['Duration_B'] )),0,1)
    return df

def create_training_set_pair(dataset):
    df = dataset.copy()
    ranked = normalize(create_idealrank(df))
    #ranked.to_csv("ranked.csv")
    #print(ranked.head())

    x_train_index, x_test_index = next(GroupShuffleSplit(test_size=0.10, random_state=42).split(ranked,groups=ranked['Build']))

    #Training set
    train_set = ranked.iloc[x_train_index]
    temp_train_set = train_set.loc[:, ~train_set.columns.isin(['Rank','NormRank'])]
    #temp_train_set = class_pairs(create_pairs(temp_train_set))
    #x_train = temp_train_set.loc[:, ~temp_train_set.columns.isin(['Class','Verdict_A','Verdict_B','Duration_A','Duration_B'])]
    #y_train = temp_train_set.loc[:, temp_train_set.columns.isin(['Class'])]

    #Test set
    test_set = ranked.iloc[x_test_index]
    temp_test_set = test_set.loc[:, ~test_set.columns.isin(['Rank','NormRank'])]
    #temp_test_set = class_pairs(create_pairs(temp_test_set))
    #x_test = temp_test_set.loc[:, ~temp_test_set.columns.isin(['Class','Verdict_A','Verdict_B','Duration_A','Duration_B'])]
    #y_test = temp_test_set.loc[:, temp_test_set.columns.isin(['Class'])]

    to_rank = test_set.loc[:, test_set.columns.isin(['Build','Test','Duration','Verdict','Rank'])]
    #comparison = temp_test_set.loc[:, temp_test_set.columns.isin(['Build_A','Test_A','Duration_A','Verdict_A','Test_B','Duration_B','Verdict_B'])]
    #x_train.to_csv("x_train.csv",index=False)
    #y_train.to_csv("y_train.csv", index=False)
    #x_test.to_csv("x_test.csv",index=False)
    #y_test.to_csv("y_test.csv",index=False)
    return temp_train_set, temp_test_set, to_rank

def rank_pointwise(dataset):
    df = dataset.copy()

    test_builds = df['Build'].unique().tolist()
    final = pd.DataFrame()
    for build in test_builds:
        cur_test = pd.DataFrame(df.loc[df['Build'] == build].copy()).reset_index(drop=True)
        cur_test.sort_values(
            ['Pred'],
            ascending=[True],
            inplace=True,
            ignore_index=True,
        )
        cur_test['PredRank'] = cur_test.index
        final = final.append(cur_test,ignore_index=True)
    return final

def train_test_pointwise(dataset,model):
    df = dataset.copy()
    ranked = create_idealrank(df)


    #DecisionTreeModel
    if model == 1:
        reg_model = DecisionTreeRegressor(min_samples_leaf=12)
        print("DecisionTree")

    #NeuralNetwork
    elif model == 2:
        reg_model = Sequential()
        reg_model.add(Dense(8, input_dim=150, kernel_initializer='normal', activation='relu'))
        reg_model.add(Dense(2536, activation='relu'))
        reg_model.add(Dense(1, activation='linear'))
        reg_model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
        print("Neural Network")

    print("Create training and test set")
    print(str(datetime.datetime.now()))
    #Create training and test set
    x_train, y_train, x_test, y_test, comparison = create_training_set(df)


    #Train
    if model == 1:
        reg_model.fit(x_train, y_train)
    elif model == 2:
        reg_model.fit(x_train, y_train, epochs=100, batch_size=400, verbose=0)

    print("Testing")
    print(str(datetime.datetime.now()))
    #Test
    test_builds = x_test['Build'].unique().tolist()
    result = pd.DataFrame()
    for build in test_builds:
        cur_test = pd.DataFrame(x_test.loc[x_test['Build'] == build].copy()).reset_index(drop=True)
        cur_test = cur_test.drop(columns=['Build','Test'])
        #print(cur_test.head())
        predictions = reg_model.predict(cur_test)
        compare = pd.DataFrame(comparison.loc[comparison['Build'] == build,['Build','Test','Duration','Verdict','Rank','NormRank']].copy())
        #print(compare.head())
        #compare['NormRank'] = y_test
        compare['Pred'] = predictions
        result = result.append(compare,ignore_index=True)
        #print(result.head())
    before = result.copy()
    after = rank_pointwise(before)
    #print(r2_score(after['NormRank'],after['Pred']))
    print("Finished")
    print(str(datetime.datetime.now()))
    return after

def train_test_pairwise(dataset,model):
    df = dataset.copy()


    #DecisionTreeModel
    if model == 1:
        class_model = DecisionTreeClassifier()
        print("DecisionTree")

    #NeuralNetwork
    elif model == 2:
        class_model = Sequential()
        class_model.add(Dense(2531, input_dim=300, kernel_initializer='random_normal', activation='relu'))
        class_model.add(Dense(1, kernel_initializer='random_normal', activation='sigmoid'))
        class_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print("Neural Network")

    print("Create training and test set")
    print(str(datetime.datetime.now()))
    train, test, to_rank = create_training_set_pair(df)


    #Train
    print("Training")
    print(str(datetime.datetime.now()))
    if model == 1:
        train_builds = train['Build'].unique().tolist()
        print("Training sets: " + str(train_builds))
        for build in train_builds:
            #print(build)
            #print(str(datetime.datetime.now()))
            temp_train_build = pd.DataFrame(train.loc[train['Build'] == build].copy()).reset_index(drop=True)
            temp_train_build = class_pairs(create_pairs(temp_train_build))
            x_train = temp_train_build.loc[:, ~temp_train_build.columns.isin(['Class','Verdict_A','Verdict_B','Duration_A','Duration_B','Build_A','Test_A','Test_B'])]
            y_train = temp_train_build.loc[:, temp_train_build.columns.isin(['Class'])]
            class_model.fit(x_train, y_train.values.ravel())
            #x_train.to_csv('xtrain.csv', index=False)
    elif model == 2:
        # Train
        train_builds = train['Build'].unique().tolist()
        print("Training sets: " + str(train_builds))
        print(str(datetime.datetime.now()))
        for build in train_builds:
            #print(build)
            #print(str(datetime.datetime.now()))
            temp_train_build = pd.DataFrame(train.loc[train['Build'] == build].copy()).reset_index(drop=True)
            temp_train_build = class_pairs(create_pairs(temp_train_build))
            x_train = temp_train_build.loc[:, ~temp_train_build.columns.isin(
                ['Class', 'Verdict_A', 'Verdict_B', 'Duration_A', 'Duration_B', 'Build_A', 'Test_A', 'Test_B'])]
            y_train = temp_train_build.loc[:, temp_train_build.columns.isin(['Class'])]
            class_model.fit(x_train, y_train.values.ravel(), batch_size=400, epochs=100, verbose=0)
            # x_train.to_csv('xtrain.csv', index=False)
    #Test
    print("Testing")
    print(str(datetime.datetime.now()))
    test_builds = test['Build'].unique().tolist()
    result = pd.DataFrame()
    print("Testing sets: " + str(test_builds))
    print(str(datetime.datetime.now()))
    for build in test_builds:
        #print(build)
        #print(str(datetime.datetime.now()))
        temp_test_build = pd.DataFrame(test.loc[test['Build'] == build].copy()).reset_index(drop=True)
        temp_test_build = class_pairs(create_pairs(temp_test_build))
        x_test = temp_test_build.loc[:,~temp_test_build.columns.isin(['Class','Verdict_A','Verdict_B','Duration_A','Duration_B','Build_A','Test_A','Test_B'])]
        y_test = temp_test_build.loc[:, temp_test_build.columns.isin(['Class'])]
        comparison = temp_test_build.loc[:, temp_test_build.columns.isin(
            ['Build_A', 'Test_A', 'Duration_A', 'Verdict_A', 'Test_B', 'Duration_B', 'Verdict_B'])]
        #print(cur_test.head())
        if model == 1:
            predictions = class_model.predict(x_test)
        elif model == 2:
            predictions = (class_model.predict(x_test) > 0.5).astype(int)
        compare = pd.DataFrame(comparison.loc[comparison['Build_A'] == build,['Build_A','Test_A','Duration_A','Verdict_A','Test_B','Duration_B','Verdict_B']].copy())
        compare['PredClass'] = predictions
        compare['Class'] = y_test
        result = result.append(compare)
        #x_test.to_csv('xtest.csv', index=False)

    print("Ranking")
    print(str(datetime.datetime.now()))
    ranked = pd.DataFrame()
    for build in test_builds:
        temp_build = result.loc[result['Build_A'] == build]
        temp_rank = to_rank.loc[to_rank['Build'] == build].copy()
        temp_rank['PredRank'] = 0
        for index, row in temp_build.iterrows():
            if row['PredClass'] == 0:
                temp_rank.loc[temp_rank['Test'] == row['Test_B'], 'PredRank'] += 1
        temp_rank.sort_values(
            ['PredRank'],
            ascending=[True],
            inplace=True,
            ignore_index=True,
        )
        ranked = ranked.append(temp_rank,ignore_index=True)
    #print(accuracy_score(result['Class'], result['PredClass']))
    print("Finish")
    print(str(datetime.datetime.now()))
    #result.to_csv("resultsPair.csv",index=False)
    #ranked.to_csv('rankedPair.csv',index=False)
    return ranked

def compute_apfdc(data):
    df = data.copy()
    test_builds = df['Build'].unique().tolist()
    apfdc_df = pd.DataFrame(columns=['Build','APFDc'])
    for build in test_builds:
        temp_build = pd.DataFrame(df.loc[df['Build'] == build].copy().reset_index(drop=True))
        n = len(temp_build)
        m = len(temp_build[temp_build["Verdict"] > 0])
        #print(n)
        #print(m)
        costs = temp_build["Duration"].values.tolist()
        #print(costs)
        failed_costs = 0.0
        for tfi in temp_build[temp_build["Verdict"] > 0].index:
            #print(tfi)
            failed_costs += sum(costs[tfi:]) - (costs[tfi] / 2)
        apfdc = failed_costs / (sum(costs) * m)
        final = float("{:.3f}".format(apfdc))
        apfdc_df = apfdc_df.append({'Build': build, 'APFDc': final}, ignore_index=True)
    #print(apfdc_df.head())
    return apfdc_df

def mean_apfdc(data):
    df = data.copy()
    apfdcs = df['APFDc'].values.tolist()
    mean = sum(apfdcs)/len(apfdcs)
    return mean

def create_model_nn_regression(trial):

    n_layers = trial.suggest_int("n_layers",1,2)
    reg_model = Sequential()
    for i in range(n_layers):
        num_hidden = trial.suggest_int("n_units_l{}".format(i), 4, 3000, log=True)
        if i == 0:
            reg_model.add(Dense(num_hidden, input_dim=150, kernel_initializer='normal', activation='relu'))
        else:
            reg_model.add(
                Dense(
                    num_hidden,
                    activation=trial.suggest_categorical('activation', ['relu', 'linear']),
                )
            )
    reg_model.add(Dense(1, kernel_initializer='normal'))
    reg_model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
    return reg_model
def objective_nn_regression(trial):
    df = pd.read_csv("dataset1.csv")
    x_train, y_train, x_test, y_test, comparison = create_training_set(df)
    reg_model = create_model_nn_regression(trial)

    reg_model.fit(x_train, y_train, batch_size=400, epochs=100)

    test_builds = x_test['Build'].unique().tolist()
    result = pd.DataFrame()
    for build in test_builds:
        cur_test = pd.DataFrame(x_test.loc[x_test['Build'] == build].copy()).reset_index(drop=True)
        cur_test = cur_test.drop(columns=['Build', 'Test'])
        # print(cur_test.head())
        predictions = reg_model.predict(cur_test)
        compare = pd.DataFrame(comparison.loc[comparison['Build'] == build, ['Build', 'Test', 'Duration', 'Verdict', 'Rank',
                                                                             'NormRank']].copy())
        # print(compare.head())
        # compare['NormRank'] = y_test
        compare['Pred'] = predictions
        result = result.append(compare, ignore_index=True)
        # print(result.head())
    before = result.copy()
    after = rank_pointwise(before)
    return r2_score(after['NormRank'], after['Pred'])

def create_model_nn_classifier(trial):
    n_layers = trial.suggest_int("n_layers",1,4)
    reg_model = Sequential()
    for i in range(n_layers):
        num_hidden = trial.suggest_int("n_units_l{}".format(i), 4, 3000, log=True)
        if i == 0:
            reg_model.add(Dense(num_hidden, input_dim=300, kernel_initializer='random_normal', activation='relu'))
        else:
            reg_model.add(
                Dense(
                    num_hidden,
                    activation=trial.suggest_categorical('activation', ['relu', 'sigmoid']),
                    kernel_initializer='random_normal'
                )
            )
    reg_model.add(Dense(1, kernel_initializer='random_normal',activation=trial.suggest_categorical('activation', ['relu', 'sigmoid'])))
    reg_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return reg_model

def objective_nn_classifier(trial):
    df = pd.read_csv("dataset1.csv")
    train, test, to_rank = create_training_set_pair(df)

    reg_model = create_model_nn_classifier(trial)

    # Train
    train_builds = train['Build'].unique().tolist()
    print("Training sets: " + str(train_builds))
    for build in train_builds:
        temp_train_build = pd.DataFrame(train.loc[train['Build'] == build].copy()).reset_index(drop=True)
        temp_train_build = class_pairs(create_pairs(temp_train_build))
        x_train = temp_train_build.loc[:, ~temp_train_build.columns.isin(
            ['Class', 'Verdict_A', 'Verdict_B', 'Duration_A', 'Duration_B', 'Build_A', 'Test_A', 'Test_B'])]
        y_train = temp_train_build.loc[:, temp_train_build.columns.isin(['Class'])]
        reg_model.fit(x_train, y_train.values.ravel(), batch_size=400, epochs=100, verbose=0)
        #x_train.to_csv('xtrain.csv', index=False)
    # Test
    test_builds = test['Build'].unique().tolist()
    result = pd.DataFrame()
    print("Testing sets: " + str(test_builds))
    print(str(datetime.datetime.now()))
    for build in test_builds:
        temp_test_build = pd.DataFrame(test.loc[test['Build'] == build].copy()).reset_index(drop=True)
        temp_test_build = class_pairs(create_pairs(temp_test_build))
        x_test = temp_test_build.loc[:, ~temp_test_build.columns.isin(
            ['Class', 'Verdict_A', 'Verdict_B', 'Duration_A', 'Duration_B', 'Build_A', 'Test_A', 'Test_B'])]
        y_test = temp_test_build.loc[:, temp_test_build.columns.isin(['Class'])]
        comparison = temp_test_build.loc[:, temp_test_build.columns.isin(
            ['Build_A', 'Test_A', 'Duration_A', 'Verdict_A', 'Test_B', 'Duration_B', 'Verdict_B'])]
        # print(cur_test.head())
        predictions = (reg_model.predict(x_test) > 0.5).astype(int)
        compare = pd.DataFrame(comparison.loc[
                                   comparison['Build_A'] == build, ['Build_A', 'Test_A', 'Duration_A', 'Verdict_A',
                                                                    'Test_B', 'Duration_B', 'Verdict_B']].copy())
        compare['PredClass'] = predictions
        compare['Class'] = y_test
        result = result.append(compare)
        #x_test.to_csv('xtest.csv', index=False)

    print("Ranking")
    print(str(datetime.datetime.now()))
    ranked = pd.DataFrame()
    for build in test_builds:
        temp_build = result.loc[result['Build_A'] == build]
        temp_rank = to_rank.loc[to_rank['Build'] == build].copy()
        temp_rank['PredRank'] = 0
        for index, row in temp_build.iterrows():
            if row['PredClass'] == 0:
                temp_rank.loc[temp_rank['Test'] == row['Test_B'], 'PredRank'] += 1
        temp_rank.sort_values(
            ['PredRank'],
            ascending=[True],
            inplace=True,
            ignore_index=True,
        )
        ranked = ranked.append(temp_rank, ignore_index=True)
    print("Finish")
    print(str(datetime.datetime.now()))
    #result.to_csv("resultsPair.csv", index=False)
    #ranked.to_csv('rankedPair.csv', index=False)
    return accuracy_score(result['Class'], result['PredClass'])

def compute_apfdc_loop(output_path):
    i = 0
    while i < 25:
        i = i + 1
        dataset_path = output_path / f"dataset{i}.csv"
        if not dataset_path.exists():
            print(f"No dataset{i}.csv found in the output directory.")
            continue
        print(f"##### Evaluating dataset{i}.csv #####")
        dataset_df = pd.read_csv(dataset_path)
        apfdc = compute_apfdc(dataset_df)
        results = 'results'
        results_path = output_path / results
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        new_dataset = results_path / f"result{i}.csv"
        apfdc.to_csv(new_dataset, index=False)

def collect_apfdc_loop(output_path):
    i = 0
    apfdcs = pd.DataFrame(columns=['Dataset','Mean_APFDc'])
    while i < 25:
        i = i + 1
        dataset_path = output_path / f"result{i}.csv"
        if not dataset_path.exists():
            print(f"No result{i}.csv found in the output directory.")
            continue
        dataset_df = pd.read_csv(dataset_path)
        mean = mean_apfdc(dataset_df)
        apfdcs = apfdcs.append({'Dataset': f"result{i}", 'Mean_APFDc': mean}, ignore_index=True)
    results_path = output_path
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    new_dataset = results_path / f"collect_apfdc.csv"
    apfdcs.to_csv(new_dataset, index=False)