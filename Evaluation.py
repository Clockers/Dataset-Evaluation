import pandas
import math
from pandas import DataFrame
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC
import os

# data folder : datasets for evaluation
path_original = './datasets/original/'
path_generated = './datasets/generated/'
path_results = './datasets/results/'


def get_datasets():
    return list(filter(lambda file: file.endswith('.csv'), os.listdir(path_generated)))

datasets = get_datasets()

algorithms = ['RandomForest100', 'RandomForest30', 'KNeighbors35', 'KNeighbors1', 'AdaBoost100', 'AdaBoost30',
              'LinearDiscriminantAnalysis', 'GradientBoosting', 'MLP', 'LogisticRegression']


result_columns = ['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC']
result_columns_dataset = ['Dataset', 'Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC']


def main():

    for dataset in datasets:
        results_classification_real = None
        results_classification_synth = None
        results_detection = None

        real_data = pandas.read_csv(path_original + dataset)
        real_data = remove_columns(real_data, dataset)

        sampled_data = pandas.read_csv(path_generated + dataset)

        # Remove string column
        real_data = remove_str_columns(real_data, dataset)
        sampled_data = remove_str_columns(sampled_data, dataset)

        sampled_data = pandas.DataFrame(sampled_data)
        real_data = pandas.DataFrame(real_data)

        train_real, test_real = split_dataset(real_data, 0.7)
        classification_real = evaluate_classification(train_real, test_real)
        classification_synth = evaluate_classification(real_data, sampled_data)
        detection = evaluate_detection(real_data, sampled_data)
        classification_real.insert(0, 'Dataset', dataset)
        classification_synth.insert(0, 'Dataset', dataset)
        detection.insert(0, 'Dataset', dataset)

        if results_classification_real is None:
            results_classification_real = pandas.DataFrame(classification_real)
        else:
            results_classification_real = results_classification_real.append(pandas.DataFrame(classification_real),
                                                                   ignore_index=True)
        if results_classification_synth is None:
            results_classification_synth = pandas.DataFrame(classification_synth)
        else:
            results_classification_synth = results_classification_synth.append(pandas.DataFrame(classification_synth),
                                                                   ignore_index=True)
        if results_detection is None:
            results_detection = pandas.DataFrame(detection)
        else:
            results_detection = results_detection.append(pandas.DataFrame(detection),
                                                         ignore_index=True)

        results_classification_real.to_csv(path_results + dataset + 'ClassificationRealEval.csv', index=False)
        results_classification_synth.to_csv(path_results + dataset + 'ClassificationSynthEval.csv', index=False)
        results_detection.to_csv(path_results + dataset + 'DetectionEval.csv', index=False)

    print('Result saved Eval')


def split_dataset(dataset: DataFrame, percent: float):
    split_index = math.ceil(len(dataset) * percent)
    train = dataset[0: split_index]
    test = dataset[split_index: len(dataset)]
    return train, test


def get_classifier(classifier_name):
    classifier = None
    if classifier_name == 'RandomForest105':
        classifier = RandomForestClassifier(n_estimators=105)
    elif classifier_name == 'RandomForest35':
        classifier = RandomForestClassifier(n_estimators=35)
    elif classifier_name == 'KNeighbors35':
        classifier = KNeighborsClassifier(n_neighbors=35)
    elif classifier_name == 'KNeighbors1':
        classifier = KNeighborsClassifier(n_neighbors=1)
    elif classifier_name == 'AdaBoost105':
        classifier = AdaBoostClassifier(n_estimators=105)
    elif classifier_name == 'AdaBoost35':
        classifier = AdaBoostClassifier(n_estimators=35)
    elif classifier_name == 'MLP':
        classifier = MLPClassifier()
    elif classifier_name == 'LinearDiscriminantAnalysis':
        classifier = LinearDiscriminantAnalysis()
    elif classifier_name == 'LogisticRegression':
        classifier = LogisticRegression()
    elif classifier_name == 'GradientBoosting':
        classifier = GradientBoostingClassifier()
    else:
        classifier = None

    return classifier


def remove_columns(data, name):
    if name == 'ADFANet_Shuffled.csv':
        data.drop('Time1', inplace=True, axis=1)
        data.drop('Time2', inplace=True, axis=1)

    elif name == 'AndMal_Shuffled.csv':
        data.drop('Timestamp', inplace=True, axis=1)
        data.drop('Flow ID', inplace=True, axis=1)
        data.drop('Source IP', inplace=True, axis=1)
        data.drop('Destination IP', inplace=True, axis=1)

    elif name == 'CICIDS17_Shuffled_Reduced.csv':
        data.drop('Flow_ID', inplace=True, axis=1)
        data.drop('Source_IP', inplace=True, axis=1)
        data.drop('Destination_IP', inplace=True, axis=1)
        data.drop('Timestamp', inplace=True, axis=1)

    elif name == 'CIDDS_Shuffled.csv':
        data.drop('Date_first_seen', inplace=True, axis=1)
        data.drop('Src_IP_Addr', inplace=True, axis=1)
        data.drop('Dst_IP_Addr', inplace=True, axis=1)

    elif name == 'CTU_Shuffled.csv':
        data.drop('Details', inplace=True, axis=1)
        data.drop('Dir', inplace=True, axis=1)
        data.drop('StartTime', inplace=True, axis=1)
        data.drop('SrcAddr', inplace=True, axis=1)
        data.drop('DstAddr', inplace=True, axis=1)

    elif name == 'ISCX_Shuffled.csv':
        data.drop('source', inplace=True, axis=1)
        data.drop('destination', inplace=True, axis=1)
        data.drop('startDateTime', inplace=True, axis=1)
        data.drop('stopDateTime', inplace=True, axis=1)

    elif name == 'NGDIS_Shuffled.csv':
        data.drop('Date', inplace=True, axis=1)
        data.drop('Time', inplace=True, axis=1)

    elif name == 'UGR_Shuffled.csv':
        data.drop('Timestamp', inplace=True, axis=1)
        data.drop('IP_S', inplace=True, axis=1)
        data.drop('IP_D', inplace=True, axis=1)

    return data


def remove_str_columns(data, name):

    if name == 'ADFANet_Shuffled.csv':
        data.drop('IP_1', inplace=True, axis=1)
        data.drop('IP_2', inplace=True, axis=1)
        data.drop('AttCat', inplace=True, axis=1)
        data = data.fillna('Empty')

    elif name == 'CICIDS18_Reduced.csv':
        data.drop('Timestamp', inplace=True, axis=1)

    elif name == 'CICIDS18_Shuffled_Reduced.csv':
        data.drop('Timestamp', inplace=True, axis=1)

    elif name == 'CIDDS_Shuffled.csv':
        data.drop('Proto', inplace=True, axis=1)
        data.drop('Flags', inplace=True, axis=1)

    elif name == 'CTU_Shuffled.csv':
        data.drop('State', inplace=True, axis=1)
        data.drop('Proto', inplace=True, axis=1)

    elif name == 'ISCX_Shuffled.csv':
        data.drop('appName', inplace=True, axis=1)
        data.drop('direction', inplace=True, axis=1)
        data.drop('sourceTCPFlagsDescription', inplace=True, axis=1)
        data.drop('destinationTCPFlagsDescription', inplace=True, axis=1)
        data.drop('protocolName', inplace=True, axis=1)

    elif name == 'NGDIS_Shuffled.csv':
        data.drop('Detail', inplace=True, axis=1)

    elif name == 'NSLKDD_All.csv':
        data.drop('protocol_type', inplace=True, axis=1)
        data.drop('service', inplace=True, axis=1)
        data.drop('flag', inplace=True, axis=1)

    elif name == 'UGR_Shuffled.csv':
        data.drop('Detail', inplace=True, axis=1)
        data.drop('Protocol', inplace=True, axis=1)

    elif name == 'UNSW_Shuffled.csv':
        data.drop('proto', inplace=True, axis=1)
        data.drop('service', inplace=True, axis=1)
        data.drop('state', inplace=True, axis=1)

    return data


def evaluate_classification(real_data, sampled_data):
    columns = real_data.columns.values

    x_train = sampled_data[columns[:-1]]
    y_train = sampled_data[columns[-1]]
    x_test = real_data[columns[:-1]]
    y_test = real_data[columns[-1]]

    results = None

    for algorithm in algorithms:
        result = []
        classifier = get_classifier(algorithm)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)

        result.append(algorithm)
        result.append(metrics.accuracy_score(y_test, y_pred))
        result.append(metrics.precision_score(y_test, y_pred, average='weighted'))
        result.append(metrics.recall_score(y_test, y_pred, average='weighted'))
        result.append(metrics.f1_score(y_test, y_pred, average='weighted'))
        result.append(metrics.matthews_corrcoef(y_test, y_pred))

        if results is None:
            results = pandas.DataFrame([result], columns=result_columns)
        else:
            results = results.append(pandas.DataFrame([result], columns=result_columns), ignore_index=True)

    return results


def evaluate_detection(real_data, sampled_data):
    columns = real_data.columns.values
    real_data = real_data[columns[:-1]]
    sampled_data = sampled_data[columns[:-1]]

    sampled_data["dataset"] = "sampled"
    real_data["dataset"] = "real"

    dataset = pandas.concat([sampled_data, real_data], ignore_index=True).sample(frac=1).reset_index(drop=True)
    columns = dataset.columns.values

    train, test = split_dataset(dataset, 0.7)

    x_train = train[columns[:-1]]
    y_train = train[columns[-1]]
    x_test = test[columns[:-1]]
    y_test = test[columns[-1]]

    results = pandas.DataFrame(columns=result_columns)

    for algorithm in algorithms:
        result = []
        classifier = get_classifier(algorithm)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)

        result.append(algorithm)
        result.append(metrics.accuracy_score(y_test, y_pred))
        result.append(metrics.precision_score(y_test, y_pred, average='weighted'))
        result.append(metrics.recall_score(y_test, y_pred, average='weighted'))
        result.append(metrics.f1_score(y_test, y_pred, average='weighted'))
        result.append(metrics.matthews_corrcoef(y_test, y_pred))

        results.append(pandas.DataFrame(result))

        if results is None:
            results = pandas.DataFrame([result], columns=result_columns)
        else:
            results = results.append(pandas.DataFrame([result], columns=result_columns), ignore_index=True)

    return results

# Categorical or boolean columns for CSTest and numerical columns for KSTest
# def evaluate_similarity(real_data, sampled_data):
#    results = pandas.DataFrame([['KSTest', KSTest.compute(real_data, sampled_data)]], columns=['Algorithm', 'Result'])
#    results = results.append(pandas.DataFrame([['CSTest', CSTest.compute(real_data, sampled_data)]], columns=['Algorithm', 'Result']), ignore_index=True)
#    return results

if __name__ == "__main__":
    main()
