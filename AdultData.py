import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

def preprocess_data(adultinfo):
    print("Initial dataset shape: ", adultinfo.shape)
    
    adultinfo = adultinfo.dropna()
    
    print("Dataset shape after dropping missing values: ", adultinfo.shape)
    scaler = StandardScaler()
    
    continuous_features = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    adultinfo[continuous_features] = scaler.fit_transform(adultinfo[continuous_features])
    print("Continuous features after scaling: ")
    
    print(adultinfo[continuous_features].head())
    categorical_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    adultinfo = pd.get_dummies(adultinfo, columns=categorical_features, drop_first=True)
    
    print("Dataset shape after converting categorical variables: ", adultinfo.shape)
    return adultinfo
    

def select_features(adultinfo):
    relevant_features = [
        'age', 'education-num', 'hours-per-week'
    ] + [col for col in adultinfo.columns if col.startswith('workclass_') or col.startswith('occupation_') or col.startswith('race_') or col.startswith('sex_')]
    X = adultinfo.drop("income", axis=1)
    y = adultinfo["income"].apply(lambda x: 1 if x == " >50K" else 0)
    return X[relevant_features], y


def logistic_regression_model(train_data, train_labels):
    model = LogisticRegression(max_iter=1000)
    model.fit(train_data, train_labels)
    return model

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def plot_metrics(metrics, model_names, metric_name):
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    
    plt.bar(model_names, [m[0] for m in metrics], color=['blue', 'green', 'red'])
    
    plt.title(f'{metric_name} - Bar Graph')
    
    plt.xlabel('Model')
    
    plt.ylabel(metric_name)
    
    plt.subplot(1, 2, 2)
    
    plt.plot(model_names, [m[0] for m in metrics], marker='o')
    
    plt.title(f'{metric_name} - Line Graph')
    
    plt.xlabel('Model')
    
    plt.ylabel(metric_name)
    
    plt.tight_layout()
    
    plt.show()


def decision_tree_model(train_data, train_labels):
    model = DecisionTreeClassifier()
    model.fit(train_data, train_labels)
    return model

def evaluate_model(model, test_data, test_labels):
    predictions = model.predict(test_data)
    
    accuracy = accuracy_score(test_labels, predictions)
    
    precision = precision_score(test_labels, predictions)
    
    recall = recall_score(test_labels, predictions)
    return accuracy, precision, recall



def random_forest_model(train_data, train_labels):
    model = RandomForestClassifier()
    
    model.fit(train_data, train_labels)
    return model



def main():
    adultinfo = pd.read_csv("adult.csv", header=None)
    adultinfo.columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'income'
    ]
    print("Dataset loaded. Preprocessing data...")
    preprocessed_data = preprocess_data(adultinfo)

    features, labels = select_features(preprocessed_data)
    
    train_data, test_data, train_labels, test_labels = split_data(features, labels)
    
    logistic_model = logistic_regression_model(train_data, train_labels)
    
    decision_tree = decision_tree_model(train_data, train_labels)
    
    random_forest = random_forest_model(train_data, train_labels)
    
    log_reg_metrics = evaluate_model(logistic_model, test_data, test_labels)
    
    print("Logistic Regression Performance: ", log_reg_metrics)
    
    decision_tree_metrics = evaluate_model(decision_tree, test_data, test_labels)
    
    print("Decision Tree Performance: ", decision_tree_metrics)
    
    random_forest_metrics = evaluate_model(random_forest, test_data, test_labels)
    
    print("Random Forest Performance: ", random_forest_metrics)
    
    model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest']
    
    metrics = [log_reg_metrics, decision_tree_metrics, random_forest_metrics]
    
    plot_metrics([(m[0],) for m in metrics], model_names, 'Accuracy')
    
    plot_metrics([(m[1],) for m in metrics], model_names, 'Precision')
    
    plot_metrics([(m[2],) for m in metrics], model_names, 'Recall')

main()
