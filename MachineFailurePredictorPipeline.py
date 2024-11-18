import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Step 3: Filter necessary columns
def filter_data():
    data = pd.read_csv('ai4i2020.csv')
    selected_data = data[['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
                          'Torque [Nm]', 'Tool wear [min]', 'Machine failure']]
    return selected_data

# Step 4: Data preprocessing 
def preprocess_data():
    selected_data = filter_data()
    encoder = LabelEncoder()
    selected_data['Type'] = encoder.fit_transform(selected_data['Type'])  
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(selected_data.drop('Machine failure', axis=1))
    return pd.DataFrame(scaled_data, columns=selected_data.drop('Machine failure', axis=1).columns)

# Step 5: Data balancing using RandomUnderSampler
def balance_data():
    selected_data = filter_data()
    scaled_data = preprocess_data()
    X = scaled_data
    y = selected_data['Machine failure']
    rus = RandomUnderSampler(sampling_strategy={0: 339, 1: 339}, random_state=42)
    X_res, y_res = rus.fit_resample(X, y)
    balanced_data = pd.DataFrame(X_res, columns=scaled_data.columns)
    balanced_data['Machine failure'] = y_res
    return balanced_data

# Part 6: 5-fold cross-validation for model tuning
def cross_validation(X_train, y_train):
    param_grids = {
        'MLP': {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (150,)],
            'activation': ['relu', 'tanh'],
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [2000],
            'early_stopping': [True],
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        },
        'KNN': {
            'n_neighbors': [3, 5, 7, 10],
            'p': [1, 2],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        },
        'DecisionTree': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 20],
            'ccp_alpha': [0.0, 0.1, 0.2]
        },
        'LogisticRegression': {
            'penalty': ['l2'],  
            'max_iter': [5000],
            'solver': ['lbfgs', 'liblinear']  
        }
    }

    models = {
        'MLP': MLPClassifier(max_iter=2000, early_stopping=True),
        'SVM': SVC(),
        'KNN': KNeighborsClassifier(),
        'DecisionTree': DecisionTreeClassifier(),
        'LogisticRegression': LogisticRegression()
    }

    results = []
    best_params = {}

    for model_name in models:
        print(f"Training {model_name} model...")
        grid_search = GridSearchCV(models[model_name], param_grids[model_name], cv=5, scoring=make_scorer(matthews_corrcoef))
        grid_search.fit(X_train, y_train)

        best_params[model_name] = grid_search.best_params_
        best_score = grid_search.best_score_

        results.append([model_name, best_params[model_name], best_score])

    results_df = pd.DataFrame(results, columns=["ML Trained Model", "Best Set of Parameter Values", "MCC Score (5-Fold CV)"])
    print("\n" + str(results_df))

    return best_params

# Part 7: Evaluate models on the test set
def evaluate_models(X_train, X_test, y_train, y_test, models, best_params):
    test_results = []

    for model_name in models:
        print(f"Evaluating {model_name} model on test set...")
        
        model = models[model_name].set_params(**best_params[model_name])
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mcc_score = matthews_corrcoef(y_test, y_pred)
        
        test_results.append([model_name, best_params[model_name], mcc_score])
    
    results_df = pd.DataFrame(test_results, columns=["ML Trained Model", "Best Set of Parameter Values", "MCC-score on Testing Data (20%)"])
    print("\n" + str(results_df))
    
    best_model_row = results_df.loc[results_df['MCC-score on Testing Data (20%)'].idxmax()]
    print(f"\nThe model with the highest MCC score is: {best_model_row['ML Trained Model']} with a score of {best_model_row['MCC-score on Testing Data (20%)']}")

def main():
    # Part 3: Filtered Data
    print("Part 3: Filtered Data")
    filtered_data = filter_data()
    print(filtered_data.head())  
    print("\n")

    # Part 4: Preprocessed Data
    print("Part 4: Preprocessed Data")
    preprocessed_data = preprocess_data()
    print(preprocessed_data.head())  
    print("\n")

    # Part 5: Balanced Data
    print("Part 5: Balanced Data")
    balanced_data = balance_data()
    print(balanced_data.head()) 
    print("\n")

    # Part 6: Cross-validation for model tuning
    print("Part 6: Cross-validation for Model Tuning")
    X = balanced_data.drop('Machine failure', axis=1)
    y = balanced_data['Machine failure']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    best_params = cross_validation(X_train, y_train)
    
    print("\n")

    # Part 7: Model Evaluation
    print("Part 7: Model Evaluation on the Test Set")
    models = {
        'MLP': MLPClassifier(),
        'SVM': SVC(),
        'KNN': KNeighborsClassifier(),
        'DecisionTree': DecisionTreeClassifier(),
        'LogisticRegression': LogisticRegression()
    }

    evaluate_models(X_train, X_test, y_train, y_test, models, best_params)
    
if __name__ == '__main__':
    main()
