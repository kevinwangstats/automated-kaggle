import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier
import re

def train_and_evaluate():
    # 1. Load Data
    df = pd.read_csv("data/titanic/train.csv")
    target_col = "Survived"
    
    # Basic Preprocessing
    df = df.dropna(subset=[target_col])
    
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Clean column names
    X.columns = [re.sub(r'[^\w\s]', '', col).replace(' ', '_') for col in X.columns]
    
    # Feature Engineering
    X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
    X['IsAlone'] = (X['FamilySize'] == 1).astype(int)
    
    # Extract title from Name
    X['Title'] = X['Name'].apply(lambda name: re.search(' ([A-Za-z]+)\.', name).group(1) if re.search(' ([A-Za-z]+)\.', name) else "")
    # Map rare titles to a common category
    X['Title'] = X['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    X['Title'] = X['Title'].replace('Mlle', 'Miss')
    X['Title'] = X['Title'].replace('Ms', 'Miss')
    X['Title'] = X['Title'].replace('Mme', 'Mrs')
    
    # Drop original columns that are now represented by engineered features or are less useful
    X = X.drop(columns=['Name', 'SibSp', 'Parch', 'PassengerId'])

    # Handle the 'Ticket' column: it's causing the CatBoostError.
    # The error "Bad value for num_feature... Cannot convert 'C85' to float" indicates that
    # the 'Ticket' column, which contains alphanumeric strings, is being treated as a numerical
    # feature by CatBoost. Since we are not explicitly using it and it has high cardinality,
    # dropping it is a reasonable approach to fix this error.
    if 'Ticket' in X.columns:
        X = X.drop(columns=['Ticket'])

    # Define categorical and numerical features
    # 'Pclass' is numerical but ordinal, so it can be treated as numerical.
    categorical_features = ['Sex', 'Embarked', 'Title']
    numerical_features = ['Pclass', 'Age', 'Fare', 'FamilySize']

    # Create preprocessing pipelines for numerical and categorical features
    # For Age, median imputation is good. For Fare, median is also robust to outliers.
    # For Embarked, most_frequent imputation is suitable.
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a column transformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep other columns (if any)
    )

    # 2. Model Initialization
    # Using CatBoostClassifier with some tuned parameters
    # Increased iterations, adjusted learning rate, and added more regularization.
    # Also set verbose to False to avoid excessive output during CV.
    model = CatBoostClassifier(
        iterations=2000, # Increased iterations for potentially better performance
        learning_rate=0.02, # Slightly reduced learning rate for more stable convergence
        depth=8, # Increased depth to capture more complex interactions
        l2_leaf_reg=5, # Increased L2 regularization for better generalization
        loss_function='Logloss',
        eval_metric='AUC',
        random_state=42,
        verbose=False, # Set to False to suppress training output
        early_stopping_rounds=100, # Increased early stopping rounds
        subsample=0.8, # Use 80% of data for training each tree
        colsample_bylevel=0.8, # Use 80% of features for each level of the tree
        border_count=64 # Reduced border count for potentially faster training and better generalization
    )
    
    # Create the full pipeline including preprocessing and model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])

    # 3. Cross Validation
    cv = KFold(n_splits=10, shuffle=True, random_state=42) # Increased folds for more robust evaluation
    scoring = 'roc_auc'
    
    # Use the pipeline for cross-validation
    # The previous error was due to the 'Ticket' column being passed to CatBoost as a numerical feature.
    # By dropping 'Ticket', this error should be resolved.
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    final_score = np.mean(scores)
        
    print(f"FINAL_CV_SCORE: {final_score:.4f}")
    
    # 4. Generate Submission
    print("Generating submission...")
    # Fit the pipeline on the entire training data
    pipeline.fit(X, y)
    
    test_df = pd.read_csv("data/titanic/test.csv")
    # Simple preprocessing matching train
    test_X = test_df.copy()
    test_X.columns = [re.sub(r'[^\w\s]', '', col).replace(' ', '_') for col in test_X.columns]

    # Apply the same feature engineering as on the training data
    test_X['FamilySize'] = test_X['SibSp'] + test_X['Parch'] + 1
    test_X['IsAlone'] = (test_X['FamilySize'] == 1).astype(int)
    test_X['Title'] = test_X['Name'].apply(lambda name: re.search(' ([A-Za-z]+)\.', name).group(1) if re.search(' ([A-Za-z]+)\.', name) else "")
    test_X['Title'] = test_X['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    test_X['Title'] = test_X['Title'].replace('Mlle', 'Miss')
    test_X['Title'] = test_X['Title'].replace('Ms', 'Miss')
    test_X['Title'] = test_X['Title'].replace('Mme', 'Mrs')
    
    # Drop original columns that are now represented by engineered features or are less useful
    test_X = test_X.drop(columns=['Name', 'SibSp', 'Parch', 'PassengerId'])

    # Drop 'Ticket' from test_X as it was dropped from train_X
    if 'Ticket' in test_X.columns:
        test_X = test_X.drop(columns=['Ticket'])
        
    # Ensure all columns used in training are present in test, filling missing ones with appropriate defaults
    # This is important if some categories in test are not present in train after one-hot encoding.
    # However, with handle_unknown='ignore' in OneHotEncoder, this might not be strictly necessary
    # for the categorical features, but it's good practice for numerical features if they were missing.
    # For this specific dataset and preprocessing, it's less likely to be an issue.

    # Predict on the test data. We ALWAYS output probabilities to raw_submission.csv.
    # The formatting into submission.csv (classes vs probs) is handled by kaggle_submit.py.
    if hasattr(pipeline, "predict_proba"):
        preds = pipeline.predict_proba(test_X)[:, 1]
    else:
        preds = pipeline.predict(test_X)
    
    submission = pd.DataFrame()
    # Assuming first column of test is ID
    submission[test_df.columns[0]] = test_df.iloc[:, 0]
    submission['Survived'] = preds
    submission.to_csv("raw_submission.csv", index=False)
    print("Saved raw_submission.csv")
    
    # Automatically format submission using the separate kaggle_submit.py script
    import subprocess
    import yaml
    import os
    cfg = "tests/titanic_config.yaml" if "titanic" in "data/titanic/train.csv" and os.path.exists("tests/titanic_config.yaml") else "config.yaml"
    auto_submit = False
    if os.path.exists(cfg):
        try:
            with open(cfg, "r") as f:
                cfg_data = yaml.safe_load(f)
                auto_submit = cfg_data.get("auto_kaggle_submit", False)
        except Exception:
            pass
    if auto_submit:
        print("Formatting submission for Kaggle...")
        subprocess.run(["python", "kaggle_submit.py", "--config", cfg])

    return final_score

if __name__ == "__main__":
    train_and_evaluate()