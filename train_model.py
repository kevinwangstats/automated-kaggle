import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import re
import json

def train_and_evaluate():
    # 1. Load Data
    df = pd.read_csv("data/titanic/train.csv")
    test_df = pd.read_csv("data/titanic/test.csv")
    
    target_col = "Survived"
    
    # Basic Preprocessing
    df = df.dropna(subset=[target_col])
    y_raw = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Align columns - crucial for consistent processing
    train_ids = X['PassengerId']
    test_ids = test_df['PassengerId']
    
    # Drop PassengerId as it's not a feature
    X = X.drop('PassengerId', axis=1)
    test_df = test_df.drop('PassengerId', axis=1)

    # Label encode target
    le_y = LabelEncoder()
    y = le_y.fit_transform(y_raw)

    # 2. Feature Engineering
    def feature_engineer(df, training_data_stats=None):
        # Create a dictionary to store stats from training data if not provided
        if training_data_stats is None:
            training_data_stats = {}

        # Impute Embarked with mode
        if 'embarked_mode' not in training_data_stats:
            training_data_stats['embarked_mode'] = df['Embarked'].mode()[0]
        df['Embarked'].fillna(training_data_stats['embarked_mode'], inplace=True)

        # Impute Fare with median
        if 'fare_median' not in training_data_stats:
            training_data_stats['fare_median'] = df['Fare'].median()
        df['Fare'].fillna(training_data_stats['fare_median'], inplace=True)
        
        # Log transform Fare to reduce skewness
        df['Fare'] = np.log1p(df['Fare'])

        # Extract Title from Name
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')

        # Create FamilySize and IsAlone features
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

        # Extract Deck from Cabin
        df['Deck'] = df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'U')

        # Impute Age based on the median age for each Title
        if 'title_age_median' not in training_data_stats:
            training_data_stats['title_age_median'] = df.groupby('Title')['Age'].median().to_dict()
        
        title_age_map = training_data_stats['title_age_median']
        df['Age'] = df.apply(
            lambda row: title_age_map.get(row['Title']) if pd.isnull(row['Age']) else row['Age'],
            axis=1
        )
        # Fallback for any titles in test set not in train set
        if df['Age'].isnull().any():
            if 'global_age_median' not in training_data_stats:
                training_data_stats['global_age_median'] = df['Age'].median()
            df['Age'].fillna(training_data_stats['global_age_median'], inplace=True)

        # New Feature: Age * Pclass
        df['Age_Pclass'] = df['Age'] * df['Pclass']

        # New Feature: Ticket Prefix
        def get_ticket_prefix(ticket):
            ticket = ticket.replace('.', '').replace('/', '').split()
            prefix = [t for t in ticket if not t.isdigit()]
            if not prefix:
                return 'X'
            return ''.join(prefix)
        df['Ticket_Prefix'] = df['Ticket'].apply(get_ticket_prefix)

        # Drop original/unnecessary columns
        df = df.drop(['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1)
        
        return df, training_data_stats

    # Apply feature engineering
    X, training_stats = feature_engineer(X)
    test_X, _ = feature_engineer(test_df, training_stats)

    # 3. Define Preprocessing Pipeline
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # 4. Model Initialization (with tuned hyperparameters)
    # Using slightly more complex models with lower learning rate
    xgb_params = {
        'n_estimators': 500, 'learning_rate': 0.02, 'max_depth': 4,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42,
        'use_label_encoder': False, 'eval_metric': 'logloss', 'n_jobs': -1
    }
    lgb_params = {
        'n_estimators': 500, 'learning_rate': 0.02, 'num_leaves': 20,
        'max_depth': 4, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'random_state': 42, 'n_jobs': -1, 'verbose': -1, 'reg_alpha': 0.1, 'reg_lambda': 0.1
    }
    cat_params = {
        'iterations': 500, 'learning_rate': 0.02, 'depth': 4,
        'l2_leaf_reg': 4, 'random_state': 42, 'verbose': 0,
        'loss_function': 'Logloss'
    }

    clf1 = XGBClassifier(**xgb_params)
    clf2 = LGBMClassifier(**lgb_params)
    clf3 = CatBoostClassifier(**cat_params)
    
    # Define base estimators for Stacking
    estimators = [
        ('xgb', clf1),
        ('lgb', clf2),
        ('cat', clf3)
    ]

    # Define meta-classifier
    meta_classifier = LogisticRegression(C=1.0, random_state=42)

    # Create the StackingClassifier
    stacker = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_classifier,
        cv=KFold(n_splits=5, shuffle=True, random_state=1), # Internal CV for stacking
        stack_method='predict_proba',
        n_jobs=-1
    )

    # 5. Create Full Pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', stacker)])
    
    # 6. Cross Validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = 'roc_auc'
    
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    final_score = np.mean(scores)
    print(f"CV Score: {final_score}")
    
    with open("metrics.json", "w") as f:
        json.dump({"cv_score": final_score}, f)

    # 7. Generate Submission
    print("Generating submission...")
    pipeline.fit(X, y)
    
    # Ensure test columns match train columns before prediction
    # This is handled by the feature engineering function and pipeline
    
    preds = pipeline.predict_proba(test_X)[:, 1]
            
    submission = pd.DataFrame({'PassengerId': test_ids, 'Survived': preds})
    submission.to_csv("raw_submission.csv", index=False)
    print("Saved raw_submission.csv")

    return final_score

if __name__ == "__main__":
    train_and_evaluate()