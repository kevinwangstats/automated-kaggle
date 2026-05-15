import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier
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
        if training_data_stats is None:
            training_data_stats = {}

        # Impute Embarked with mode
        if 'embarked_mode' not in training_data_stats:
            training_data_stats['embarked_mode'] = df['Embarked'].mode()[0]
        df['Embarked'] = df['Embarked'].fillna(training_data_stats['embarked_mode'])

        # Impute Fare with median
        if 'fare_median' not in training_data_stats:
            training_data_stats['fare_median'] = df['Fare'].median()
        df['Fare'] = df['Fare'].fillna(training_data_stats['fare_median'])
        
        # Create FamilySize and IsAlone features
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

        # New Feature: Fare per Person (before log transform)
        df['Fare_per_Person'] = df['Fare'] / (df['FamilySize'] + 1e-6)

        # Log transform skewed features
        df['Fare'] = np.log1p(df['Fare'])
        df['Fare_per_Person'] = np.log1p(df['Fare_per_Person'])

        # Extract Title from Name
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')

        # Impute Age based on the median age for each Title
        if 'title_age_median' not in training_data_stats:
            training_data_stats['title_age_median'] = df.groupby('Title')['Age'].median().to_dict()
        
        title_age_map = training_data_stats['title_age_median']
        df['Age'] = df.apply(
            lambda row: title_age_map.get(row['Title']) if pd.isnull(row['Age']) else row['Age'],
            axis=1
        )
        # Global median as a fallback for any titles not seen in training
        if 'global_age_median' not in training_data_stats:
            training_data_stats['global_age_median'] = df['Age'].median()
        df['Age'] = df['Age'].fillna(training_data_stats['global_age_median'])

        # New Feature: Age * Pclass
        df['Age_Pclass'] = df['Age'] * df['Pclass']
        
        # New Feature: Cabin Known (strong signal)
        df['Cabin_Known'] = (df['Cabin'].notnull()).astype(int)

        # Extract Deck from Cabin and group rare decks
        df['Deck'] = df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'U')
        if 'deck_map' not in training_data_stats:
            # Grouping rare decks
            training_data_stats['deck_map'] = {'A': 'ABC', 'B': 'ABC', 'C': 'ABC', 'D': 'DE', 'E': 'DE', 'F': 'FG', 'G': 'FG', 'T': 'U'}
        df['Deck'] = df['Deck'].map(training_data_stats['deck_map']).fillna('U')

        # New Feature: Sex_Pclass interaction
        df['Sex_Pclass'] = df['Sex'].astype(str) + "_" + df['Pclass'].astype(str)

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

    # 4. Model Initialization (Tuned for speed and performance)
    xgb_params = {
        'n_estimators': 400, 'learning_rate': 0.03, 'max_depth': 3,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42,
        'use_label_encoder': False, 'eval_metric': 'logloss', 'n_jobs': -1,
        'gamma': 0.1
    }
    lgb_params = {
        'n_estimators': 400, 'learning_rate': 0.03, 'num_leaves': 15,
        'max_depth': 3, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'random_state': 42, 'n_jobs': -1, 'verbose': -1, 'reg_alpha': 0.1, 'reg_lambda': 0.1
    }
    cat_params = {
        'iterations': 400, 'learning_rate': 0.03, 'depth': 3,
        'l2_leaf_reg': 5, 'random_state': 42, 'verbose': 0,
        'loss_function': 'Logloss'
    }

    clf1 = XGBClassifier(**xgb_params)
    clf2 = LGBMClassifier(**lgb_params)
    clf3 = CatBoostClassifier(**cat_params)
    
    # Define base estimators for Voting
    estimators = [
        ('xgb', clf1),
        ('lgb', clf2),
        ('cat', clf3)
    ]

    # Use a VotingClassifier for faster ensembling
    voter = VotingClassifier(
        estimators=estimators,
        voting='soft',
        weights=[0.35, 0.35, 0.30], # Giving slightly more weight to XGB and LGBM
        n_jobs=-1
    )

    # 5. Create Full Pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', voter)])
    
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
    test_X = test_X[X.columns]
    
    preds = pipeline.predict_proba(test_X)[:, 1]
            
    submission = pd.DataFrame({'PassengerId': test_ids, 'Survived': preds})
    submission.to_csv("raw_submission.csv", index=False)
    print("Saved raw_submission.csv")

    return final_score

if __name__ == "__main__":
    train_and_evaluate()