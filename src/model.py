import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def train_and_save_model(data_path, models_dir):
    df = pd.read_csv(data_path)
    
    df = df.dropna()
    
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), X.columns.tolist())
        ])

    base_estimator = DecisionTreeClassifier(max_depth=2, random_state=42)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', AdaBoostClassifier(estimator=base_estimator, algorithm='SAMME', random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.1, 1.0]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_pipeline = grid_search.best_estimator_

    os.makedirs(models_dir, exist_ok=True)
    artifact_path = os.path.join(models_dir, 'multibagger_adaboost_pipeline.pkl')
    
    with open(artifact_path, 'wb') as f:
        pickle.dump(best_pipeline, f)
        
    return best_pipeline, X_test, y_test

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'cleaned_fundamentals.csv')
    models_dir = os.path.join(base_dir, 'models')
    train_and_save_model(data_path, models_dir)