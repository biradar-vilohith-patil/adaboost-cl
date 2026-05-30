import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier

def train_and_save_model(data_path, models_dir):
    df = pd.read_csv(data_path)
    
    df = df.dropna()
    
    X = df[['ssc_p', 'hsc_p', 'degree_p', 'workex', 'specialisation']]
    y = df['status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['ssc_p', 'hsc_p', 'degree_p']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['workex', 'specialisation'])
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', AdaBoostClassifier(algorithm='SAMME', random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators': [50, 100, 150],
        'classifier__learning_rate': [0.1, 0.5, 1.0]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_pipeline = grid_search.best_estimator_

    os.makedirs(models_dir, exist_ok=True)
    artifact_path = os.path.join(models_dir, 'placement_adaboost_pipeline.pkl')
    
    with open(artifact_path, 'wb') as f:
        pickle.dump(best_pipeline, f)
        
    return best_pipeline, X_test, y_test

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'cleaned_placement.csv')
    models_dir = os.path.join(base_dir, 'models')
    train_and_save_model(data_path, models_dir)import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier

def train_and_save_model(data_path, models_dir):
    df = pd.read_csv(data_path)
    
    df = df.dropna()
    
    X = df[['ssc_p', 'hsc_p', 'degree_p', 'workex', 'specialisation']]
    y = df['status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['ssc_p', 'hsc_p', 'degree_p']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['workex', 'specialisation'])
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', AdaBoostClassifier(algorithm='SAMME', random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators': [50, 100, 150],
        'classifier__learning_rate': [0.1, 0.5, 1.0]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_pipeline = grid_search.best_estimator_

    os.makedirs(models_dir, exist_ok=True)
    artifact_path = os.path.join(models_dir, 'placement_adaboost_pipeline.pkl')
    
    with open(artifact_path, 'wb') as f:
        pickle.dump(best_pipeline, f)
        
    return best_pipeline, X_test, y_test

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'cleaned_placement.csv')
    models_dir = os.path.join(base_dir, 'models')
    train_and_save_model(data_path, models_dir)