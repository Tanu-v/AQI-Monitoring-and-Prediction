import pytest
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from app import df, all_features

# Fixtures
@pytest.fixture
def features():
    return df[all_features]

@pytest.fixture
def target_encoded():
    le = LabelEncoder()
    return le.fit_transform(df['AQI_Bucket'].astype(str)), le

@pytest.fixture
def scaled_features(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)

@pytest.fixture
def trained_model(features, target_encoded):
    X = features
    y, _ = target_encoded
    clf = RandomForestClassifier(n_estimators=15, random_state=123)
    clf.fit(X, y)
    return clf

# Tests
def test_dataframe_not_empty():
    assert not df.empty, "DataFrame should not be empty."

def test_no_missing_aqi_bucket():
    assert df['AQI_Bucket'].isnull().sum() == 0, "No missing AQI_Bucket values allowed."

def test_features_standardized(scaled_features):
    mean_vals = np.mean(scaled_features, axis=0)
    assert np.all(np.abs(mean_vals) < 0.1), "Features are not centered around 0."

def test_labels_consecutive(target_encoded):
    y, le = target_encoded
    assert set(y) == set(range(len(le.classes_))), "Labels are not consecutive integers."

def test_model_predicts_all(trained_model, features, target_encoded):
    y, _ = target_encoded
    preds = trained_model.predict(features)
    assert len(preds) == len(y), "Model should predict for all samples."
