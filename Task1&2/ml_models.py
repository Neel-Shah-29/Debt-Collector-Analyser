from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from utils import annotate_with_similarity_orderaware

def train_ml_with_similarity(convs: dict, task: str):
    """
    Auto-annotate with semantic+order-aware rules, then train a logistic model.
    Returns None if only one class is present.
    """
    records = annotate_with_similarity_orderaware(convs)
    X = [r["text"] for r in records]
    y = [r[task] for r in records]

    # If only one class present, skip training
    if len(set(y)) < 2:
        return None

    pipe = Pipeline([
        ("vect", CountVectorizer()),
        ("clf", LogisticRegression(max_iter=500))
    ])
    pipe.fit(X, y)
    return pipe

