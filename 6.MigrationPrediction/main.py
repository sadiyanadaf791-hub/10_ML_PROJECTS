from src.data_preprocessing import load_and_preprocess
from src.train_model import train_model
from src.evaluate import evaluate_model

X, y = load_and_preprocess("data/migration_nz.csv")
model, X_test, y_test = train_model(X, y)
evaluate_model(model, X_test, y_test)
