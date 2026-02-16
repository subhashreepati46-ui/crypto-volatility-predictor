from src.data.load_data import load_dataset
from src.data.preprocess import clean_data
from src.data.feature_engineering import add_features
from src.models.train import train_model
from src.models.evaluate import evaluate

def run():
    df = load_dataset()
    df = clean_data(df)
    df = add_features(df)

    X_test, y_test, symbols, model = train_model(df)
    evaluate(model, X_test, y_test, symbols)


if __name__ == "__main__":
    run()
