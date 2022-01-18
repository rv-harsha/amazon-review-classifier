from modules import load_datasets
from utils import model_evaluation, print_roc_curve, evaluate_ssl_model

if __name__ == "__main__":

    # Please uncomment the below uncommented code, if you would like to download 
    # the datasets, extract features and then run evaluation.
    # I would highly recommend to use X.pickle and Y.pickle which already have the datasets.
    # load_datasets() will directly load the datasets from the pickle files. 

    # maybe_download(DATASET_NAME, DATASET_SIZE) # This will download the datasets
    # data = extract_and_load()                  # Extract the data and load to dataframe
    # extract_features(pre_process_data(data))   # Pre-process the data, extract TF-IDF features and save to features.pkl and labels.pkl
    # create_datasets()                          # Split the data and create the datasets and save them to X.pickle and y.pickle

    X_train, X_val, X_test, y_train, y_val, y_test = load_datasets()
    x_tr = X_train.toarray()
    x_te = X_test.toarray()

    scores = model_evaluation(x_tr, y_train, x_te, y_test)
    evaluate_ssl_model(x_tr, y_train, x_te, y_test)
    print_roc_curve(y_test, scores)

    # To view SSL plot, run python semi-supervised-learning.py