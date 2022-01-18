from modules import *
from utils import *

# maybe_download(DATASET_NAME, DATASET_SIZE)
# data = extract_and_load()
# exploratory_data_analysis()
# extract_features(pre_process_data(data))
# create_datasets()

X_train, X_val, X_test, y_train, y_val, y_test = load_datasets()

#densify sparse matricies
x_tr = X_train.toarray()
x_va = X_val.toarray()


hypothesis_complexity_study(x_tr, x_va, y_val, y_train)
# grid_output = grid_search(x_tr, y_train, x_va, y_val)
# print_roc_curve(y_val, grid_output)
# model_evaluation(x_tr, y_train, x_te, y_test)