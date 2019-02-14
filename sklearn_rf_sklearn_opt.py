# Run rf with sci-kit learn with tuning
#
# Greg


import os
import sys
import argparse
import pickle
import time
import numpy as np
import pandas as pd
import scipy.stats as st

from datetime import datetime
#from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, ShuffleSplit
sys.path.append('/dors/capra_lab/users/sliwosgr/cancer_project/random_forest/')
from performance_optimization import *
import IO

#from train_test_rf import load_labels, load_X_y, compute_metrics, metrics_to_df, plot_roc, plot_pr
#from get_feature_importances import get_feature_importance, barplot_feat_importance, recurs_feature_elim
#from hyperparam_tune import initialize, validate_best_model, create_held_out


DATE = datetime.now().strftime('%Y-%m-%d')


# -----------
# HELPERS
# -----------

def create_held_out(X_mat, y_labels, full_df):
    '''
    Creates a train-test set for model optimization and held-out set for optimized model validation. 

    INPUTS: 
        - X_mat: observations x feature matrix
        - y_label: true labels vector 
        - full_df: dataframe observation x features matrix 

    RETURNS: 
        - X_train, y_train: data to be used for model optimization 
        - X_test, y_test: held out set 
        - full_df: full_df with an added column with two values: 'grid_cv', 'held out' 
            - 'grid_cv': train split used for model tuning 
            - 'held_out': test/held-out set used for validating optimized model

    '''
    splitter = ShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
    train_ind, test_ind = next(splitter.split(X_mat, y_labels))

    annotated_df = full_df.copy()

    X_train = X_mat.iloc[train_ind]
    X_test = X_mat.iloc[test_ind]
    y_train = y_labels.iloc[train_ind]
    y_test = y_labels.iloc[test_ind]
    annotated_df.loc[train_ind, 'partition'] = 'grid_cv'
    annotated_df.loc[test_ind, 'partition'] = 'held_out'
    print("Training label summary:")
    print(annotated_df.loc[train_ind].label.describe())
    print("Testing label summary:")
    print(annotated_df.loc[test_ind].label.describe())

    return X_train, y_train, X_test, y_test, annotated_df

# -----------
# MAIN
# -----------


if __name__ == "__main__":
    start_m = time.time()

    ###
    #   FILE PATH
    ###

    # if no args passed, run with dummy/demo, else use argparse
    FEATURE_FILE, LABEL_FILE, LABEL, PARAM_FILE, output_suffix, OUTPUT_DIR = IO.initialize()

    # outputs
    outfiles = {
    'INPUT_DATA_FILE' : os.path.join(OUTPUT_DIR, 'input_data_{}-{}-{}.tsv'.format(output_suffix, LABEL, DATE)),
    'GRID_SEARCH_TEST_FILE' : os.path.join(OUTPUT_DIR, 'grid_search_report_test_{}-{}-{}.tsv'.format(output_suffix, LABEL, DATE)),
    'GRID_SEARCH_TRAIN_FILE' : os.path.join(OUTPUT_DIR, 'grid_search_report_train_{}-{}-{}.tsv'.format(output_suffix, LABEL, DATE)),
    'BEST_MODEL_FILE' : os.path.join(OUTPUT_DIR, "best_xgb_model_{}-{}-{}.pickle".format(output_suffix, LABEL, DATE)),
    'ROC_FIG_FILE' : os.path.join(OUTPUT_DIR, "roc_auc_optimized_ptb_{}-{}-{}.png".format(output_suffix, LABEL, DATE)),
    'PR_FIG_FILE' : os.path.join(OUTPUT_DIR, "pr_auc_optimized_ptb_{}-{}-{}.png".format(output_suffix, LABEL, DATE)),
    'BEST_MODEL_PARAM_FILE' : os.path.join(OUTPUT_DIR, "best_hyperparam_{}-{}-{}.tsv".format(output_suffix, LABEL, DATE)),
    'VALIDATION_FILE' : os.path.join(OUTPUT_DIR, "held_out_metrics_{}-{}-{}.tsv".format(output_suffix, LABEL, DATE)),
    'FEATURE_IMPORT_FILE' : os.path.join(OUTPUT_DIR, 'feature_importance_{}-{}-{}.tsv'.format(output_suffix, LABEL, DATE)),
    'FEATURE_FIG_FILE' : os.path.join(OUTPUT_DIR, 'plot_feat_importance_{}-{}-{}.png'.format(output_suffix, LABEL, DATE)),
    'FEATURE_RECUR_ELIM_RANKS_FILE' : os.path.join(
        OUTPUT_DIR, 'feat_ranks_recur_elim_{}-{}-{}.tsv'.format(output_suffix, LABEL, DATE)),
    'SCATTER_PLOT_FILE' : os.path.join(OUTPUT_DIR, 'scatter_plot_{}-{}-{}.png'.format(output_suffix, LABEL, DATE)),
    'PREDS_PLOT_FILE' : os.path.join(OUTPUT_DIR, 'predictions_plot_{}-{}-{}.png'.format(output_suffix, LABEL, DATE)),
    'PERFORMANCE_FILE' : os.path.join(OUTPUT_DIR, 'performance.log'),
    'PREDICTIONS_FILE' : os.path.join(OUTPUT_DIR, 'predictions_{}-{}-{}.tsv'.format(output_suffix,LABEL,DATE))
     }

    ###
    #   TUNE HYPERPARAMETERS
    ###

    # load data
    final_labels_df = IO.load_labels(LABEL_FILE, LABEL)

    X_mat, y_labels, full_df = IO.load_X_y(FEATURE_FILE, final_labels_df)

    # split train, test
    X_train, y_train, X_test, y_test, annotated_df = create_held_out(X_mat, y_labels, full_df)

    # set up model
#    xgb_rf = XGBClassifier(random_state=32, silent=1,
#                           objective='reg:linear',
#                           booster='gbtree',
#                           importance_type='gain')

    rf = RandomForestRegressor(random_state=32, verbose=0)
    param_grid = IO.load_param_grid(PARAM_FILE)
    
    # # param grid to search
#    param_grid = {'max_depth': [10, 15, 20],
#                  'max_features': ["sqrt"],
#                  'n_estimators': [50, 75, 100]}

    # tune hyperparamter
    #hp_dict = {'num_iters':2000, 'scoring':'r2', 'cv':5}
    hp_dict = {
               'scoring':'r2',
               'cv':5
               }
    
    # tune hyperparamter
    best_rf, best_param, tune_hp_df_train, tune_hp_df_test = grid_search(
        rf, param_grid, X_train, y_train, **hp_dict)


    ###
    #   VALIDATE
    ###

    # validate on test data
    metrics_results, metrics_df, model_params, y_pred = validate_best_model(best_rf, X_test, y_test)
    model_type = 'regressor'
    model_id = "{}_{}_{}_{}".format(output_suffix, LABEL, DATE, model_type[:3])
    
    # plot curves
    IO.plot_curves(metrics_results, outfiles, y_test, y_pred, output_suffix, model_id, model_type)
    
    # write the predictions
    final_label_df = pd.DataFrame({'actual' : y_test, 'predicted' : y_pred})
    final_label_df.to_csv(outfiles['PREDICTIONS_FILE'],sep="\t",index=False,header=True)

    # log performance
    IO.record_performance(metrics_df, model_id, outfiles)
    
    ###
    #   FEATURE IMPORTANCE
    ###

    feat_df = get_feature_importance(best_rf, full_df, cols_to_drop=['sample', 'label'])
    IO.plot_feature_importance(feat_df, 25, model_id, outfiles['FEATURE_FIG_FILE'])

    ###
    #   WRITE
    ###

    # write model
    pickle.dump(best_rf, open(outfiles['BEST_MODEL_FILE'], 'wb'))

    # write dataset used with train/test annotated
    annotated_df.to_csv(outfiles['INPUT_DATA_FILE'], sep="\t", index=False)

    # write tuning_summary
    tune_hp_df_train.to_csv(outfiles['GRID_SEARCH_TRAIN_FILE'], sep="\t", index=False)
    tune_hp_df_test.to_csv(outfiles['GRID_SEARCH_TEST_FILE'], sep="\t", index=False)

    # write final model evaluation
    metrics_df.to_csv(outfiles['VALIDATION_FILE'], sep="\t", index=False)

    # write feature importance
    feat_df.to_csv(outfiles['FEATURE_IMPORT_FILE'], sep="\t", index=False)
    # ranked_feat_df.to_csv(FEATURE_RECUR_ELIM_RANKS_FILE, sep="\t", index=False)

    # write predictions
    
    # write run summary and model details
    with open(outfiles['BEST_MODEL_PARAM_FILE'], 'w') as fopen:
        fopen.write("Trained and optimized sklearn Random Forest model to predict {}.\n".format(LABEL))
        fopen.write("Run details:\n\t{}\n".format(output_suffix))
        fopen.write("Feature file used:\n\t{}\n".format(FEATURE_FILE))
        fopen.write("Label file used:\n\t{}\n".format(LABEL_FILE))
        fopen.write("Generated on {}\n\n".format(DATE))
        fopen.write("Random Forest Model Parameters:\n")

        for key, value in model_params.items():
            fopen.write("{}:{}\n".format(key, value))
        
        fopen.write("GridCV Tuning Settings:\n".format(DATE))
        for key, value in hp_dict.items(): 
            fopen.write("{}:{}\n".format(key, value))

    print("Wrote model settings to:{}".format(outfiles['BEST_MODEL_FILE']))
    end_m = time.time()
    print("Output files written to:\n{}".format(OUTPUT_DIR))
    print("DONE, took {:.2f} minutes.".format((end_m-start_m)/60))
