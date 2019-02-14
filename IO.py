import os
import argparse
import math
import sys
import time
import pandas as pd
sys.path.append('/dors/capra_lab/users/sliwosgr/cancer_project/random_forest/')
import estimator_plots


ROOTDIR = "/dors/capra_lab/users/sliwosgr/cancer_project/random_forest"

def initialize():
    
    if len(sys.argv) == 1:
        # run with default args:
        print("Running with test matrix and Activity areas")

        feature_data_dir = os.path.join(ROOTDIR,"features/")
        feature_file_path = os.path.join(feature_data_dir, "test_matrix1.tab")
        label = "Activity_area"
        label_file_path = os.path.join(ROOTDIR,"labels/erlotinib_labels.tab")

        output_suffix = 'test_matrix1'
        db_name ='demo_db'
        param_file_path = os.path.join(ROOTDIR,"test_params.ls")

    else:
        # run with arguments passed in
        parser = argparse.ArgumentParser(description='train, tune hyper-paramters, and validate best RF model')

        # REQUIRED ARGUMENTS IN ORDER
        parser.add_argument('feature_file', action='store', type=str)
        parser.add_argument('label_file', action='store', type=str)
        parser.add_argument('label', action='store', type=str)
        parser.add_argument('param', action='store', type=str)
        parser.add_argument('output_suffix', action='store', type=str)
        
        results = parser.parse_args()

        # retrieve passed arguments
        feature_file_path = results.feature_file
        label_file_path = results.label_file
        label = results.label
        output_suffix = results.output_suffix
        param_file_path = results.param

    output_dir = os.path.join(ROOTDIR,"output/")        
    if not os.path.isdir(output_dir):
        try:
            os.mkdir(output_dir)
        except:
            sys.exit("Failed to create output directory")
    return feature_file_path, label_file_path, label, param_file_path, output_suffix, output_dir

def load_labels(labels_file, label_column):
    """
    Load the labels file and filter for the desired label
    
    Params
    ------
    labels_file : str
    label_column : str
        full path to tsv file with at least following columns: 
        'sample':       cell line sample ID, for joining with features
        label_column:   desired label (eg: Activity_area)

    Returns
    -------
    final_labels_df: pandas.DataFrame
        Dataframe with sample, label_column
    """

    # load labels
    print("Loading labels...")
    labels_df = pd.read_csv(labels_file, sep="\t")
#    labels_df.sort_values(['sample'], inplace=True, ascending=True)
    print("labels summary:")
    print(labels_df.columns)
    print(labels_df[label_column].describe())

    labels_df = labels_df[['sample',label_column]]
    labels_df.rename(columns={label_column:'label'},inplace=True)
    
    return labels_df

def load_X_y(feature_file, label_df):
    """
        Load 'feature_file', inner join files in 'add_features_files', then output 'X_mat', 'y_labels' and full 'merged_df'.

        Params
        ------
        feature_file : str
            full path to tsv file with sample x FEATURES (first column should be sample) with header
        label_df : pandas.DataFrame
            one row per 'sample' and its desired label

        Returns
        -------
        X_mat : numpy.array
            sample by FEATURE array with values from feature file
        y_labels : numpy.array (sample x 1 )
            labels of interest
        merged_df : pandas.DataFrame
            full dataframe used to create X_mat and y_labels with headers
    """
    stime = time.time()
    print("Loading X and y matrices...")

    # load feature matrix
    feat_df = pd.read_csv(feature_file, sep="\t")

    # merge label to feat_df
    merged_df = pd.merge(feat_df, label_df, how='inner', on='sample')
    X_mat = merged_df.drop(['sample','label'],axis=1, inplace=False)
    y_labels = merged_df.label
    print("\tDone creating X_mat {} and y_labels. Took {:.2f} seconds".format(X_mat.shape, time.time()-stime))

    return X_mat, y_labels, merged_df

def load_param_grid(param_grid_file):
    """
        Load a parameter grid definition file:
            Each line is a parameter argument to use in grid and colon and values (eg: max_depth: 10,20)
            Values can be csv or a range defined by (min - max, step_size) [will always include min and max though]
            Example of range = max_depth: 1 - 10, 2 (will create 1,3,5,7,9,10)
            Any repeated keys will be skipped
            
        Params
        ------
        full path to parameter grid file
        
        Returns
        -------
        param_grid : Dict
            keys are param labels and values are lists            
    """
    param_grid = dict()
    with open(param_grid_file) as infile:
        for line in infile:
            line = line.strip().split("#")[0]
            line = line.split(":")
            if len(line)<2: continue
            label = line[0]
            if label in param_grid: continue
            vals = line[1]
            if "-" in vals:
                try:                
                    vals = vals.split("-")
                    try:
                        a = int(vals[0])
                    except ValueError:
                        a = float(vals[0])
                    try:
                        b = int(vals[1].split(",")[0])
                    except:
                        b = float(vals[1].split(",")[0])                        
                    try:                        
                        step_size = vals[1].split(",")[1]
                        try:
                            step_size = int(step_size)
                        except:
                            step_size = float(step_size)
                    except IndexError:
                        step_size = 1
                    vals = [a]
                    while (vals[-1] + step_size) < b:
                        vals.append(vals[-1]+step_size)
                    vals.append(b)                                                                          
                except ValueError:
                    pass          
            else:
                tmpv = list()
                for x in vals.split(","):
                    try:
                        tmpv.append(int(x))
                    except ValueError:
                        try:
                            tmpv.append(float(x))
                        except ValueError:
                            tmpv.append(x.strip())
                vals = tmpv
            param_grid[label] = vals
    print("Read parameter grid:")
    for x,y in param_grid.items():
        print(f"{x}: {y}")           
    
    return param_grid

def plot_curves(metric_results, outfiles, y_test, y_pred, output_suffix, model_id, model_type = 'regressor'):
    if model_type == 'regressor':
        estimator_plots.plot_scatter(y_test, y_pred, model_id, outfiles['SCATTER_PLOT_FILE'])
        estimator_plots.plot_predictions(y_test, y_pred, model_id, outfiles['PREDS_PLOT_FILE'])
    else:
        return

def plot_feature_importance(df, nfeat, modelid, outfile):
    estimator_plots.barplot_feat_importance(df, nfeat, modelid, outfile)
    
def record_performance(metric_df, model_id, outfiles):
    cols = metric_df.columns
    metric_df['model_id'] = model_id
    cols = ['model_id'] + list(cols)
    if os.path.isfile(outfiles['PERFORMANCE_FILE']):
        metric_df.to_csv(outfiles['PERFORMANCE_FILE'],sep="\t",columns=cols,index=False,header=False,mode='a')
    else:
        metric_df.to_csv(outfiles['PERFORMANCE_FILE'],sep="\t",columns=cols,index=False)
        
    # plot curves
#    pos_prop = np.sum(y_test == 1)/len(y_test)
#    plot_roc([metrics_results['fpr']], [metrics_results['tpr']], [metrics_results['roc_auc']],
#             output_suffix, roc_fig_file=ROC_FIG_FILE)
#    plot_pr([metrics_results['pr_curve']], [metrics_results['rc_curve']], [metrics_results['avg_prec']],
#            output_suffix, pr_fig_file=PR_FIG_FILE, pos_prop=pos_prop)
