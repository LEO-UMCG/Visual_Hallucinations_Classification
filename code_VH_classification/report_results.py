import pandas as pd
from plotting import conf_matrix_plot


def report_average_over_folds(csv_path, config, n_folds, classes, activity_map_type, feature_vector_size, verbose):

    """
    Final logging of average performance over all folds and experiments
    """

    final_df =  pd.read_csv(csv_path)

    print(final_df)

    for activity_map in activity_map_type:
        for feature_vector_s in feature_vector_size:

            sum_acc = 0
            sum_sens = 0
            sum_spec = 0

            tp = 0
            fn = 0
            fp = 0
            tn = 0
            
            for fold in range(n_folds):
                fold+=1

                condition = ((final_df['this_fold'] == fold) & (final_df['activity_map'] == activity_map) & (final_df['feat_vec_size'] == str(feature_vector_s)))

                if len(final_df.loc[condition]) != 0:
                    # get string confusion matrix and turn into an array of ints 
                    conf_arr = final_df.loc[condition, ['confusion_matrix']]['confusion_matrix'].values[0]
                    conf_arr = conf_arr.replace('[', '').replace(']', '')
                    elements = conf_arr.split(' ')
                    conf_arr_int = [int(element) for element in elements]

                    sum_acc += float(final_df.loc[condition, ['test_acc']]['test_acc'].values[0])
                    sum_sens += float(final_df.loc[condition, ['test_sens']]['test_sens'].values[0])
                    sum_spec += float(final_df.loc[condition, ['test_spec']]['test_spec'].values[0])
                    tp += conf_arr_int[0]
                    fn += conf_arr_int[1]
                    fp += conf_arr_int[2]
                    tn += conf_arr_int[3]

            mean_acc = sum_acc/n_folds
            mean_sens = sum_sens/n_folds
            mean_spec = sum_spec/n_folds
            conf_mat = [tp, fn, fp, tn] 

            print("Mean Accuracy {}, Sensitivity {}, Specificity {} over {} folds".format(mean_acc, mean_sens, mean_spec, n_folds))

            print("Over all folds: TP {}, FN {}, FP {}, TN {}".format(tp, fn, fp, tn)) 

            df = pd.DataFrame({
                'id' : str("Mean values over all " + str(n_folds) + " folds"),
                'type': config["Type"]["model_type"], 
                'activity_map' : activity_map, 
                'feat_vec_size' : feature_vector_s, 
                'classes' : str(classes[0] + "vs" + classes[1]),
                'folds' : n_folds, 
                'this_fold' : 'all',
                'test_acc' : mean_acc, 
                'test_sens' : mean_sens, 
                'test_spec' : mean_spec, 
                'confusion_matrix' : [conf_mat],
                'list_train': "", 
                'list_tst': "", 
                'list_predictions': "", 
                'list_true_test_labels': ""})


            df.to_csv(csv_path, index=False, header=False, mode="a")
            
            if verbose == 'true':
                conf_matrix_plot(tp, fn, fp, tn, str(csv_path[:-4] + activity_map + str(feature_vector_s) +'.pdf'))

