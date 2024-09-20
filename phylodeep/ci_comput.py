import os

import pandas as pd
import warnings
import numpy as np

from phylodeep import BD, BDEI, TIME_DEPENDENT_COLUMNS, P, TREE_SIZE
import phylodeep_data_bd
import phylodeep_data_bdei
import phylodeep_data_bdss

PREDICTED_VALUE = 'predicted_value'

CI_97_5 = 'ci_97_5_boundary'

CI_2_5 = 'ci_2_5_boundary'

warnings.filterwarnings('ignore')

# PREDICTED = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'ci_computation/predicted_values')
# TARGET = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'ci_computation/target_values')

min_max = {'R_naught': [1, 5], 'X_transmission': [3, 10], 'Superspreading_individuals_fraction': [0.05, 0.20]}


def standardize_for_knn(predicted, target_ci):
    """
    standardize predicted and target_ci (scaler trained on target_ci)
    :param predicted: pd.DataFrame, predicted parameter values for input tree
    :param target_ci: pd.DataFrame, real/target parameter values of CI set
    :return: rescaled input pd.DataFrames (one for predicted, one for target_ci)
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    target_ci_standardized = pd.DataFrame(scaler.fit_transform(target_ci))
    predicted_standardized = pd.DataFrame(scaler.transform(predicted))

    # synchronizing indexes and column names
    target_ci_standardized.columns = target_ci.columns
    target_ci_standardized.index = target_ci.index

    predicted_standardized.columns = predicted.columns
    predicted_standardized.index = predicted.index

    return predicted_standardized, target_ci_standardized


def get_indexes_of_closest_single_factor(test_value, ci_values, n):
    """Returns indexes of n nearest neighbors for given set
    :param test_value: float, value of parameter (e.g. sampling proba or tree size) on which we select given observation
    :param ci_values: dataframe, values of these parameters in CI set
    :param n: int, number of KNNs to find
    :return: list, indexes of n KNNs
    """
    ref = ci_values.iloc[(ci_values-test_value).abs().argsort()].index
    return [ref[k] for k in range(n)]


def get_indexes_of_closest(test_s, ci_s, n):
    """Returns indexes of n nearest neighbors for given set
    :param test_s: dataframe, param set given observation
    :param ci_s: dataframe, param sets of CI set
    :param n: int, number of KNNs to find
    :return: list, indexes of n KNNs
    """
    ref = ci_s.iloc[(ci_s - test_s.values).pow(2).sum(axis=1).pow(0.5).argsort()].index
    return [ref[k] for k in range(n)]


def get_predicted_closest_single(indexes, pred_value_table, targ):
    """ returns the absolute errors for knn
    :param indexes: list, index of knn
    :param pred_value_table: dataframe, predicted parameter values of CI set
    :param targ: str, parameter name
    :return: list of predictions for each knn
    """
    # subset the real and predicted values of the closest neighbors
    closest_pred = pred_value_table.loc[indexes, :]

    # for single parameter, get the absolute difference between these
    pred_d = list(closest_pred[targ][:])
    return pred_d


def get_error_closest_single(indexes, real_value_table, pred_value_table, targ):
    """ returns the absolute errors for knn
    :param indexes: list, index of knn
    :param real_value_table: dataframe, real/target parameter values of CI set
    :param pred_value_table: dataframe, predicted parameter values of CI set
    :param targ: str, parameter name
    :return: list of absolute error in predictions for each knn
    """
    # subset the real and predicted values of the closest neighbors
    closest_pred = pred_value_table.loc[indexes, :]
    closest_real = real_value_table.loc[indexes, :]

    # for single parameter, get the absolute difference between these
    error_d = closest_pred[targ] - closest_real[targ]
    return error_d


def apply_filter(df1, df2, df3, df4, indexes):
    """
    subsets rows in input pd.Dataframes based on indexes
    :param df1: pd.DataFrame
    :param df2: pd.DataFrame
    :param df3: pd.DataFrame
    :param df4: pd.DataFrame
    :param indexes: list of indexes
    :return: modified input pd.Dataframes
    """

    df1, df2, df3, df4 = df1.loc[indexes], df2.loc[indexes], df3.loc[indexes], df4.loc[indexes]
    df1.index = df2.index = df3.index = df4.index = range(0, len(indexes))

    return df1, df2, df3, df4


def ci_comp(pred_vals, model, resc_factor, nb_tips, tr_size, samp_proba, vector_repre):
    """
    computes 95% confidence intervals for each predicted parameter, using approximated parametric bootstrap
    :param pred_vals: containing point estimates (non rescaled)
    :type pred_vals: pd.Dataframe
    :param model: 'BD', 'BDEI' or 'BDSS' corresponding to model under which the values were estimated
    :type model: str
    :param resc_factor: factor by which tree branches were rescaled prior to prediction
    :type resc_factor: float
    :param nb_tips: number of tips in the tree, for inclusion of simulations having similar tree size
    :type nb_tips: int
    :param tr_size: 'LARGE' or 'SMALL', corresponding to the size of the tree,
     ('SMALL', if 49<#tips<200; 'LARGE', if 199<#tips<501)
    :type tr_size: str
    :param samp_proba: value of sampling probability, for inclusion of simulations having similar value
    :type samp_proba: float
    :param vector_repre: 'FFNN_SUMSTATS' or 'CNN_FULL_TREE' depending on which network was used for prediction
    :type vector_repre: str
    :return:
    """
    pred_vals_original = pred_vals.copy()

    get_ci_tables = phylodeep_data_bd.get_ci_tables if BD == model else phylodeep_data_bdei.get_ci_tables \
        if BDEI == model else phylodeep_data_bdss.get_ci_tables

    predicted_vals_ci, target_vals_ci = get_ci_tables(encoding=vector_repre, tree_size=tr_size)

    # subset sampling probability and tree size parameters

    ci_samp_proba = target_vals_ci[P]
    ci_tr_size = target_vals_ci[TREE_SIZE]
    target_vals_ci.drop([P, TREE_SIZE], axis=1, inplace=True)

    # standardize data for KNN search
    pred_vals_standardized, target_vals_ci_standardized = standardize_for_knn(pred_vals, target_vals_ci)

    # first filter: keep only the closest 100k CI sets with respect to tree size
    tree_size_indexes = get_indexes_of_closest_single_factor(nb_tips, ci_tr_size, 100000)
    filt_1_predicted_ci, filt_1_param_ci_standardized, filt_1_ci_param, filt_1_ci_samp_proba = \
        apply_filter(predicted_vals_ci, target_vals_ci_standardized, target_vals_ci, ci_samp_proba, tree_size_indexes)

    # second filter: keep only the closest 10k CI sets with respect to the sampling proba
    sampling_proba_indexes = get_indexes_of_closest_single_factor(samp_proba, filt_1_ci_samp_proba, 10000)
    filt_2_predicted_ci, filt_2_param_ci_standardized, filt_2_ci_param, filt_2_ci_sampling_proba = \
        apply_filter(filt_1_predicted_ci, filt_1_param_ci_standardized, filt_1_ci_param,
                     filt_1_ci_samp_proba, sampling_proba_indexes)

    # rescaled to original values of pred_vals:
    predicted_col_names = phylodeep_data_bd.PREDICTED_NAMES if BD == model else phylodeep_data_bdei.PREDICTED_NAMES \
        if BDEI == model else phylodeep_data_bdss.PREDICTED_NAMES

    for elt in predicted_col_names:
        if elt in TIME_DEPENDENT_COLUMNS:
            pred_vals[elt] = pred_vals[elt] * resc_factor

    # compute ci intervals based on 1,000 knn from the remaining 10,000 CI sets
    ci_2_5 = []
    ci_97_5 = []

    for elt in predicted_col_names:
        # find indexes of closest parameter sets within predicted values of CI set (evaluated altogether)
        top_ind = get_indexes_of_closest_single_factor(pred_vals_standardized[elt].values,
                                                       filt_2_param_ci_standardized[elt], 1000)

        # errors on closest parameters sets, list
        error_closest = get_error_closest_single(top_ind, filt_2_ci_param, filt_2_predicted_ci, elt)

        median_error = np.median(error_closest)
        # center the values around the given prediction
        centered = [float(item - median_error + pred_vals_original[elt].values) for item in error_closest.values]

        # rescale if time period:
        if elt in TIME_DEPENDENT_COLUMNS:
            centered_resc = [item * resc_factor for item in centered]
        else:
            centered_resc = centered

        # min_max: not used on periods for real data, as they depend on the scale
        if elt not in TIME_DEPENDENT_COLUMNS:
            centered_resc = [max(min_max[elt][0], item) for item in centered_resc]
            centered_resc = [min(min_max[elt][1], item) for item in centered_resc]

        qtls = np.percentile(centered_resc, np.array(np.array([2.5, 97.5])))
        ci_2_5.append(qtls[0])
        ci_97_5.append(qtls[1])

    # add ci values to the output table
    pred_vals.index = [PREDICTED_VALUE]
    pred_vals.loc[CI_2_5, predicted_col_names] = ci_2_5
    pred_vals.loc[CI_97_5, predicted_col_names] = ci_97_5

    return pred_vals
