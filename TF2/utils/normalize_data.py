# normalize the trainings/test data set
import pandas as pd


def normalizeStandard(data_df,moments):
    '''
    :param data_df: original data set
    :param moments: mean and std read in from disk. usually called moments_UPRIMEXY.csv
    :return: normalized data set
    '''

    # create empty dataframe
    data_normalized = pd.DataFrame(data=None,columns=data_df.columns,index=data_df.index)

    # fill the data frame
    for c in data_df.columns:
        try:
            data_normalized[c] = (data_df[c] - moments.loc[c]['mean']) / moments.loc[c]['std']
        except KeyError as e:
            print('Features in moments und data set do no match:',e)
            break

    return data_normalized


def reTransformStandard(data_normalized,moments):
    '''
    :param data_df: original data set
    :param moments: mean and std read in from disk. usually called moments_UPRIMEXY.csv
    :return: normalized data set
    '''

    # create empty dataframe
    data_df = pd.DataFrame(data=None,columns=data_normalized.columns,index=data_normalized.index)

    # fill the data frame
    for c in data_normalized.columns:
        try:
            data_df[c] = (data_normalized[c] * moments.loc[c]['std']) +moments.loc[c]['mean']
        except KeyError as e:
            print('Features in moments und data set do no match:',e)
            break

    return data_df