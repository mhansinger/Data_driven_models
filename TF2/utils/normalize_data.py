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
    :return: data set in absolute values
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

def reTransformTarget(y_hat,moments):
    '''
    :param y_hat: Predicted target
    :param moments: mean and st read in from disk. usually called_moments_UPRIMEXY.csv
    :return: absolute values of predicted target (for comparison)
    '''

    TARGET = 'omega_DNS_filtered'

    # create empty dataframe
    data_df = pd.DataFrame(data=None,columns=[TARGET],index=range(0,len(y_hat)))

    # fill the data frame
    try:
        data_df[TARGET] = (y_hat * moments.loc[TARGET]['std']) +moments.loc[TARGET]['mean']
    except KeyError as e:
        print('Features in moments und data set do no match:',e)

    return data_df
