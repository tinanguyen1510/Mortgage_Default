import pandas as pd
import numpy as np
import os

# Raw data downloaded should be in text formal and save in the following folder
# ~\DATS 6501 - Capstone Project - Tina Nguyen\02. Data'
DIR = os.path.join( os.path.dirname(os.getcwd()))
print(DIR)
DATA_DIR = os.path.join(os.path.join(os.path.join(DIR, 'Desktop'),'DATS 6501 - Capstone Project - Tina Nguyen'), '02. Data')
print(DATA_DIR)
print(os.listdir(DATA_DIR))

# SEPARATE PERFORMANCE AND TIME DATA INTO 2 DIFFERENT LISTS
keyword = 'time'
perf = []  # list containing all performance data - performance data has the word _time_
orign = []  # list containing all origination data
for f in os.listdir(SAMPLE_DIR):
    if keyword in f:
        perf.append(f)
    else:
        orign.append(f)

print('Performance data: ',perf)
print()
print('Origination data:', orign)

# data = pd.read_csv(os.path.join(SAMPLE_DIR, 'historical_data1_time_Q12013.txt'))
# PREPROCESS PERFORMANCE DATA
performance = []
for file in perf:
    filename = os.path.join(SAMPLE_DIR, file)
    perf_df = pd.read_csv(filename, sep="|", usecols=[0,1,3], names=['id_loan', 'mthly_rpt', 'default'], skipinitialspace=True, error_bad_lines=False,
                              index_col=False, dtype='unicode') # only select specific columns to save computing time
    perf_df['default'] = perf_df['default'].replace('XX', np.NaN) # convert XX values in default column to NaN
    perf_df['default'] = perf_df['default'].replace('R', np.NaN) # convert R values in default column to NaN
    perf_df.dropna(inplace=True)
    perf_df = perf_df[(perf_df['default'] != '0') & (perf_df['default'] != '1') & (perf_df['default'] != '2')]  # only select default > 3 (more than 90 days delinquent D90)
    perf = pd.DataFrame(np.unique(perf_df['id_loan']), columns=['id_loan'])  # find unique id loan with the above criteria
    performance.append(perf)

master_perf = pd.concat(performance)  # a dataframe containing all loan ids with D90 or worse
print('Shape of performance data: ', master_perf.shape)
print(master_perf.head())

# PREPROCESS ORIGINATION DATA
origination = []
for file in orign:
    filename = os.path.join(SAMPLE_DIR, file)
    orign_df = pd.read_csv(filename, sep="|",
                              names=['cr_scr', 'frst_pmt', 'frst_homebuyer', 'mtry_date', 'MSA', 'MI_pct', 'unit','occ_sts', 'cltv', 'dti','upb',
                 'ltv','interest_rate', 'channel', 'ppm','pdt_type','ppty_state','ppty_type','pstl_code','id_loan','loan_prps',
                 'term','total_borr','slr','srvc','cnfm_flag'], skipinitialspace=True, error_bad_lines=False,
                              index_col=False, dtype='unicode') # import text files, include all data
    # replace placeholder values with NaN and drop NaN
    orign_df[['cr_scr', 'ltv', 'dti', 'interest_rate', 'cltv', 'MI_pct', 'upb']] = orign_df[['cr_scr', 'ltv', 'dti', 'interest_rate', 'cltv', 'MI_pct', 'upb']].astype('float64')
    orign_df['cr_scr'] = [np.NaN if x == 9999 else x for x in (orign_df['cr_scr'].apply(lambda x: x))]
    orign_df['frst_homebuyer'] = [np.NaN if x == '9' else x for x in (orign_df['frst_homebuyer'].apply(lambda x: x))]
    orign_df['MI_pct'] = orign_df['MI_pct'].astype('int64')
    orign_df['interest_rate'] = orign_df['interest_rate'].astype(float)
    orign_df['MI_pct'] = [np.NaN if x == 999 else x for x in (orign_df['MI_pct'].apply(lambda x: x))]
    orign_df['unit'] = [np.NaN if x == 99 else x for x in (orign_df['unit'].apply(lambda x: x))]
    orign_df['occ_sts'] = [np.NaN if x == 9 else x for x in (orign_df['occ_sts'].apply(lambda x: x))]
    orign_df['cltv'] = [np.NaN if x == 999 else x for x in (orign_df['cltv'].apply(lambda x: x))]
    orign_df['dti'] = [np.NaN if (x == 9999 or x == 999) else x for x in (orign_df['dti'].apply(lambda x: x))]
    orign_df['cnfm_flag'] = orign_df['cnfm_flag'].fillna('N')
    orign_df.drop(['ppm', 'pdt_type'], axis=1,  inplace=True)
    orign_df.dropna(inplace=True)
    origination.append(orign_df)

master_orign = pd.concat(origination) # dataframe with all origination data
print('Shape of origin data: ', master_orign.shape)
print()
print('Total loans funded: ', master_orign.shape[0])
print()

# MASTER DATAFRAME PREPROCESSING
master_df = master_perf.merge(master_orign, on='id_loan', how='outer', indicator=True)  # merge performing and origination data
# drop all remanining NaN values
master_df = master_df.dropna()

# create year and Year+Quarter columns by extracting from columns loan id
master_df['Year'] = ['20' + x for x in (master_df['id_loan'].apply(lambda x: x[2:4]))]
master_df['YrQtr'] = ['20' + x for x in (master_df['id_loan'].apply(lambda x: x[2:6]))]
# create a default column. Loans in both origination and D90 perf dataframes are delinquent. Loans only in origination are not delinquent
master_df['default'] = master_df['_merge'].map({'both': 1, 'right_only': 0})

# Merge w quarterly interest rate
int_rate = pd.read_excel(os.path.join(DATA_DIR + '\\30-fixed-rates.xlsx'), sheet_name='Qtr Avg')
int_rate['Rate'] = int_rate['Rate'].astype(float)
# print('Total rows for quarterly interest rate: ', int_rate.shape[0])
master_df = master_df.merge(int_rate, on='YrQtr', how='left')
print('Final df shape: ', master_df.shape)
# create a variable to indicate the spread of interest rate at origination (30-year fixed-rate less rate of the loan)
master_df['sato'] = master_df['Rate'] - master_df['interest_rate']
print('Number of good and bad loans: ', '\n', master_df['default'].value_counts()) # count delinquent vs good loans
print()
print('Total number of unique loans: ', len(np.unique(master_df['id_loan'])))

# master_df.to_csv('C:/Users/tina/Desktop/Capstone/Data/d90_full_data.csv', index=False) (full dataset)

master_df.to_csv(os.path.join(DATA_DIR + '\d90_full_population.csv'), index=False) # generate sampled dataset

