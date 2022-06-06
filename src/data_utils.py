"""Data Processing Utilities
"""
import collections
import copy
import numpy as np
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import yaml

def get_processed_data(modeling_data_object, task_nums=range(9), missing_resp_train=False):
    """Samples Data and Processes continuous and categorical features, and normalize responses
    
      Features are processed as follows:
      (i) Ordinal features: Missing values are imputed using the median. After imputation, all features are standardized. 
      (ii) Categorical features: Missing values are assumed to have a separate category. One-hot encoding is then applied.

      Responses are min-max normalized to have same range for equal weighting in multi-task.

    Args:
      modeling_data_object: full data object descriptor, object.
      
    Returns:
      data_processed: processed data tuple. 
    """
    #Sample a dataset which contains responses for all the nine tasks.
    if missing_resp_train==False:
        train, valid, train_valid, test = modeling_data_object.train_valid_test_split_multitask2(task_nums=task_nums, sample_frac=[0.1, 0.1, 0.1])
    else:
        train, valid, train_valid, test = modeling_data_object.train_valid_test_split_multitask_with_missing_responses(task_nums=task_nums, sample_frac=[0.1, 0.1, 0.1])

    #train, test, valid = modeling_data_object.train_valid_test_split_multitask2(task_nums=range(9), sample_frac = [1.0,0.05,0.05])
    x_train, y_train, w_train = create_data_for_lgbm(train, modeling_data_object.my_task_names, 
                                                     modeling_data_object.all_feat_col, 
                                                     modeling_data_object.wgt_col)
    x_valid, y_valid, w_valid = create_data_for_lgbm(valid, modeling_data_object.my_task_names, 
                                                     modeling_data_object.all_feat_col, 
                                                     modeling_data_object.wgt_col)
    x_train_valid, y_train_valid, w_train_valid = create_data_for_lgbm(train_valid,
                                                     modeling_data_object.my_task_names, 
                                                     modeling_data_object.all_feat_col, 
                                                     modeling_data_object.wgt_col)
    x_test, y_test, w_test = create_data_for_lgbm(test, modeling_data_object.my_task_names, 
                                                     modeling_data_object.all_feat_col, 
                                                     modeling_data_object.wgt_col)

    print("Number of samples in training set: ", x_train.shape[0])
    print("Number of samples in validation set: ", x_valid.shape[0])
    print("Number of samples in train+validation set: ", x_train_valid.shape[0])
    print("Number of samples in testing set: ", x_test.shape[0])
    print("Percentage of missing vals in training covariates: ", 100*np.count_nonzero(np.isnan(x_train.values))/(x_train.values.size))
    print("Percentage of missing vals in validation covariates: ", 100*np.count_nonzero(np.isnan(x_valid.values))/(x_valid.values.size))
    print("Percentage of missing vals in train+validation covariates: ", 100*np.count_nonzero(np.isnan(x_train_valid.values))/(x_train_valid.values.size))
    print("Percentage of missing vals in testing covariates: ", 100*np.count_nonzero(np.isnan(x_test.values))/(x_test.values.size))
    print("Number of NaNs in tasks responses in training set: ", np.isnan(y_train.values).sum(axis=0))
    print("Number of NaNs in tasks responses in validation set: ", np.isnan(y_valid.values).sum(axis=0))
    print("Number of NaNs in tasks responses in train+validation set: ", np.isnan(y_train_valid.values).sum(axis=0))
    print("Number of NaNs in tasks responses in testing set: ", np.isnan(y_test.values).sum(axis=0))
    
    ordinal_features = modeling_data_object.ord_feat_col
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_features = modeling_data_object.cat_feat_col
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=1)),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    x_preprocessor = ColumnTransformer(
        transformers=[
            ('ord', ordinal_transformer, ordinal_features),
            ('cat', categorical_transformer, categorical_features)])

    x_train_processed = x_preprocessor.fit_transform(x_train)
    x_valid_processed = x_preprocessor.transform(x_valid)
    x_train_valid_processed = x_preprocessor.transform(x_train_valid)    
    x_test_processed = x_preprocessor.transform(x_test)

    y_preprocessor = MinMaxScaler()
    y_train_processed = y_preprocessor.fit_transform(y_train)
    y_valid_processed = y_preprocessor.transform(y_valid)
    y_train_valid_processed = y_preprocessor.transform(y_train_valid)
    y_test_processed = y_preprocessor.transform(y_test)
    
    data_coll = collections.namedtuple('data', ['x_train', 'x_valid', 'x_train_valid', 'x_test',
                                                'y_train', 'y_valid', 'y_train_valid', 'y_test',
                                                'w_train', 'w_valid', 'w_train_valid', 'w_test',
                                                'x_train_processed', 'x_valid_processed',
                                                'x_train_valid_processed', 'x_test_processed',
                                                'y_train_processed', 'y_valid_processed',
                                                'y_train_valid_processed', 'y_test_processed',
                                                'x_preprocessor', 'y_preprocessor'])
    data_processed = data_coll(x_train, x_valid, x_train_valid, x_test,
                               y_train.values, y_valid.values, y_train_valid.values, y_test.values,
                               w_train.values, w_valid.values, w_train_valid.values, w_test.values,
                               x_train_processed.toarray(), x_valid_processed.toarray(),
                               x_train_valid_processed.toarray(), x_test_processed.toarray(),
                               y_train_processed, y_valid_processed, y_train_valid_processed, y_test_processed,
                               x_preprocessor, y_preprocessor)
    
    return data_processed

def load_multitask_public_data(
    data='scm1d',
    path='s3://cortex-mit1003-lmdl-workbucket/public-datasets-processed'):
    drop_columns = None
    if data in ['atp1d', 'atp7d', 'oes97', 'oes10', 'rf1', 'rf2',
                'scm1d', 'scm20d', 'edm', 'sf1', 'sf2', 'jura',
                'enb', 'slump','andro', 'osales']:
        folder = 'multitask-datasets/mtr-datasets/regression'
        df = pd.read_csv(os.path.join(path, folder, data, data+'.csv'), index_col='id')
        if data == 'atp1d':
            num_features = 411
            # target_columns = ["'ALLminpA'", "'ALLminp0'", "'aDLminpA'",
            #                   "'aCOminpA'", "'aFLminpA'", "'aUAminpA'"]
            target_columns = ['LBL+ALLminpA+fut_001', 'LBL+ALLminp0+fut_001', 'LBL+aDLminpA+fut_001',
                              'LBL+aCOminpA+fut_001', 'LBL+aFLminpA+fut_001', 'LBL+aUAminpA+fut_001']
        elif data == 'atp7d':
            num_features = 411
            # target_columns = ["'ALLminpA'", "'ALLminp0'", "'aDLminpA'",
            #                   "'aCOminpA'", "'aFLminpA'", "'aUAminpA'"]
            target_columns = ['LBL+ALLminpA+bt7d_000', 'LBL+ALLminp0+bt7d_000', 'LBL+aDLminpA+bt7d_000',
                              'LBL+aCOminpA+bt7d_000', 'LBL+aFLminpA+bt7d_000', 'LBL+aUAminpA+bt7d_000']
        elif data == 'oes97':
            num_features = 263
            target_columns = [
                '58028_Shipping__Receiving__and_Traffic_Clerks',
                '15014_Industrial_Production_Managers',
                '32511_Physician_Assistants',
                '15017_Construction_Managers',
                '98502_Machine_Feeders_and_Offbearers',
                '92965_Crushing__Grinding__Mixing__and_Blending_Machine_Operators_and_Tenders',
                '32314_Speech-Language_Pathologists_and_Audiologists',
                '13008_Purchasing_Managers',
                '21114_Accountants_and_Auditors',
                '85110_Machinery_Maintenance_Mechanics',
                '27311_Recreation_Workers',
                '98902_Hand_Packers_and_Packagers',
                '65032_Cooks__Fast_Food',
                '92998_All_Other_Machine_Operators_and_Tenders',
                '27108_Psychologists',
                '53905_Teacher_Aides_and_Educational_Assistants__Clerical'
            ]
        elif data == 'oes10':
            num_features = 298
            target_columns = [
                '513021_Butchers_and_Meat_Cutters',
                '292071_Medical_Records_and_Health_Information_Technicians',
                '392021_Nonfarm_Animal_Caretakers',
                '151131_Computer_Programmers',
                '151141_Database_Administrators',
                '291069_Physicians_and_Surgeons__All_Other',
                '119032_Education_Administrators__Elementary_and_Secondary_School',
                '432011_Switchboard_Operators__Including_Answering_Service',
                '419022_Real_Estate_Sales_Agents',
                '292037_Radiologic_Technologists_and_Technicians*',
                '519061_Inspectors__Testers__Sorters__Samplers__and_Weighers',
                '291051_Pharmacists',
                '172141_Mechanical_Engineers',
                '431011_First-Line_Supervisors_of_Office_and_Administrative_Support_Workers',
                '291127_Speech-Language_Pathologists',
                '412021_Counter_and_Rental_Clerks'
            ]
        elif data == 'rf1':
            num_features = 64
            target_columns = ['CHSI2_48H__0', 'NASI2_48H__0', 'EADM7_48H__0', 'SCLM7_48H__0',
                              'CLKM7_48H__0', 'VALI2_48H__0', 'NAPM7_48H__0', 'DLDI4_48H__0']
        elif data == 'rf2':
            num_features = 576
            target_columns = ['CHSI2_48H__0', 'NASI2_48H__0', 'EADM7_48H__0', 'SCLM7_48H__0',
                              'CLKM7_48H__0', 'VALI2_48H__0', 'NAPM7_48H__0', 'DLDI4_48H__0']
        elif data == 'scm1d':
            num_features = 280
            target_columns = ['LBL', 'MTLp2', 'MTLp3', 'MTLp4','MTLp5', 'MTLp6', 'MTLp7', 'MTLp8',
                              'MTLp9', 'MTLp10', 'MTLp11', 'MTLp12', 'MTLp13', 'MTLp14', 'MTLp15', 'MTLp16']
            drop_columns = ['MTLp5', 'MTLp6', 'MTLp7', 'MTLp8',
                              'MTLp9', 'MTLp10', 'MTLp11', 'MTLp12', 'MTLp13', 'MTLp14', 'MTLp15', 'MTLp16']
        elif data == 'scm20d':
            num_features = 61
            target_columns = ['LBL', 'MTLp2A', 'MTLp3A','MTLp4A','MTLp5A', 'MTLp6A', 'MTLp7A',
                            'MTLp8A', 'MTLp9A', 'MTLp10A', 'MTLp11A', 'MTLp12A', 'MTLp13A', 'MTLp14A',
                            'MTLp15A', 'MTLp16A']
            drop_columns = ['MTLp5A', 'MTLp6A', 'MTLp7A',
                            'MTLp8A', 'MTLp9A', 'MTLp10A', 'MTLp11A', 'MTLp12A', 'MTLp13A', 'MTLp14A',
                            'MTLp15A', 'MTLp16A']
        elif data == 'edm':
            num_features = 16
            target_columns = ['DFlow', 'DGap']
        elif data == 'sf1':
            num_features = 10
            target_columns = ['c-class', 'm-class', 'x-class']
        elif data == 'sf2':
            num_features = 10
            target_columns = ['c-class', 'm-class', 'x-class']
        elif data == 'jura':
            num_features = 15
            target_columns = ['Cd', 'Co', 'Cu']
        elif data == 'enb':
            num_features = 8
            target_columns = ['Y1', 'Y2']
        elif data == 'slump':
            num_features = 7
            target_columns = ['SLUMP_cm', 'FLOW_cm', 'Compressive_Strength_Mpa']
        elif data == 'andro':
            num_features = 30
            target_columns = ['Target', 'Target_2', 'Target_3',
                              'Target_4', 'Target_5', 'Target_6']
        elif data == 'osales':
            num_features = 401
            target_columns = ['Outcome_M1', 'Outcome_M2', 'Outcome_M3', 'Outcome_M4',
                              'Outcome_M5', 'Outcome_M6', 'Outcome_M7', 'Outcome_M8',
                              'Outcome_M9', 'Outcome_M10', 'Outcome_M11', 'Outcome_M12']
        # elif data == 'wq':
        #     num_features = 16
        #     target_columns = []
        # elif data == 'scfp':
        #     num_features = 23
        #     target_columns = []
        else:
            raise ValueError("Data {} is not supported".format(data))
    elif data=='news-multiple-platforms':
        folder = 'zero-inflated-datasets/uci-datasets'
        df = pd.read_csv(os.path.join(path, folder, data, 'News_Final.csv'), encoding='iso-8859-1')  
        df = df[['Topic', 'SentimentTitle', 'SentimentHeadline', 'Facebook', 'GooglePlus', 'LinkedIn']]
        df = df.replace(-1, np.nan)
        num_features = 3
        target_columns = ['Facebook', 'GooglePlus', 'LinkedIn']  
    elif data=='yrbs':
        folder = 'zero-inflated-datasets/youth-risk-behavior-datasets'
        df = pd.read_csv(os.path.join(path, folder, data, 'XXH2019_YRBS_Data.csv'), encoding='iso-8859-1')  
        df = df[[
            'Q1','Q2','Q3','Q4','Q6','Q7','Q8',
            'Q23','Q24','Q25','Q26','Q27','Q28',
            'Q30','Q31','Q32','Q33','Q34','Q35','Q36','Q37','Q38','Q39','Q40',
            'Q45','Q46','Q47','Q48',
            'Q67','Q68','Q69','Q70','Q71','Q72','Q73','Q74','Q75','Q76','Q77','Q78','Q79',
            'Q80','Q81','Q82','Q83',
            'Q88','Q89','Q90','Q92','Q93','Q94','Q95','Q96','Q98','Q99',
            'Q41', 'Q42', 'Q49', 'Q50', 'Q52', 'Q53', 'Q91',
        ]]
        df[['Q41', 'Q42', 'Q49', 'Q50', 'Q52', 'Q53', 'Q91']] -= 1 # assigns 0 times to 0.
        num_features = 55
        target_columns = ['Q41', 'Q42', 'Q49', 'Q50', 'Q52', 'Q53', 'Q91']        
    elif data=='fire-peril':
        folder = 'zero-inflated-datasets/kaggle-datasets'        
        df = pd.read_csv(os.path.join(path, folder, data, 'data.csv'), index_col='id', encoding='iso-8859-1')  
        num_features = 300
        df['class'] = df['target']>0.0
        df = df.replace('Z', np.nan)
        target_columns = ['target', 'class']
    elif data in ["beijing1", "beijing2", "beijing3", "beijing4", "beijing5", "beijing6", "beijing7", "beijing8", "beijing9", "beijing10", "beijing11", "beijing12"]:
        folder = 'multitask-datasets/uci-datasets/regression'
        df = pd.read_csv(os.path.join(path, folder, 'beijing', data+'.csv'), index_col=0)
        num_features = 9
        target_columns = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
        drop_columns = ['SO2', 'NO2', 'CO', 'O3']  
    elif data in ["BJPM", "SHPM", "SYPM", "CDPM", "GZPM"]:
        folder = 'multitask-datasets/uci-datasets/regression'
        df = pd.read_csv(os.path.join(path, folder, data, data+'.csv'), index_col=0)
        num_features = 9
        target_columns = ['PM2.5-Area1', 'PM2.5-Area2', 'PM2.5-Area3']
    elif data == 'sarcos':
        folder = 'multitask-datasets/uci-datasets/regression'
        df = pd.read_csv(os.path.join(path, folder, data, data+'.csv'), index_col=0)
        num_features = 21
        target_columns = ['torque-{}'.format(i) for i in range(1,8)]
        drop_columns = ['torque-1', 'torque-2', 'torque-5', 'torque-6']
    elif data == 'bike':
        folder = 'multitask-datasets/uci-datasets/regression'
        df = pd.read_csv(os.path.join(path, folder, data, data+'.csv'), index_col=0)
        num_features = 12
        target_columns = ['casual', 'registered', 'cnt']        
    elif data in ['abalone',
                'pumadyn-32nm', 'pumadyn-32nh', 'pumadyn-32fm', 'pumadyn-32fh',
                'pumadyn-8nm', 'pumadyn-8nh', 'pumadyn-8fm', 'pumadyn-8fh',
                'cpu', 'cpuSmall', 'concrete-compressive-strength',
               ]:
        folder = 'singletask-datasets/pmlb-datasets/regression'
        if data == 'abalone':
            df = pd.read_csv(os.path.join(path, folder, data, 'Dataset.data'), sep=" ", header=None)
            num_features = 8
            target_columns = [8]
        elif data in ['pumadyn-32nm', 'pumadyn-32nh', 'pumadyn-32fm', 'pumadyn-32fh']:
            df = pd.read_fwf(os.path.join(path, folder, "pumadyn-family-datasets", data, 'Dataset.data'), header=None)
            num_features = 32
            target_columns = [32]
        elif data in ['pumadyn-8nm', 'pumadyn-8nh', 'pumadyn-8fm', 'pumadyn-8fh']:
            df = pd.read_fwf(os.path.join(path, folder, "pumadyn-family-datasets", data, 'Dataset.data'), header=None)
            num_features = 8
            target_columns = [8]
        elif data in ['cpu', 'cpuSmall']:
            df = pd.read_csv(os.path.join(path, folder, "comp-activ", data, 'Prototask.data'), sep=" ", header=None)
            if data == 'cpu':
                num_features = 21
                target_columns = [21]
            elif data == 'cpuSmall':
                num_features = 12
                target_columns = [12]  
        elif data=='concrete-compressive-strength':
            df = pd.read_excel(os.path.join(path, folder, data, '{}.xls'.format(data)))
            num_features = 8
            target_columns = ['Concrete compressive strength(MPa, megapascals) ']            
    elif data=="census":
        folder = 'zero-inflated-datasets/census-datasets/census-planning-database'
        df = pd.read_csv(os.path.join(path, folder, 'pdb2021bgv3_us.csv'), encoding='iso-8859-1') 
        # Remove margin of error variables and 2010 variables
        df = df.loc[:,[col for col in df.columns if "MOE" not in col and "2010" not in col]]
        # Convert price columns to float numbers
        dollar_columns = [
            'Med_HHD_Inc_BG_ACS_15_19',
            'Med_HHD_Inc_TR_ACS_15_19',
            'Aggregate_HH_INC_ACS_15_19',
            'Med_House_Value_BG_ACS_15_19',
            'Med_House_Value_TR_ACS_15_19',
            'Aggr_House_Value_ACS_15_19',
            'avg_Agg_HH_INC_ACS_15_19',
            'avg_Agg_House_Value_ACS_15_19'
        ]
        df[dollar_columns] = df[dollar_columns].replace('[\$,]', '', regex=True).astype(np.float64)

        # Combine duplicate rows 
        df = df.drop(columns=['County_name'])
        df = df.groupby(['GIDBG', 'State', 'State_name', 'County', 'Tract', 'Block_group']).sum(min_count=1)
        feature_columns = [
            'Flag', 'LAND_AREA', 'AIAN_LAND',
            'Tot_Population_ACS_15_19',
            'pct_Males_ACS_15_19', 'pct_Females_ACS_15_19',
            'Median_Age_ACS_15_19',
            'pct_Pop_under_5_ACS_15_19', 'pct_Pop_5_17_ACS_15_19', 'pct_Pop_18_24_ACS_15_19', 'pct_Pop_25_44_ACS_15_19', 'pct_Pop_45_64_ACS_15_19', 'pct_Pop_65plus_ACS_15_19',
            'pct_Hispanic_ACS_15_19', 'pct_NH_White_alone_ACS_15_19', 'pct_NH_Blk_alone_ACS_15_19', 'pct_NH_AIAN_alone_ACS_15_19', 'pct_NH_Asian_alone_ACS_15_19', 'pct_NH_NHOPI_alone_ACS_15_19', 'pct_NH_SOR_alone_ACS_15_19',
            'pct_Pop_1yr_Over_ACS_15_19', 'pct_Pop_5yrs_Over_ACS_15_19', 'pct_Pop_25yrs_Over_ACS_15_19',
            'pct_Othr_Lang_ACS_15_19',
            'pct_Not_HS_Grad_ACS_15_19',
            'pct_College_ACS_15_19',
            'pct_Pov_Univ_ACS_15_19',
            'pct_Prs_Blw_Pov_Lev_ACS_15_19',    
            'pct_Diff_HU_1yr_Ago_ACS_15_19',
            'pct_ENG_VW_SPAN_ACS_15_19', 'pct_ENG_VW_INDOEURO_ACS_15_19', 'pct_ENG_VW_API_ACS_15_19', 'pct_ENG_VW_OTHER_ACS_15_19', 'pct_ENG_VW_ACS_15_19',
            'pct_Rel_Family_HHD_ACS_15_19',
            'pct_MrdCple_HHD_ACS_15_19', 'pct_Not_MrdCple_HHD_ACS_15_19',
            'pct_Female_No_SP_ACS_15_19',
            'pct_NonFamily_HHD_ACS_15_19',
            'pct_Sngl_Prns_HHD_ACS_15_19',
            'pct_HHD_PPL_Und_18_ACS_15_19',
            'avg_Tot_Prns_in_HHD_ACS_15_19',
            'pct_Rel_Under_6_ACS_15_19',
            'pct_HHD_Moved_in_ACS_15_19',
            'pct_PUB_ASST_INC_ACS_15_19',
            'pct_Tot_Occp_Units_ACS_15_19',
            'pct_Vacant_Units_ACS_15_19',
            'pct_Renter_Occp_HU_ACS_15_19',
            'pct_Owner_Occp_HU_ACS_15_19',
            'pct_Single_Unit_ACS_15_19',
            'pct_MLT_U2_9_STRC_ACS_15_19', 'pct_MLT_U10p_ACS_15_19',
            'pct_Mobile_Homes_ACS_15_19',
            'pct_Crowd_Occp_U_ACS_15_19',
            'pct_NO_PH_SRVC_ACS_15_19',
            'pct_No_Plumb_ACS_15_19',
            'pct_Recent_Built_HU_ACS_15_19',
            'Tot_Housing_Units_ACS_15_19',
            'Med_HHD_Inc_BG_ACS_15_19', 'Med_HHD_Inc_TR_ACS_15_19', 'avg_Agg_HH_INC_ACS_15_19',
            'Med_House_Value_BG_ACS_15_19', 'Med_House_Value_TR_ACS_15_19', 'avg_Agg_House_Value_ACS_15_19',
        ]
        num_features = len(feature_columns)
        target_columns = ['One_Health_Ins_ACS_15_19', 'Two_Plus_Health_Ins_ACS_15_19', 'No_Health_Ins_ACS_15_19']
        df = df.loc[:, [col for col in df.columns if col in feature_columns or col in target_columns]]
    else:
        raise ValueError("Data {} is not supported".format(data))
    
    df_X = df.loc[:,~df.columns.isin(target_columns)]
    df_y = df.loc[:,df.columns.isin(target_columns)]
    if drop_columns is not None:
        df_y = df_y.drop(columns=drop_columns)
    print(df_X.shape, df_y.shape)
    assert df_X.shape[1] == num_features

    if data == 'atp1d':
        categorical_features = []
    elif data == 'atp7d':
        categorical_features = []
    elif data == 'oes97':
        categorical_features = []
    elif data == 'oes10':
        categorical_features = []
    elif data == 'rf1':
        categorical_features = []
    elif data == 'rf2':
        categorical_features = []
    elif data == 'scm1d':
        categorical_features = []
    elif data == 'scm20d':
        categorical_features = []
    elif data == 'edm':
        categorical_features = []
    elif data == 'sf1':
        categorical_features = [
            'mod_zurich_class', 'largest_spot_size', 'spot_distribution',
            'activity', 'evolution', 'previous_day_activity', 'hist_complex',
            'become_hist_complex', 'area', 'area_largest'
        ]
    elif data == 'sf2':
        categorical_features = [
            'mod_zurich_class', 'largest_spot_size', 'spot_distribution',
            'activity', 'evolution', 'previous_day_activity', 'hist_complex',
            'become_hist_complex', 'area', 'area_largest'
        ]
    elif data == 'jura':
        categorical_features = [] # pre- onehot encoded features: Landuse=1, Landuse=2, Landuse=3, Landuse=4, Rock=1, Rock=2, Rock=3, Rock=4, Rock=5
    elif data == 'enb':
        categorical_features = []
    elif data == 'slump':
        categorical_features = []
    elif data == 'andro':
        categorical_features = []
    elif data == 'osales': # pre- onhot encoded 350 features
        categorical_features = []
    # elif data == 'wq':
    #     categorical_features = []
    # elif data == 'scfp':
    #     categorical_features = []
    elif data in ["beijing1", "beijing2", "beijing3", "beijing4", "beijing5", "beijing6", "beijing7", "beijing8", "beijing9", "beijing10", "beijing11", "beijing12"]:
        categorical_features = ['year', 'month']
    elif data in ["BJPM", "SHPM", "SYPM", "CDPM", "GZPM"]:
        categorical_features = ['year', 'month', 'season']
    elif data == 'sarcos':
        categorical_features = []
    elif data == 'bike':
        categorical_features = ["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]
    elif data == 'abalone':
        categorical_features = [0]
    elif data in ['pumadyn-32nm', 'pumadyn-32nh', 'pumadyn-32fm', 'pumadyn-32fh',
                  'pumadyn-8nm', 'pumadyn-8nh', 'pumadyn-8fm', 'pumadyn-8fh',
                  'cpu', 'cpuSmall',
                  'concrete-compressive-strength']:
        categorical_features = []        
    elif data=='news-multiple-platforms':
        categorical_features = ['Topic']
    elif data == 'yrbs':
        categorical_features = ['Q3','Q4','Q23','Q24','Q25','Q26','Q27','Q30','Q31','Q34','Q36','Q68','Q89','Q94']
    elif data == 'fire-peril':
        categorical_features = ['var2', 'var4', 'var5', 'var6', 'var9', 'dummy']        
    elif data == 'census':
        categorical_features = ['Flag', 'AIAN_LAND']
    else:
        raise ValueError("Data {} is not supported".format(data))

    
    ordinal_features = [col for col in df_X.columns.values if col not in categorical_features]
    metadata = {
        'categorical_features': categorical_features,
        'ordinal_features': ordinal_features
    }
    df_X = df_X.replace('?', np.nan)
    df_X[metadata['ordinal_features']] = df_X[metadata['ordinal_features']].apply(pd.to_numeric)

    # # Insert missing responses
    # np.random.seed(8) 
    # for i, col in enumerate(df_y.columns):
    #     if missing_percentage>0.0:
    #         N = df_y.shape[0]
    #         indices = np.random.choice(N, (int)(missing_percentage*N), replace=False)
    #         df_y.iloc[indices, i] = np.nan
    #         print(df_y.isna().sum(axis=0))
    return df_X, df_y, metadata

def load_processed_multitask_public_data(df_X, df_y, metadata, task, val_size=0.2, test_size=0.2, seed=8, missing_percentage=0.0, num_tests=10):
    np.random.seed(seed)        
#     df_y_binned = df_y.copy()
#     y_min = df_y.min(axis=0)
#     y_max = df_y.max(axis=0)
#     for index in y_min.index:
#         bins = np.linspace(start=y_min[index], stop=y_max[index], num=5)
#         y_binned = np.digitize(df_y.loc[:,index], bins, right=True)
#         df_y_binned[index] = y_binned
    x_train_valid, x_test, y_train_valid, y_test = train_test_split(df_X, df_y, test_size=test_size, random_state=seed)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_valid, y_train_valid, test_size=val_size, random_state=seed)
    print(x_train.nunique())
    for i, col in enumerate(y_train.columns):
        if missing_percentage>0.0:
            N = y_train.shape[0]
            indices = np.random.choice(N, (int)(missing_percentage*N), replace=False)
            y_train.iloc[indices, i] = np.nan
    for i, col in enumerate(y_valid.columns):
        if missing_percentage>0.0:
            N = y_valid.shape[0]
            indices = np.random.choice(N, (int)(missing_percentage*N), replace=False)
            y_valid.iloc[indices, i] = np.nan
    
    y_test = y_test.copy(deep=True)
    for i, col in enumerate(y_test.columns):
        if missing_percentage>0.0:
            N = y_test.shape[0]
            indices = np.random.choice(N, (int)(missing_percentage*N), replace=False)
            y_test.iloc[indices, i] = np.nan
    
    if task != 'all':
        y_train = y_train.iloc[:,task:task+1]
        y_valid = y_valid.iloc[:,task:task+1]
        y_test = y_test.iloc[:,task:task+1]
        # indices = df_y.sample(frac=0.3, random_state=args.seed).index
        # df_X = df_X.loc[indices,:]
        # df_y = df_y.loc[indices,:]
    x_train_valid = pd.concat([x_train, x_valid],axis=0)
    y_train_valid = pd.concat([y_train, y_valid],axis=0)
    
    print("Number of samples in training set: ", x_train.shape[0], y_train.shape[0])
    print("Number of samples in validation set: ", x_valid.shape[0], y_valid.shape[0])
    print("Number of samples in train+validation set: ", x_train_valid.shape[0], y_train_valid.shape[0])
    print("Number of samples in testing set: ", x_test.shape[0], y_test.shape[0])
    print("Percentage of missing vals in training covariates: ", 100*np.count_nonzero(x_train.isna().values)/(x_train.values.size))
    print("Percentage of missing vals in validation covariates: ", 100*np.count_nonzero(x_valid.isna().values)/(x_valid.values.size))
    print("Percentage of missing vals in train+validation covariates: ", 100*np.count_nonzero(x_train_valid.isna().values)/(x_train_valid.values.size))
    print("Percentage of missing vals in testing covariates: ", 100*np.count_nonzero(x_test.isna().values)/(x_test.values.size))
    print("Number of NaNs in tasks responses in training set: ", y_train.isna().values.sum(axis=0))
    print("Number of NaNs in tasks responses in validation set: ", y_valid.isna().values.sum(axis=0))
    print("Number of NaNs in tasks responses in train+validation set: ", y_train_valid.isna().values.sum(axis=0))
    print("Number of NaNs in tasks responses in testing set: ", y_test.isna().values.sum(axis=0))
    
    w_train = np.ones((y_train.shape[0],))
    w_valid = np.ones((y_valid.shape[0],))
    w_train_valid = np.ones((y_train_valid.shape[0],))
    w_test = np.ones((y_test.shape[0],))
    print(x_train.shape, x_valid.shape, x_train_valid.shape, x_test.shape)
    print(y_train.shape, y_valid.shape, y_train_valid.shape, y_test.shape)
    print(w_train.shape, w_valid.shape, w_train_valid.shape, w_test.shape)

    ordinal_features = metadata['ordinal_features']
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_features =  metadata['categorical_features']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    x_preprocessor = ColumnTransformer(
        transformers=[
            ('ord', ordinal_transformer, ordinal_features),
            ('cat', categorical_transformer, categorical_features)])
    
    x_train_processed = x_preprocessor.fit_transform(x_train)
    x_valid_processed = x_preprocessor.transform(x_valid)
    x_train_valid_processed = x_preprocessor.transform(x_train_valid)    
    x_test_processed = x_preprocessor.transform(x_test)

    y_preprocessor = MinMaxScaler()
    y_train_processed = y_preprocessor.fit_transform(y_train)
    y_valid_processed = y_preprocessor.transform(y_valid)
    y_train_valid_processed = y_preprocessor.transform(y_train_valid)
    y_test_processed = y_preprocessor.transform(y_test)
    
    print(y_train_processed.shape, y_valid_processed.shape, y_train_valid_processed.shape, y_test_processed.shape)
    data_coll = collections.namedtuple('data', ['x_train', 'x_valid', 'x_train_valid', 'x_test',
                                                'y_train', 'y_valid', 'y_train_valid', 'y_test',
                                                'w_train', 'w_valid', 'w_train_valid', 'w_test',
                                                'x_train_processed', 'x_valid_processed',
                                                'x_train_valid_processed', 'x_test_processed',
                                                'y_train_processed', 'y_valid_processed',
                                                'y_train_valid_processed', 'y_test_processed',
                                                'x_preprocessor', 'y_preprocessor'])
    data_processed = data_coll(x_train, x_valid, x_train_valid, x_test,
                               y_train, y_valid, y_train_valid, y_test,
                               w_train, w_valid, w_train_valid, w_test,
                               x_train_processed, x_valid_processed,
                               x_train_valid_processed, x_test_processed,
                               y_train_processed, y_valid_processed, y_train_valid_processed, y_test_processed,
                               x_preprocessor, y_preprocessor)
    return data_processed


def load_processed_classification_public_data(
    name="human-activity-recognition",
    val_size=0.2,
    test_size=0.2,
    seed=8,
    path="s3://cortex-mit1003-lmdl-workbucket/public-datasets-processed",
    ):
    
    if name in ["mice-protein", "isolet", "human-activity-recognition", "mnist", "fashion-mnist"]:
        path = os.path.join(path, "singletask-datasets/fetch-openml-datasets/classification")
        df_X = pd.read_csv(os.path.join(path, "{}/features.csv".format(name)))
        df_y = pd.read_csv(os.path.join(path, "{}/target.csv".format(name)))
        df_y = df_y['target']
#     if name=="mice-protein":
#         df_X, df_y = fetch_openml(name="miceprotein", return_X_y=True)
#     elif name=="isolet":    
#         df_X, df_y = fetch_openml(name="ISOLET", version=1, return_X_y=True)
#     elif name=="human-activity-recognition":    
#         df_X, df_y = fetch_openml(name="HAR", version=1, return_X_y=True)
#     elif name=="mnist":
#         df_X, df_y = fetch_openml('mnist_784', version=1, return_X_y=True)
#     elif name=="fashion-mnist":    
#         df_X, df_y = fetch_openml(name="Fashion-MNIST", version=1, return_X_y=True)
#     elif name=="coil-20":    
#         df_X, df_y = fetch_openml(name="COIL-20", version=1, return_X_y=True)
    elif name in ['breast', 'breast-cancer-wisconsin', 'car-evaluation', 'churn', 'crx',
                  'dermatology', 'diabetes', 'dna', 'ecoli', 'flare', 'heart-c', 'hypothyroid',
                  'magic', 'nursery', 'optdigits', 'pima', 'poker', 'satimage', 'sleep', 'solar-flare-2', 'spambase',
                  'texture', 'twonorm', 'vehicle', 'wine-recognition', 'yeast']:
        path = os.path.join(path, "singletask-datasets/pmlb-datasets/classification")
        df = pd.read_csv(path+"/{}/{}.tsv".format(name.replace("-", "_"), name.replace("-", "_")), sep="\t")
        df_y = df['target']
        df_X = df.drop(columns='target')
        if name in ['sleep', 'poker']:
            _, p = df_X.shape
            features = np.arange(p)
            features_to_permute = np.random.choice(features, p,  replace=False)
            num_permutes = 10
            cols = df_X.columns
            orig_f_names = []
            new_f_names = []
            for f in features_to_permute:
                f_name = cols[f]                
                for j in range(num_permutes):
                    orig_f_names.append(f_name)
                    new_f_name = '{}-Noise-{}'.format(f_name, j)
                    new_f_names.append(new_f_name)
                    df_X[new_f_name] = np.random.permutation(df_X.iloc[:, f].values)
    else:
        raise ValueError("Data: '{}' is not supported".format(name))
    classes = list(set(df_y.values))
    print("classes:", classes)
    
    print("X.min:", df_X.min().sort_values(), "X.max:", df_X.max().sort_values())
    print("name:", name, "X.shape:", df_X.shape, "y.shape:", df_y.shape)

    np.random.seed(seed)
    x_train_valid, x_test, y_train_valid, y_test = train_test_split(df_X, df_y, test_size=test_size, stratify=df_y, random_state=seed)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_valid, y_train_valid, test_size=val_size, stratify=y_train_valid, random_state=seed)
    print(x_train.nunique())
        
    print("Number of samples in training set: ", x_train.shape[0], y_train.shape[0])
    print("Number of samples in validation set: ", x_valid.shape[0], y_valid.shape[0])
    print("Number of samples in train+validation set: ", x_train_valid.shape[0], y_train_valid.shape[0])
    print("Number of samples in testing set: ", x_test.shape[0], y_test.shape[0])
    print("Percentage of missing vals in training covariates: ", 100*np.count_nonzero(x_train.isna().values)/(x_train.values.size))
    print("Percentage of missing vals in validation covariates: ", 100*np.count_nonzero(x_valid.isna().values)/(x_valid.values.size))
    print("Percentage of missing vals in train+validation covariates: ", 100*np.count_nonzero(x_train_valid.isna().values)/(x_train_valid.values.size))
    print("Percentage of missing vals in testing covariates: ", 100*np.count_nonzero(x_test.isna().values)/(x_test.values.size))
    print("Number of NaNs in tasks responses in training set: ", y_train.isna().values.sum(axis=0))
    print("Number of NaNs in tasks responses in validation set: ", y_valid.isna().values.sum(axis=0))
    print("Number of NaNs in tasks responses in train+validation set: ", y_train_valid.isna().values.sum(axis=0))
    print("Number of NaNs in tasks responses in train+validation set: ", y_test.isna().values.sum(axis=0))
    
    w_train = np.ones((y_train.shape[0],))
    w_valid = np.ones((y_valid.shape[0],))
    w_train_valid = np.ones((y_train_valid.shape[0],))
    w_test = np.ones((y_test.shape[0],))
    print(x_train.shape, x_valid.shape, x_train_valid.shape, x_test.shape)
    print(y_train.shape, y_valid.shape, y_train_valid.shape, y_test.shape)
    print(w_train.shape, w_valid.shape, w_train_valid.shape, w_test.shape)
    
    if name in ['mice-protein', 'isolet', 'human-activity-recognition', 'mnist', 'fashion-mnist']:
        metadata = {
            'continuous_features': df_X.columns,
            'categorical_features': [],
            'binary_features': [],
            'ordinal_features': [],
            'nominal_features': [],
        }
    elif name in ['breast', 'breast-cancer-wisconsin', 'car-evaluation', 'churn', 'crx',
                  'dermatology', 'diabetes', 'dna', 'ecoli', 'flare', 'heart-c', 'hypothyroid',
                  'magic', 'nursery', 'optdigits', 'pima', 'poker', 'satimage', 'sleep', 'solar-flare-2', 'spambase',
                  'texture', 'twonorm', 'vehicle', 'wine-recognition', 'yeast']:
        with open(os.path.join(path, "{}/metadata.yaml".format(name.replace("-", "_"))), "r") as stream:
            try:
                metadata = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        df_metadata = pd.DataFrame(metadata['features'])
        if  name in ['poker', 'sleep']:  
            for orig_f_name, new_f_name in zip(orig_f_names, new_f_names):
                df_metadata_sub = df_metadata[df_metadata.name==orig_f_name]
                df_metadata_sub['name'] = new_f_name
                df_metadata = pd.concat([df_metadata, df_metadata_sub], axis=0)
        from IPython.display import display
        display(df_metadata)
        metadata = {
            'continuous_features': df_metadata[df_metadata['type']=='continuous'].name.astype(str).values,
            'categorical_features': df_metadata[df_metadata['type']=='categorical'].name.astype(str).values,
            'binary_features': df_metadata[df_metadata['type']=='binary'].name.astype(str).values,
            'nominal_features': df_metadata[df_metadata['type']=='nominal'].name.astype(str).values,
            'ordinal_features': df_metadata[df_metadata['type']=='ordinal'].name.astype(str).values,
        }
        print(metadata['ordinal_features'])
    df_X[metadata['ordinal_features']] = df_X[metadata['ordinal_features']].apply(pd.to_numeric)
    df_X[metadata['continuous_features']] = df_X[metadata['continuous_features']].apply(pd.to_numeric)
    print(df_X.shape)
    
    if name in ['mice-protein', 'isolet', 'human-activity-recognition',
                'breast', 'breast-cancer-wisconsin', 'car-evaluation', 'churn', 'crx',
                'dermatology', 'diabetes', 'dna', 'ecoli', 'flare', 'heart-c', 'hypothyroid',
                'magic', 'nursery', 'optdigits', 'pima', 'poker', 'satimage', 'sleep', 'solar-flare-2', 'spambase',
                'texture', 'twonorm', 'vehicle', 'wine-recognition', 'yeast']:
        continuous_features = metadata['continuous_features']
        continuous_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_features =  metadata['categorical_features']
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        binary_features =  metadata['binary_features']
        binary_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        nominal_features =  metadata['nominal_features']
        nominal_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        ordinal_features = metadata['ordinal_features']
        ordinal_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])
        
        x_preprocessor = ColumnTransformer(
            transformers=[
                ('continuous', continuous_transformer, continuous_features),
                ('categorical', categorical_transformer, categorical_features),
                ('binary', binary_transformer, binary_features),
                ('nominal', nominal_transformer, nominal_features),
                ('ordinal', ordinal_transformer, ordinal_features),                
            ])

        print(x_train.nunique().sort_values())
        x_train_processed = x_preprocessor.fit_transform(x_train)
        x_valid_processed = x_preprocessor.transform(x_valid)
        x_train_valid_processed = x_preprocessor.transform(x_train_valid)    
        x_test_processed = x_preprocessor.transform(x_test)
    elif name in ['mnist', 'fashion-mnist', 'coil-20']:
        x_train_processed = x_train.values/255
        x_valid_processed = x_valid.values/255
        x_train_valid_processed = x_train_valid.values/255    
        x_test_processed = x_test.values/255
        

    y_preprocessor = LabelEncoder()
    y_train_processed = y_preprocessor.fit_transform(y_train)
    y_valid_processed = y_preprocessor.transform(y_valid)
    y_train_valid_processed = y_preprocessor.transform(y_train_valid)
    y_test_processed = y_preprocessor.transform(y_test)
    
    print(x_train_processed.shape, x_valid_processed.shape, x_train_valid_processed.shape, x_test_processed.shape)
    print(y_train_processed.shape, y_valid_processed.shape, y_train_valid_processed.shape, y_test_processed.shape)
    data_coll = collections.namedtuple('data', ['x_train', 'x_valid', 'x_train_valid', 'x_test',
                                                'y_train', 'y_valid', 'y_train_valid', 'y_test',
                                                'w_train', 'w_valid', 'w_train_valid', 'w_test',
                                                'x_train_processed', 'x_valid_processed',
                                                'x_train_valid_processed', 'x_test_processed',
                                                'y_train_processed', 'y_valid_processed',
                                                'y_train_valid_processed', 'y_test_processed'])
    data_processed = data_coll(x_train, x_valid, x_train_valid, x_test,
                               y_train, y_valid, y_train_valid, y_test,
                               w_train, w_valid, w_train_valid, w_test,
                               x_train_processed, x_valid_processed,
                               x_train_valid_processed, x_test_processed,
                               y_train_processed, y_valid_processed, y_train_valid_processed, y_test_processed)
    return data_processed

