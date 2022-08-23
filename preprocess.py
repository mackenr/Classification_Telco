from acquire import *
from prepare import *














def telco_split_final(df,stratify='churn_encoded'):
    '''  
    this stratifys to churn_encode as a defalt 
    
    '''
    train_telco,validate_telco,test_telco =split_data(df,tostratify=stratify)
    telco_tvt_array=[train_telco,validate_telco,test_telco]
    telco_tvt_array=[imput_telco(i) for i in telco_tvt_array]
    return telco_tvt_array[0],telco_tvt_array[1],telco_tvt_array[2]


def get_telco_tidy():
    '''
    This function reads in telco data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('telco_tidy.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('telco_tidy.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df=get_telco_clean()
        df=telco_tidy(df)
        
        
        # Cache data
        df.to_csv('telco_tidy.csv')

    print('This is the Tidy Dataset:\n\n\n')
    pd_DF_one_shot_info(df)      
    return df

def telco_tidy(df):
    telco=df
  
    telco_new=pd.DataFrame()
   
    
    dummy_df = pd.get_dummies(telco[['multiple_lines', \
                              'online_security', \
                              'online_backup', \
                              'device_protection', \
                              'tech_support', \
                              'streaming_tv', \
                              'streaming_movies', \
                              'contract_type', \
                              'internet_service_type', \
                              'payment_type',\
                              'total_in_household'
                            ]],
                              drop_first=False)

    telco=telco.drop(columns=['multiple_lines', \
                              'online_security', \
                              'online_backup', \
                              'device_protection', \
                              'tech_support', \
                              'streaming_tv', \
                              'streaming_movies', \
                              'contract_type', \
                              'internet_service_type', \
                              'payment_type',\
                              'total_in_household'
                            ])

    telco_new = pd.concat( [telco, dummy_df], axis=1 )                              
    telcocolums=telco_new.columns.to_list()
    snakecase_telco_columns=[i.strip().replace(' ','_').lower() for i in telcocolums]
    telco_new.columns =snakecase_telco_columns
 
    # telco_new_dummies=pd.get_dummies(telco_new['total_in_household'],prefix ='total_in_household')
    # telco_new.drop(columns=['total_in_household'],inplace=True)
    # telco_new = pd.concat( [telco_new, telco_new_dummies], axis=1 )    

    return telco_new 






