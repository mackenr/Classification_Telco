from acquire import *


def prep_telco(df):
    telco=df
    telco = telco.drop(columns=['internet_service_type_id', 'contract_type_id', 'payment_type_id','customer_id'])
    telco_new=pd.DataFrame()
    telco_new['tenure']=telco.tenure
    telco_new['monthly_charges']=telco.monthly_charges                        
    telco_new['total_charges']=telco.total_charges 

    telco_new['gender_encoded'] = telco.gender.map({'Female': 1, 'Male': 0})
    telco_new['partner_encoded'] = telco.partner.map({'Yes': 1, 'No': 0})
    telco_new['dependents_encoded'] = telco.dependents.map({'Yes': 1, 'No': 0})
    telco_new['phone_service_encoded'] = telco.phone_service.map({'Yes': 1, 'No': 0})
    telco_new['paperless_billing_encoded'] = telco.paperless_billing.map({'Yes': 1, 'No': 0})
    telco_new['churn_encoded'] = telco.churn.map({'Yes': 1, 'No': 0})
    
    dummy_df = pd.get_dummies(telco[['multiple_lines', \
                              'online_security', \
                              'online_backup', \
                              'device_protection', \
                              'tech_support', \
                              'streaming_tv', \
                              'streaming_movies', \
                              'contract_type', \
                              'internet_service_type', \
                              'payment_type'
                            ]],
                              drop_first=True)

    telco_new = pd.concat( [telco_new, dummy_df], axis=1 )                              
    telcocolums=telco_new.columns.to_list()
    snakecase_telco_columns=[i.strip().replace(' ','_') for i in telcocolums]
    telco_new.columns =snakecase_telco_columns

    return telco_new 



def get_telco_clean():
    '''
    This function reads in telco data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('telco_clean.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('telco_clean.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df=get_telco_data()
        df=prep_telco(df)
        
        # Cache data
        df.to_csv('telco_clean.csv')

    print('This is the Cleaned Dataset:\n\n\n')
    pd_DF_one_shot_info(df)      
    return df
    