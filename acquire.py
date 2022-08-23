import env
import os
import numpy as np
import seaborn as sns
import pandas as pd
from IPython.display import display





def pd_DF_one_shot_info(df,log_bool=True):
    oneshot=pd.concat(
        [df.head(3).rename(index={0:' head_1', 1:'head_2',2:'head_3'}),
        pd.DataFrame(df.dtypes,columns=['dtypes']).T,
        df.describe(include='all')],axis=0)
    uniquevales=df.nunique().sort_values()
    plot_u=uniquevales.plot.bar(logy=log_bool,figsize=(15,5),title='Unique values per column',rot=90,grid=True)

# Plot information with y-axis in log-scale
        
        
    return display(oneshot,df.shape,plot_u)




def get_db_url(db, env_file=env):
    '''
    returns a formatted string ready to utilize as a sql url connection

    args: db: a string literal representing a schema
    env_file: bool: checks to see if there is an env.py present in the cwd

    make sure that if you have an env file that you import it outside of the scope 
    of this function call, otherwise env.user wont mean anything ;)
    '''
    if env_file:
        username, password, host = (env.username, env.password, env.host)
        return f'mysql+pymysql://{username}:{password}@{host}/{db}'
    else:
        return 'yo you need some credentials to access a database usually and I dont want you to type them here.'







def new_telco_data():
    '''
    This function reads the telco data from the Codeup db into a df.
    '''
    sql_query = """
                select * from customers
                join contract_types using (contract_type_id)
                join internet_service_types using (internet_service_type_id)
                join payment_types using (payment_type_id)
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_db_url('telco_churn'))
    
    return df




def get_telco_data():
    '''
    This function reads in telco data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('telco.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('telco.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_telco_data()
        
        # Cache data
        df.to_csv('telco.csv')
    
    pd_DF_one_shot_info(df)    
    return df









    