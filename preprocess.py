from acquire import *
from prepare import *


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix   



df=get_telco_clean()



def split_data(df,tostratify=None, test_size=.2,validate_size=.25):
    '''
    Takes in a dataframe and return train, validate, test subset dataframes
    '''
    if tostratify!=None:
        train, test = train_test_split(df, test_size=test_size, 
                               random_state=123, stratify=df[tostratify])
        train, validate = train_test_split(train, test_size=validate_size, 
                 random_state=123, stratify=train[tostratify])
    else:
        train, test = train_test_split(df, test_size=test_size, 
                               random_state=123)
        train, validate = train_test_split(train, test_size=validate_size, 
                 random_state=123)
    # df=pd.DataFrame([{['Prepared Data',,,,,,,]:df.shape},{'Train':train.shape},{'Validate':validate.shape},{'Test':test.shape}])
    df=pd.DataFrame([df.shape,train.shape,validate.shape,test.shape],index=['Prepared Data','Train','Validate','Test'],columns=['Length','Width'])
    display(df)
    return train, validate, test




def imput_telco(df):
    colmean=df.total_charges[df.total_charges.str.strip()!=''].astype(float).mean()
    df.total_charges=df.total_charges[df.total_charges.str.strip()!=''].astype(float)
    return df

def final_telco_split(stratify='churn_encoded'):
    '''  
    this stratifys to churn_encode as a defalt 
    
    '''
    train_telco,validate_telco,test_telco =split_data(df,tostratify=stratify)
    telco_tvt_array=[train_telco,validate_telco,test_telco]
    telco_tvt_array=[imput_telco(i) for i in telco_tvt_array]
    return telco_tvt_array[0],telco_tvt_array[1],telco_tvt_array[2]

