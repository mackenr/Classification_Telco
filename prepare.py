from acquire import *
import scipy.stats as stats
from sympy import symbols

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from functools import reduce
from itertools import combinations , product

def prep_telco(df):
    telco=df
    telco = telco.drop(columns=['internet_service_type_id', 'contract_type_id', 'payment_type_id','customer_id'])
    telco_new=pd.DataFrame()
 

    telco_new['gender_encoded'] = telco.gender.map({'Female': 1, 'Male': 0})
    telco_new['partner_encoded'] = telco.partner.map({'Yes': 1, 'No': 0})
    telco_new['dependents_encoded'] = telco.dependents.map({'Yes': 1, 'No': 0})
    telco_new['phone_service_encoded'] = telco.phone_service.map({'Yes': 1, 'No': 0})
    telco_new['paperless_billing_encoded'] = telco.paperless_billing.map({'Yes': 1, 'No': 0})
    telco_new['churn_encoded'] = telco.churn.map({'Yes': 1, 'No': 0})
    telco_new['total_in_household']=telco_new.dependents_encoded+telco_new.partner_encoded+1 
    telco = telco.drop(columns=['gender', 'partner', 'dependents','phone_service','paperless_billing','churn'])
    telco_new = pd.concat( [telco_new, telco], axis=1 )                 
    
        

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

def telco_split(df,stratify='churn_encoded'):
    '''  
    this stratifys to churn_encode as a defalt 
    
    '''
    train_telco,validate_telco,test_telco =split_data(df,tostratify=stratify)
    telco_tvt_array=[train_telco,validate_telco,test_telco]
    telco_tvt_array=[imput_telco(i) for i in telco_tvt_array]
    return telco_tvt_array[0],telco_tvt_array[1],telco_tvt_array[2]












def imput_telco(df):
    colmean=df.total_charges[df.total_charges.str.strip()!=''].astype(float).mean()
    df.total_charges=df.total_charges[df.total_charges.str.strip()!=''].astype(float)
    colmean=df.total_charges.mean()
    df.total_charges.fillna(colmean,inplace=True)
    
    return df







def one_tail_hypthonesis(p,alpha):
    '''
    add logic to input our null hypothesis
    
    ''' 
    p=p
    alpha=alpha
    if p < alpha:
        x=('We reject the null hypothesis')
    else:
        x=("We fail to reject the null hypothesis")
    return x

def two_tail_hyp_test(p,t,alpha):
    '''
    add logic to input our null hypothesis  
    
    ''' 
    if (t > 0) and ((p/2) < alpha):
        x=('We reject our null hypothesis.' )
    else:
        x=('We fail to reject our null hypothesis')
    return x



def num_to_num(df,num_combos,alpha=.05):
    
    
    display(symbols('Correlation'))
    dflist1=[]
    dflist2=[]
    dflist3=[]
    dflist4=[]
    dflist5=[]
    dflist6=[]
    dflist7=[]
    count=0
   

  
   
    for i in num_combos:
        r,p_person=stats.pearsonr(df[i[0]],df[i[1]])
        pearson_hyp=two_tail_hyp_test(p_person,r,alpha)
        rho,p_spear=stats.spearmanr(df[i[0]],df[i[1]])
        spearman_hyp=two_tail_hyp_test(p_spear,rho,alpha)
        pearName='Pearson'
        r=f'{r:.4g}'
        p_person=f'{p_person:.4g}'
        spearName='Spearman'
        rho=f'{rho:.4g}'
        p_spear=f'{p_spear:.4g}'
        dflist1.append(r)
        dflist2.append(p_person)
       
        dflist3.append(rho)
        dflist4.append(p_spear)
        dflist5.append(pearson_hyp)
        dflist6.append(spearman_hyp)

        dflist7.append({count:i})
        count+=1

    dflist=pd.DataFrame([dflist1,dflist2,dflist5,dflist3,dflist4,dflist6])
    
    dflist.rename(index={0:f'{pearName} r',1:f'{pearName} p stat',3:f'{spearName} rho',4:f'{spearName} p stat',2:'pearson p test',5:'spearman ptest'},inplace=True)
    # dflist5=[dict(dflist5)]
    [dflist.rename(columns=dflist7[i],inplace=True) for i in range(0,len(dflist7))]
    

    display(dflist)


def cat_to_cat(df,cat_combos, alpha=(100-95)/100,rejected_chi=True):
    '''
    Intended to test all bivariate categorical to categorical relationships in a given df. The output becomes messy so there is a boolean option to select 
    '''
  
  
    reject=[]
    accept=[]

    
    for i in range(0,len(cat_combos)):
        var1=cat_combos[i][0]
        var2=cat_combos[i][1]

      
        

        x=pd.crosstab(df[var1],df[var2])
        
        
        chi2float,pfloat,dofint,expectedndarray=stats.chi2_contingency(x)
        chi_hyp=one_tail_hypthonesis(pfloat,alpha)
        chilist=[chi2float,pfloat,dofint]
        chilist=[f'{i:.4g}' for i in chilist]
        # expectedndarray=[float(f'{i:.4g}') for i in expectedndarray.flatten().tolist()]
        # expectedndarray=pd.DataFrame(np.array(expectedndarray).reshape(2,2))
        # expectedndarray.rename(columns={0: "Expected ", 1: 'Cross'},inplace=True)
        # chilist=pd.DataFrame(chilist)
        # chilist.rename(index={0: "Chi Sq. Stat", 1: "P(chi^2)",2:"D.F."},inplace=True)
        # x.rename(columns={0: "Actual ", 1: 'Cross'},inplace=True)
        # x=pd.concat([x,expectedndarray],axis=1)
        # x.rename(index={0: " ", 1: ''},inplace=True)
        
        out=(f'{var1}/{var2}')
        if chi_hyp=='We reject the null hypothesis':
            reject.append([chilist[0],chilist[1],out])
        else:
            accept.append([chilist[0],chilist[1],out])
        ##need to partition this if output is larger than 15
    if rejected_chi==True:
        reject=pd.DataFrame(reject)
        reject.rename(columns={0: "Chi Square Stat",1:"Chi Square p",2:'Vars'},inplace=True)
        display(symbols("\chi^2"),symbols('We~reject~the~null~hypothesis'),symbols('Thus~dependendace~is~likely'))
        display(reject)
        return reject


            
    elif rejected_chi==False:
        accept=pd.DataFrame(accept)
        accept.rename(columns={0: "Chi Square Stat",1:"Chi Square p",2:'Vars'},inplace=True)
        display(symbols("\chi^2"),symbols('We~accept~the~null~hypothesis'),symbols('Thus~independence~is~likely'))
        display(accept)
        return accept
    else:
        print("check your logic in cat_to_cat")









    


def cat_to_num(df,cat_numcombos,alpha=0.05,rejected_null=True):
    
  
  
    reject=[]
    accept=[]
    
    for i in range(0,len(cat_numcombos)):
        res=stats.mannwhitneyu(df[cat_numcombos[i][1]],df[cat_numcombos[i][0]])
        hyp=one_tail_hypthonesis(res[1],alpha)
        first=(cat_numcombos[i][0])   
        second=(cat_numcombos[i][1])
        out=(first+'/'+second)
        if hyp=='We reject the null hypothesis':
            reject.append([f'{res[0]:.3g}',f'{res[1]:.3g}',out])
        else:
            accept.append([f'{res[0]:.3g}',f'{res[1]:.3g}',out])


    if rejected_null==True:
        reject=pd.DataFrame(reject)
        reject.rename(columns={0:  "Man Whit Stat",1:"Alt p p",2:'Vars'},inplace=True)
        display(symbols('Mann~Whitney'),symbols('We~reject~the~null~hypothesis'),reject)
        return reject


            
    elif rejected_null==False:
        accept=pd.DataFrame(accept)
        accept.rename(columns={0: "Man Whit Stat",1:"Alt p",2:'Vars'},inplace=True)
        display(symbols('Mann~Whitney'),symbols('We~accept~the~null~hypothesis'),accept)
        return accept
        
 

def scatter_churn(df, x,y):
    for col, subset in df.groupby(['churn_encoded']):
        plt.scatter(subset[x], subset[y], label=col,marker='.')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.legend()


