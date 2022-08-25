from acquire import *
from prepare import *
from preprocess import *

import time

from sympy import Matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix,ConfusionMatrixDisplay,precision_score   
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import GridSearchCV
from math import sqrt




def X_y_all(train,validate,test):


    X_train = train.drop(columns=["churn_encoded"])
    y_train = train.churn_encoded

    X_validate = validate.drop(columns=["churn_encoded"])
    y_validate = validate.churn_encoded

    X_test = test.drop(columns=["churn_encoded"])
    y_test = test.churn_encoded

    return X_train,y_train,X_validate,y_validate,X_test,y_test,train,validate,test







def decision_tree(X_train,y_train,X_validate,y_validate,X_test,y_test,random):
    
    # make the thing
    tree_model = DecisionTreeClassifier()
    # tree_params=tree_model.get_params()
    
    
    
    ## Hyperparameters
    max_depth=range(4,10)
    max_features=["sqrt", "log2"]
    class_weight=['balanced']
    ccp_alpha=np.arange(0,0.05,0.1)
    tree_params={'ccp_alpha':ccp_alpha,'class_weight':class_weight,'max_depth':max_depth,'max_features':max_features}
    
    # fit the thing
    tree_model.fit(X_train, y_train)
    # y_pred_tree = tree_model.predict(X_train)
    # precision = precision_score(y_train, y_pred_tree)
    grid_search_tree = GridSearchCV(estimator=tree_model, param_grid=tree_params, cv= 20, scoring='recall',n_jobs=-2)
    grid_search_tree.fit(X_train, y_train)
    gs_tree_params = grid_search_tree.best_params_
    gs_tree_params.update({'random_state':random})

#     gs_tree_params={'ccp_alpha': 0.0,
#  'class_weight': 'balanced',
#  'max_depth': 8,
#  'max_features': 'log2',
#  'random_state': 4563}
    tree_opt= DecisionTreeClassifier(**gs_tree_params)
    display(gs_tree_params)
    # Fit the model (on train and only train)
    tree_opt_dict={}
    tree_opt.fit(X_train, y_train)
    tree_opt_dict[f'Parameters:{gs_tree_params}]'] = {
    'train_score': round(tree_opt.score(X_train, y_train), 2),
    'validate_score': round(tree_opt.score(X_validate, y_validate), 2),
    'diff': round(abs(tree_opt.score(X_train, y_train)-tree_opt.score(X_validate, y_validate)),2)}
    display(symbols('Decision~Tree~Classifier'),pd.DataFrame(tree_opt_dict))



    plt.figure(figsize=(20,20))
    plot_tree(tree_opt, feature_names=X_train.columns.to_list(), class_names=['Churn', 'No Churn'],
    filled=True,   rounded=True );


def random_forest(X_train,y_train,X_validate,y_validate,X_test,y_test,random):


    

    # Make the model
    forest1 = RandomForestClassifier(max_depth=10, random_state=123)
    forest_params=forest1.get_params()
    forest_params={'n_estimators': [10,100],'max_features':['sqrt'],'max_depth':[5,20],'criterion':['gini']}
    
    
    # Fit the model (on train and only train)
    forest1.fit(X_train, y_train)
    
    # Use the model
    # We'll evaluate the model's performance on train, first
    y_pred_forest = forest1.predict(X_train)
    
    
    # precision = precision_score(y_train, y_pred_forest)
    # precision
    grid_search_rf = GridSearchCV(estimator=forest1, param_grid=forest_params, cv= 20, scoring='recall',n_jobs=-2)
    grid_search_rf.fit(X_train, y_train)
    gs_rf_params = grid_search_rf.best_params_
    gs_rf_params.update({'random_state':random})
    # gs_rf_params={'criterion': 'gini',
    # 'max_depth': 5,
    # 'max_features': 'sqrt',
    # 'n_estimators': 100,
    # 'random_state': 4563}
    forest_opt = RandomForestClassifier(**gs_rf_params)
    # Fit the model (on train and only train)
    forest_opt.fit(X_train, y_train)
    
    # Use the model
    # We'll evaluate the model's performance on train, first
    y_pred_forest_opt = forest_opt.predict(X_train)
    
    opt_rf= RandomForestClassifier(**gs_rf_params)
    # Fit the model (on train and only train)
    opt_rf_dict={}
    opt_rf.fit(X_train, y_train)
    opt_rf_dict[f'Parameters:{gs_rf_params}]'] = {
    'train_score': round(opt_rf.score(X_train, y_train), 2),
    'validate_score': round(opt_rf.score(X_validate, y_validate), 2),
    'diff': round(abs(opt_rf.score(X_train, y_train)-opt_rf.score(X_validate, y_validate)),2)}
    display(symbols("Random~Forest"),pd.DataFrame(opt_rf_dict))
    # Produce the classification report on the actual y values and this model's predicted y values
    forest_report = classification_report(y_train, y_pred_forest_opt, output_dict=True)
    display(pd.DataFrame(forest_report))
    # sklearn confusion matrix
    rf_cm = confusion_matrix(y_train, y_pred_forest_opt)
    display(Matrix(rf_cm))
    disp = ConfusionMatrixDisplay(confusion_matrix=rf_cm, display_labels=opt_rf.classes_)
    disp.plot()
    plt.show()    






def logistic_reg( X_train,y_train,X_validate,y_validate,X_test,y_test,random):
  
    logit = LogisticRegression()
    logit_params=logit.get_params()
    ## Hyperparameters
    penalty=['l2']
    C_list=np.geomspace(1e-2,1e3,6)
    class_weight_list=[{0:1, 1:99},'balanced']
    solver_list=['newton-cg','lbfgs']
    max_iter_list=range(500,1000,100)
    # intercept_scaling=np.arange(1,2,.25)
    logit_params={'C':C_list, 'class_weight': class_weight_list, 'max_iter': max_iter_list, 'penalty': penalty, 'solver': solver_list}
    #  fit the model on train data
    logit.fit(X_train, y_train)
    y_pred_logit = logit.predict(X_train)
    precision = precision_score(y_train, y_pred_logit)
    grid_search_log = GridSearchCV(estimator=logit, param_grid=logit_params, cv= 20, scoring='recall',n_jobs=-2)
    grid_search_log.fit(X_train, y_train)
    gs_log_params = grid_search_log.best_params_
    gs_log_params.update({'random_state':random})
    display(gs_log_params)
   



    log_opt= LogisticRegression(**gs_log_params)
    # Fit the model (on train and only train)
    log_opt_dict={}
    log_opt.fit(X_train, y_train)
    log_opt_dict[f'Parameters:{gs_log_params}]'] = {
    'train_score': round(log_opt.score(X_train, y_train), 2),
    'validate_score': round(log_opt.score(X_validate, y_validate), 2),
    'diff': round(abs(log_opt.score(X_train, y_train)-log_opt.score(X_validate, y_validate)),2)}
    display(symbols('Logistic~Regression'),pd.DataFrame(log_opt_dict))
    y_pred_log_opt = log_opt.predict(X_train)

    # log_coeffs = pd.DataFrame(log_opt.coef_[0], index=X_train.columns,
    #                          columns=['Coeff'])
    # log_coeffs#odds log scaled
    # odds = np.exp(log_coeffs)
    # display(odds.T) #odds
    logit_cm = confusion_matrix(y_train, y_pred_log_opt)
    #display cm
    disp = ConfusionMatrixDisplay(logit_cm, display_labels=logit.classes_)
    disp.plot()
    plt.show()
    logit_report=classification_report(y_train, y_pred_log_opt,output_dict=True)
    display(pd.DataFrame(logit_report))







def knn(X_train,y_train,X_validate,y_validate,X_test,y_test):
    



    knn = KNeighborsClassifier()
    knn_params=knn.get_params()
    leaf_size=range(20,45,5)
    n_neighbors=[5,6,7]
    weights=['uniform','distance']
    knn_params={'n_neighbors':n_neighbors,'weights':weights,'p':[1,2],'leaf_size':leaf_size}

    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_train)


    

    grid_search_knn = GridSearchCV(estimator=knn, param_grid=knn_params, cv= 20, scoring='recall',n_jobs=-2)
    grid_search_knn.fit(X_train, y_train)
    gs_knn_params = grid_search_knn.best_params_
    display(gs_knn_params)
   

    knn_opt = KNeighborsClassifier(**gs_knn_params)
    # Fit the model (on train and only train)
    knn_opt.fit(X_train, y_train)

    # Use the model
    # We'll evaluate the model's performance on train, first
    y_pred_knn_opt = knn_opt.predict(X_train)





    knn_opt_dict={}

    knn_opt_dict[f'Parameters:{gs_knn_params}]'] = {
    'train_score': round(knn_opt.score(X_train, y_train), 2),
    'validate_score': round(knn_opt.score(X_validate, y_validate), 2),
    'diff': round(abs(knn_opt.score(X_train, y_train)-knn_opt.score(X_validate, y_validate)),2)}
    display(symbols('KNN'),pd.DataFrame(knn_opt_dict))



    knn_cm = confusion_matrix(y_train, y_pred_knn_opt)
    #display cm
    disp = ConfusionMatrixDisplay(knn_cm, display_labels=knn_opt.classes_)
    disp.plot()
    plt.show()
    knn_report=classification_report(y_train, y_pred_knn,output_dict=True)
    pd.DataFrame(knn_report)









  











def confusion_matrix_analyis(tn,fp,fn,tp):
   
    
    

    
    npv= tn/(tn+fn)
    ppv=tp/(tp+fp)
    acc=(tp+tn)/(tp+fp+fn+tn)
    recall=tp/(tp+fn)
    mcc= ((tp*tn)-(fp*fn))/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
      ##recall=TPR
    f1=2*((ppv*recall)/(ppv+ppv))


    # tn=float(f'{tn:.4g}')
    # fp=float(f'{fp:.4g}')
    # fn=float(f'{fn:.4g}')
    # tp=float(f'{tp:.4g}')
    # npv=float(f'{npv:.4g}')
    # ppv=float(f'{ppv:.4g}')
    # acc=float(f'{acc:.4g}')
    # recall=float(f'{recall:.4g}')
    # mcc=float(f'{mcc:.4g}')
    # f1=float(f'{f1:.4g}')
    
    dfindex=[
    'tn',
    'fp',
    'fn',
    'tp',
    'npv',
    'ppv',
    'acc',
    'recall',
    'mcc',
    'f1']

    dfvals=[
    tn,
    fp,
    fn,
    tp, 
    npv,
    ppv,
    acc,
    recall,
    mcc,
    f1]


    df= dict(zip(dfindex, dfvals))
   
    return df
    



def best_model_recall(X_train,y_train,X_validate,y_validate,X_test,y_test,random):


    

    # Make the model
    forest1 = RandomForestClassifier()
   
    forest_params={'n_estimators': [10,100],'max_features':['sqrt'],'max_depth':[5,20],'criterion':['gini']}
    forest1.fit(X_train, y_train)
    
    # # Fit the model (on train and only train)
    # forest1.fit(X_train, y_train)
    
    # Use the model
    # We'll evaluate the model's performance on train, first
  
    grid_search_rf = GridSearchCV(estimator=forest1, param_grid=forest_params, cv= 20, scoring='recall',n_jobs=-2)
    grid_search_rf.fit(X_train, y_train)
    gs_rf_params = grid_search_rf.best_params_
    gs_rf_params.update({'random_state':random})
   
    forest_opt = RandomForestClassifier(**gs_rf_params)
    # Fit the model (on train and only train)
    forest_opt.fit(X_train, y_train)
    y_pred_forest_opt_train= forest_opt.predict(X_train)
    y_pred_forest_opt_val= forest_opt.predict(X_validate)
    
   
    y_pred_forest_opt_test= forest_opt.predict(X_test)
    display(gs_rf_params)
    opt_rf= RandomForestClassifier(**gs_rf_params)
    # Fit the model (on train and only train)
    opt_rf_dict={}
    opt_rf.fit(X_train, y_train)
    tntr, fptr, fntr, tptr = confusion_matrix(y_train, y_pred_forest_opt_train).ravel()
    cmtrain=confusion_matrix_analyis(tntr, fptr, fntr, tptr)
    tnv, fpv, fnv, tpv = confusion_matrix(y_validate, y_pred_forest_opt_val).ravel()
    cmval=confusion_matrix_analyis( tnv, fpv, fnv, tpv)
    tntst, fptst, fntst, tptst = confusion_matrix(y_test, y_pred_forest_opt_test).ravel()
    cmtest=confusion_matrix_analyis(tntst, fptst, fntst, tptst )





    # opt_rf_dict[f'Parameters:{gs_rf_params}]'] = {
    # 'train_score': round(opt_rf.score(X_train, y_train), 2),
    # 'validate_score': round(opt_rf.score(X_validate, y_validate), 2),
    # 'test_score':round(opt_rf.score(X_test, y_test), 2),
    # 'diff train test': round(abs(opt_rf.score(X_train, y_train)-opt_rf.score(X_test, y_test)),2)}
    
   
    l=[cmtrain,cmval,cmtest]

    d=pd.DataFrame(l,index=['train','validate','test'])
  




    # opt_rf_dict[f'Parameters:{gs_rf_params}]'] = d



    display(symbols("Random~Forest~Test"),d)
    # Produce the classification report on the actual y values and this model's predicted y values
    forest_report = classification_report(y_test, y_pred_forest_opt_test, output_dict=True)
    display(pd.DataFrame(forest_report))
    # sklearn confusion matrix
    rf_cm = confusion_matrix(y_test, y_pred_forest_opt_test)
  
    display(Matrix(rf_cm))
    disp = ConfusionMatrixDisplay(confusion_matrix=rf_cm, display_labels=opt_rf.classes_)
    disp.plot()
    plt.show() 

    feature_names = [i for i in X_train.columns.tolist()]





    start_time = time.time()
    importances = opt_rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in opt_rf.estimators_], axis=0)
    elapsed_time = time.time() - start_time

    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")






    forest_importances = pd.Series(importances, index=feature_names)
    forest_importances=forest_importances.sort_values(ascending=False)
   

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    plt.gcf().set_size_inches( 12,6)
    fig.tight_layout() 
    return   opt_rf,forest_importances









 







def best_model_precision(X_train,y_train,X_validate,y_validate,X_test,y_test,random):
        # Make the model
    forest1 = RandomForestClassifier()

    forest_params={'n_estimators': [10,100],'max_features':['sqrt'],'max_depth':[5,20],'criterion':['gini']}
    forest1.fit(X_train, y_train)

    # # Fit the model (on train and only train)
    # forest1.fit(X_train, y_train)

    # Use the model
    # We'll evaluate the model's performance on train, first

    grid_search_rf = GridSearchCV(estimator=forest1, param_grid=forest_params, cv= 20, scoring='precision',n_jobs=-2)
    grid_search_rf.fit(X_train, y_train)
    gs_rf_params = grid_search_rf.best_params_
    gs_rf_params.update({'random_state':random})

    forest_opt = RandomForestClassifier(**gs_rf_params)
    # Fit the model (on train and only train)
    forest_opt.fit(X_train, y_train)
    y_pred_forest_opt_train= forest_opt.predict(X_train)
    y_pred_forest_opt_val= forest_opt.predict(X_validate)


    y_pred_forest_opt_test= forest_opt.predict(X_test)
    display(gs_rf_params)
    opt_rf= RandomForestClassifier(**gs_rf_params)
    # Fit the model (on train and only train)
    opt_rf_dict={}
    opt_rf.fit(X_train, y_train)
    tntr, fptr, fntr, tptr = confusion_matrix(y_train, y_pred_forest_opt_train).ravel()
    cmtrain=confusion_matrix_analyis(tntr, fptr, fntr, tptr)
    tnv, fpv, fnv, tpv = confusion_matrix(y_validate, y_pred_forest_opt_val).ravel()
    cmval=confusion_matrix_analyis( tnv, fpv, fnv, tpv)
    tntst, fptst, fntst, tptst = confusion_matrix(y_test, y_pred_forest_opt_test).ravel()
    cmtest=confusion_matrix_analyis(tntst, fptst, fntst, tptst )
    # opt_rf_dict[f'Parameters:{gs_rf_params}]'] = {
    # 'train_score': round(opt_rf.score(X_train, y_train), 2),
    # 'validate_score': round(opt_rf.score(X_validate, y_validate), 2),
    # 'test_score':round(opt_rf.score(X_test, y_test), 2),
    # 'diff train test': round(abs(opt_rf.score(X_train, y_train)-opt_rf.score(X_test, y_test)),2)}


    l=[cmtrain,cmval,cmtest]
    d=pd.DataFrame(l,index=['train','validate','test'])

    # opt_rf_dict[f'Parameters:{gs_rf_params}]'] = d
    display(symbols("Random~Forest~Test"),d)
    # Produce the classification report on the actual y values and this model's predicted y values
    forest_report = classification_report(y_test, y_pred_forest_opt_test, output_dict=True)
    display(pd.DataFrame(forest_report))
    # sklearn confusion matrix
    rf_cm = confusion_matrix(y_test, y_pred_forest_opt_test)

    display(Matrix(rf_cm))
    disp = ConfusionMatrixDisplay(confusion_matrix=rf_cm, display_labels=opt_rf.classes_)
    disp.plot()
    plt.show()

    feature_names = [i for i in X_train.columns.tolist()]





    start_time = time.time()
    importances = opt_rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in opt_rf.estimators_], axis=0)
    elapsed_time = time.time() - start_time

    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")






    forest_importances = pd.Series(importances, index=feature_names)
    forest_importances=forest_importances.sort_values(ascending=False)
    plt.figure(figsize=(10,6))
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    plt.gcf().set_size_inches( 12,6)
    fig.tight_layout() 
    return   opt_rf,forest_importances
#   accuracy,recall
