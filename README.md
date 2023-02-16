# Telco Project
----

***
[[Objectives]](#objectives)]
[[Project Description](#project_description)]
[[Project Planning](#planning)]
[[Data Dictionary](#dictionary)]
[[Data Acquire and Prep](#wrangle)]
[[Data Exploration](#explore)]
[[Modeling](#model)]
[[Conclusion](#conclusion)]
[[Steps to Reproduce](#reproduce)]

---





## <a name="objectives"></a> Objectives:
- Document the data science pipeline. Ensure the findings are presented clearly and the documentation is clear enough for independent reproduction
- Create modules that can be downloaded for the sake of reproducibility.






## <a name="project_description"></a> Project Description and goals:

- The goal is to use data to find and explore predictive factors of churn.
- Ultimately we hope to use these factors to drive actions which help to maintain a strong customer base and drive  profits.


## Questions:

Generally we would ask what relationships that might affect churn?

Is there a distinguishable relationship between household size and churn?

>This required some feature engineering.

Is there a relationship between churn and paperless billing?

Is there a correlation between customer duration and monthly charges?

- If so, how strong is it?

Due total charges have a relationship with churn?












## <a name="dictionary"></a> Data Dictionary:
Specify Target add definintions



<table>
  <tr>
   <td><strong>Variable Name</strong>
   </td>
   <td><strong>Data Type</strong>
   </td>
   <td><strong>Categorical/Numerical</strong>
   </td>
  </tr>
  <tr>
   <td>tenure
   </td>
   <td>int64
   </td>
   <td>Numerical
   </td>
  </tr>
  <tr>
   <td>monthly_charges
   </td>
   <td>float64
   </td>
   <td>Numerical
   </td>
  </tr>
  <tr>
   <td>total_charges
   </td>
   <td>float64
   </td>
   <td>Numerical
   </td>
  </tr>
  <tr>
   <td>gender_encoded
   </td>
   <td>int64
   </td>
   <td>Categorical
   </td>
  </tr>
  <tr>
   <td>partner_encoded
   </td>
   <td>int64
   </td>
   <td>Categorical
   </td>
  </tr>
  <tr>
   <td>dependents_encoded
   </td>
   <td>int64
   </td>
   <td>Categorical
   </td>
  </tr>
  <tr>
   <td>phone_service_encoded
   </td>
   <td>int64
   </td>
   <td>Categorical
   </td>
  </tr>
  <tr>
   <td>paperless_billing_encoded
   </td>
   <td>int64
   </td>
   <td>Categorical
   </td>
  </tr>
  <tr>
   <td>churn_encoded
   </td>
   <td>int64
   </td>
   <td>Categorical
   </td>
  </tr>
  <tr>
   <td>multiple_lines_No_phone_service
   </td>
   <td>uint8
   </td>
   <td>Categorical
   </td>
  </tr>
  <tr>
   <td>multiple_lines_Yes
   </td>
   <td>uint8
   </td>
   <td>Categorical
   </td>
  </tr>
  <tr>
   <td>online_security_No_internet_service
   </td>
   <td>uint8
   </td>
   <td>Categorical
   </td>
  </tr>
  <tr>
   <td>online_security_Yes
   </td>
   <td>uint8
   </td>
   <td>Categorical
   </td>
  </tr>
  <tr>
   <td>online_backup_No_internet_service
   </td>
   <td>uint8
   </td>
   <td>Categorical
   </td>
  </tr>
  <tr>
   <td>online_backup_Yes
   </td>
   <td>uint8
   </td>
   <td>Categorical
   </td>
  </tr>
  <tr>
   <td>device_protection_No_internet_service
   </td>
   <td>uint8
   </td>
   <td>Categorical
   </td>
  </tr>
  <tr>
   <td>device_protection_Yes
   </td>
   <td>uint8
   </td>
   <td>Categorical
   </td>
  </tr>
  <tr>
   <td>tech_support_No_internet_service
   </td>
   <td>uint8
   </td>
   <td>Categorical
   </td>
  </tr>
  <tr>
   <td>tech_support_Yes
   </td>
   <td>uint8
   </td>
   <td>Categorical
   </td>
  </tr>
  <tr>
   <td>streaming_tv_No_internet_service
   </td>
   <td>uint8
   </td>
   <td>Categorical
   </td>
  </tr>
  <tr>
   <td>streaming_tv_Yes
   </td>
   <td>uint8
   </td>
   <td>Categorical
   </td>
  </tr>
  <tr>
   <td>streaming_movies_No_internet_service
   </td>
   <td>uint8
   </td>
   <td>Categorical
   </td>
  </tr>
  <tr>
   <td>streaming_movies_Yes
   </td>
   <td>uint8
   </td>
   <td>Categorical
   </td>
  </tr>
  <tr>
   <td>contract_type_One_year
   </td>
   <td>uint8
   </td>
   <td>Categorical
   </td>
  </tr>
  <tr>
   <td>contract_type_Two_year
   </td>
   <td>uint8
   </td>
   <td>Categorical
   </td>
  </tr>
  <tr>
   <td>internet_service_type_Fiber_optic
   </td>
   <td>uint8
   </td>
   <td>Categorical
   </td>
  </tr>
  <tr>
   <td>internet_service_type_None
   </td>
   <td>uint8
   </td>
   <td>Categorical
   </td>
  </tr>
  <tr>
   <td>payment_type_Credit_card_(automatic)
   </td>
   <td>uint8
   </td>
   <td>Categorical
   </td>
  </tr>
  <tr>
   <td>payment_type_Electronic_check
   </td>
   <td>uint8
   </td>
   <td>Categorical
   </td>
  </tr>
  <tr>
   <td>payment_type_Mailed_check
   </td>
   <td>uint
   </td>
   <td>Categorical
   </td>
  </tr>
</table>












## Procedure:




#### <a name="planning"></a>  Planning:
Our plan is to follow the data science pipeline best practices. The steps are included below. Ultimately we are buil

#### <a name="wrangle"></a>Acquisition:
An aquire.py file is created and used. It aquires the data from the database then saves it a .csv file locally (telco.csv). Also it outputs simple graphs of the counts of unique values per variable in order to give a quick visual of whether or not the data is going to be categorical or not.

#### <a name="wrangle"></a>Preparation:
A prepare.py file is created and used. Here the data is cleaned. Categorical columns are encoded. Numerical columns are designated as floats if there are no nulls. The columns with nulls are noted to be treated at the next step. The results of this step are saved into a csv file (telco_clean.csv). 

#### <a name="explore"></a>Exploration and Pre-processing:
A preprocess.py file is created and used. Here we split our data into subsets which are train, validate and test repsectively. From here we address the colums with null values. We take the mean of the non null values in each column and impute them as the null values in the respective columns. This process is done independtley with train, validate and test in order to avoid any "data poisoning".
From here we start doing a deep dive into exploration on the train dataset. We ask questions of our data and create graphs in order to better understand our data and ask better questions. We then formulate those questions into hypothesises and do some statisitical tests to find the answers to our questions. 

#### <a name="model"></a> Modeling:
Here we use select various machine learning algorithms from the sklean library to create models. Once we have our models we can further vary our hyperparmeters in each model. From here we 

#### Delivery:
A final report is created which gives a highlevel overview of the the process. 

## <a name="reproduce"></a> Explanations for Reproducibility:
 
In order to repoduce these you will need a env.py file which contains host, username and password creditials to access the sql server. The remaining files are availble within my github repo. If you clone this repo, add a env.py file in the format shown below you will be able to reproduce the outcome. As and aside the random state is included in the file. If you were to change this your results my slightly differ.

```python

host='xxxxx'
username='xxxxxx'
password='xxxxxx'
## Where the strings are your respective credentials

```

## Executive Summary:

#### <a name="conclusion"></a> Conclusion:
We beat the baseline with our models. 

Hence, their predictive power is useful.

More data might highlight some interesting relationships.

#### Specific Recommendations:
It would be nice to obtain more quantitative data related to the projected disposable income of each household. 

#### Actionable Example:
Offer a Telco credit card. In doing so, we are able to collect more quantitative information on credit ratings and household income. This could give us insights on the projected disposable income of each household. The goal being to maximize profits by the data gained and perhaps offering incentives that diminish churn.


#### Closing Quote:
>
>“Errors using inadequate data are much less than those using no data at all.”
(Charles Babbage, English Mathematician)



