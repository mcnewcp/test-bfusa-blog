---
layout: post
title: End-to-End Machine Learning Project
author: Coy McNew
date: '2021-09-09 09:00:00'
---
# Purpose

This post is the first entry in a series where I'll be working through Aurelien Geron's exceptional book [Hands-on Machine Learning with Scikit-Learn, Keras & Tensorflow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/).  I find the most effective way to learn from these kinds of materials is to carefully work through the code and editorialize as I go.  This series will be the documentation of my efforts doing just that.  In this post, I'm working through Chapter 2 of the book - "End-to-End Machine Learning Project".  The github repo of my ongoing work through this book can be found [here](https://github.com/mcnewcp/book-geron-ml-sklearn-keras-tensorflow).

# Get The Data

## Loading the Data

First I'm going to download the data from github using a function defined in `utils.py`.


```python
import utils as ut
ut.fetch_housing_data()
```

And then I'm going to load the csv as a df using another function defined in `utils.py`.


```python
import pandas as pd
housing = ut.load_housing_data()
```

## Initial Data Investigation

Now, taking a quick look at the data.


```python
housing.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-122.23</td>
      <td>37.88</td>
      <td>41.0</td>
      <td>880.0</td>
      <td>129.0</td>
      <td>322.0</td>
      <td>126.0</td>
      <td>8.3252</td>
      <td>452600.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-122.22</td>
      <td>37.86</td>
      <td>21.0</td>
      <td>7099.0</td>
      <td>1106.0</td>
      <td>2401.0</td>
      <td>1138.0</td>
      <td>8.3014</td>
      <td>358500.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-122.24</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1467.0</td>
      <td>190.0</td>
      <td>496.0</td>
      <td>177.0</td>
      <td>7.2574</td>
      <td>352100.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1274.0</td>
      <td>235.0</td>
      <td>558.0</td>
      <td>219.0</td>
      <td>5.6431</td>
      <td>341300.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1627.0</td>
      <td>280.0</td>
      <td>565.0</td>
      <td>259.0</td>
      <td>3.8462</td>
      <td>342200.0</td>
      <td>NEAR BAY</td>
    </tr>
  </tbody>
</table>
</div>




```python
housing.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20640 entries, 0 to 20639
    Data columns (total 10 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   longitude           20640 non-null  float64
     1   latitude            20640 non-null  float64
     2   housing_median_age  20640 non-null  float64
     3   total_rooms         20640 non-null  float64
     4   total_bedrooms      20433 non-null  float64
     5   population          20640 non-null  float64
     6   households          20640 non-null  float64
     7   median_income       20640 non-null  float64
     8   median_house_value  20640 non-null  float64
     9   ocean_proximity     20640 non-null  object 
    dtypes: float64(9), object(1)
    memory usage: 1.6+ MB


A couple observations:
* all features are floats except for ocean_proximity
* total_bedrooms has some missing values

And now taking a look at the categorical feature `ocean_proximity`.


```python
housing["ocean_proximity"].value_counts()
```




    <1H OCEAN     9136
    INLAND        6551
    NEAR OCEAN    2658
    NEAR BAY      2290
    ISLAND           5
    Name: ocean_proximity, dtype: int64



Using describe to take a look at all numerical features.


```python
housing.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20433.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-119.569704</td>
      <td>35.631861</td>
      <td>28.639486</td>
      <td>2635.763081</td>
      <td>537.870553</td>
      <td>1425.476744</td>
      <td>499.539680</td>
      <td>3.870671</td>
      <td>206855.816909</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.003532</td>
      <td>2.135952</td>
      <td>12.585558</td>
      <td>2181.615252</td>
      <td>421.385070</td>
      <td>1132.462122</td>
      <td>382.329753</td>
      <td>1.899822</td>
      <td>115395.615874</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-124.350000</td>
      <td>32.540000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.499900</td>
      <td>14999.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-121.800000</td>
      <td>33.930000</td>
      <td>18.000000</td>
      <td>1447.750000</td>
      <td>296.000000</td>
      <td>787.000000</td>
      <td>280.000000</td>
      <td>2.563400</td>
      <td>119600.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-118.490000</td>
      <td>34.260000</td>
      <td>29.000000</td>
      <td>2127.000000</td>
      <td>435.000000</td>
      <td>1166.000000</td>
      <td>409.000000</td>
      <td>3.534800</td>
      <td>179700.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-118.010000</td>
      <td>37.710000</td>
      <td>37.000000</td>
      <td>3148.000000</td>
      <td>647.000000</td>
      <td>1725.000000</td>
      <td>605.000000</td>
      <td>4.743250</td>
      <td>264725.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>-114.310000</td>
      <td>41.950000</td>
      <td>52.000000</td>
      <td>39320.000000</td>
      <td>6445.000000</td>
      <td>35682.000000</td>
      <td>6082.000000</td>
      <td>15.000100</td>
      <td>500001.000000</td>
    </tr>
  </tbody>
</table>
</div>



Now using matplotlib to look at histograms of each feature.


```python
%matplotlib inline 
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()
```


    
![histograms](/assets/2021-09-09-end-to-end-ml-project_files/2021-09-09-end-to-end-ml-project_12_0.png)
    


Histogram observations:
* median income is not in USD, ~10s of thousands
* housing_median_age & median_house_value were capped
* features have very different scales
* many histograms are tail-heavy

## Split Data
Now I'm going to split data into train and test sets.


```python
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
```

Or if I want to do stratified splitting instead, for example median_income is likely a very important feature and looking at the histogram, most values are 1.5 - 6.0.  So defining custom income categories below and taking a look at a quick histogram.


```python
import numpy as np
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

housing["income_cat"].hist()
plt.show()
```


    
![income category](/assets/2021-09-09-end-to-end-ml-project_files/2021-09-09-end-to-end-ml-project_16_0.png)
    


Now I'll use `StratifiedShuffleSplit` from `sklearn` along with my newly defined income category.


```python
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

#making sure it worked
print("test ratios:\n", strat_test_set["income_cat"].value_counts()/len(strat_test_set))
print("train ratios:\n", strat_train_set["income_cat"].value_counts()/len(strat_train_set))

#dropping income_cat
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
```

    test ratios:
     3    0.350533
    2    0.318798
    4    0.176357
    5    0.114583
    1    0.039729
    Name: income_cat, dtype: float64
    train ratios:
     3    0.350594
    2    0.318859
    4    0.176296
    5    0.114402
    1    0.039850
    Name: income_cat, dtype: float64


# Discover and Visualize the Data

## Geographical
I'm using `matplotlib` to take a look at the geographical spread of data and further investigate.


```python
#create a copy of training set to explore
housing = strat_train_set.copy()

#take a look at the data geographically 
housing.plot(kind="scatter", x="longitude", y="latitude")
plt.show()
```


    
![mapped locations](/assets/2021-09-09-end-to-end-ml-project_files/2021-09-09-end-to-end-ml-project_20_0.png)
    


All I see here is the state of Ca, so I'm going to start by increasing transparency to see density.


```python
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.show()
```


    
![cluster map](/assets/2021-09-09-end-to-end-ml-project_files/2021-09-09-end-to-end-ml-project_22_0.png)
    


Now I can see the clusters around LA, SF, and Sacramento.  I'll add a couple more variables to the plot:
* point size (s) is population
* point color (c) is median house value, the target variable


```python
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             )
plt.legend()
plt.show()
```


    
![final map](/assets/2021-09-09-end-to-end-ml-project_files/2021-09-09-end-to-end-ml-project_24_0.png)
    


This plot gives me much more information, including the follow observations which will likely help me in model selection/building:
* Median house value is highly dependent upon location, e.g. how close to the ocean?
* Median house value is highly dependent upon population density.
    * A clustering algorithm could be used for detecting main population clusters and defining a new feature of distance to main clusters.

## Correlation Analysis
In order to get an idea of how well each feature correlates with the target variable, I'm going to run a correlation analysis and check the results for median house value.


```python
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
```




    median_house_value    1.000000
    median_income         0.687160
    total_rooms           0.135097
    housing_median_age    0.114110
    households            0.064506
    total_bedrooms        0.047689
    population           -0.026920
    longitude            -0.047432
    latitude             -0.142724
    Name: median_house_value, dtype: float64



There are a few promising features here, including `median_income`, `total_rooms`, and `housing_median_age`, so I'm going to look at scatter plots for each.


```python
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()
```


    
![biplots](/assets/2021-09-09-end-to-end-ml-project_files/2021-09-09-end-to-end-ml-project_28_0.png)
    



```python
#taking a closer look at the median_income correlation
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
plt.show()
```


    
![biplot](/assets/2021-09-09-end-to-end-ml-project_files/2021-09-09-end-to-end-ml-project_29_0.png)
    


The price cap at 500,000 is clearly visible in this plot.  There are also a couple other horizontal lines at 450,000, 350,000, etc.  These districts may need to be removed before training a model.  The next thing I'll try is combining a couple features to see if they correlate well with the target variable.


```python
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
```




    median_house_value          1.000000
    median_income               0.687160
    rooms_per_household         0.146285
    total_rooms                 0.135097
    housing_median_age          0.114110
    households                  0.064506
    total_bedrooms              0.047689
    population_per_household   -0.021985
    population                 -0.026920
    longitude                  -0.047432
    latitude                   -0.142724
    bedrooms_per_room          -0.259984
    Name: median_house_value, dtype: float64



Nice!  `bedrooms_per_room` is much more correlated with median house value than `total_rooms` or `total_bedrooms`. `rooms_per_household` is also more informative than `total_rooms`.

# Prepare the Data

## Handling Missing Values

`total_bedrooms` has missing values in approx 200 rows.  A couple options for handling this include:
* drop all observations with missing values
    * housing.dropna(subset=["total_bedrooms"])
* drop the feature entirely
    * housing.drop("total_bedrooms", axis=1)
* impute missing values with constant, e.g. median
    * median = housing["total_bedrooms"].median()
    * housing["total_bedrooms"].fillna(median, inplace=True)

Instead I'm going to use `SimpleImputer` from `sklearn` so I can build into a pipeline later.


```python
#fresh copy of training dataset
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

#this can only be applied to numerical attributes
housing_num = housing.drop("ocean_proximity", axis=1)

#now to fit the imputer to the training data
imputer.fit(housing_num)

#only the total_bedrooms varialbe had missing values but we can't be sure that 
#   will be true for new data so it's safer to fit all features

#now to transform the numeric values
X = imputer.transform(housing_num)
#and if I want to put it back into a df
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
```

## Handling Text and Categorical Features

The only text feature in this dataset is `ocean_proximity` and when we looked at it above we saw that it wasn't free or arbitrary text, but rather categorical with 5 possible values.  There are a couple common ways to encode this variable for use in our model:
* Ordinal encoding, where each possible category is assigned an integer value.  This works well for categories which have an inherent order to them, e.g. bad, average, good, because the model will assume that two values which are near each other are more similar than two that are far apart.
* One-hot encoding, where a new binary feature is defined for each possible category and a value of 0 or 1 is assigned.  

### Ordinal Encoding


```python
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat = housing[["ocean_proximity"]]
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
ordinal_encoder.categories_
```




    [array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
           dtype=object)]



I could either think carefully around exactly the order of categories I need for ordinal encoding or simply apply one-hot encoding, as it generally seems to be the preferred option if your memory can handle it.

### One-Hot Encoding


```python
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
```




    <16512x5 sparse matrix of type '<class 'numpy.float64'>'
    	with 16512 stored elements in Compressed Sparse Row format>



It is interesting to note here that the resulting object is stored as a sparse matrix rather than a more typical numpy array.  This is a far more memory efficient format as it only stores the locations of the non-zero elements.  It can be used just like any other 2D arrays.

## Custom Transformers

You can also write your own custom transformers for tasks such as custom cleanup ooperations or combining specific attributes.  If you're careful to write these transformers in such a way to work with Scikit-Learn functionalities (e.g. pipelines) then they can be very powerful.  In particular you need to create a class and implement the following three methods:
* `fit()`, returning self
* `transform()`
* `fit_transform()`

You can get `fit_transform()` for free by simply adding `TransformerMixin` as a base class.  If you add `BaseEstimator` as a base class (and avoid `*args` and `**kargs` in your constructor), you will also get two extra methods (`get_params()` and `set_params()`) that will be useful for automatic hyperparameter tuning.  As an example, below is a small transformer class that adds the combined features I investigated above.


```python
from sklearn.base import BaseEstimator, TransformerMixin

#define column locations
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    #one hyperparameter: add_bedrooms_per_room
    def __init__(self, add_bedrooms_per_room = True): #no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self #nothing to fit
    def transform(self, X):
        #generate combined attributes
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
```

In this example the transformer has one hyperparameter, `add_bedrooms_per_room`, which is set to True by default.  By using this in the hyperparameter tuning step, I will be able to tell whether or not adding this additional feature actually improves the model.  Furthermore, adding hyperparameters to gate any data preparation step that you are not 100% sure about is generally a good idea.  The more automation you do up front, the more combinations you can automatically try out, making it much more likely that you'll end up with a good combination.

## Feature Scaling

Feature scaling is typically very important when it comes to ML models as most don't work well with features that are of differing scales.  There are two common ways of feature scaling in Sci-Kit Learn:
* `MinMaxScaler` - sometimes called normalization is the simplest.  It takes each value, subtracts the min and divides by the max - min.  In this way, all features are scaled to 0 - 1.  
* `StandardScaler` - first subtracts the mean (so standardized values always have a zero mean) and then divides by the standard deviation so the resulting distribution has unit variance.

Standardization does not bound values to a specific range, which may be a problem for some algorithms (e.g. neural networks) but standardization is much less affect by outliers.  For example, suppose a district had a median income of 100 by mistake, then normalization would cram all other values from 0-15 down to 0-0.15, whereas standardization would not be much affected.  I'm going to apply standardization in the next section on pipelines.

## Pipelines

Pipelines are hugely helpful in ML workflows, especially when tuning hyperparameters.  The pipeline constructor takes a list of name/estimator pairs defining a sequence of steps.  All but the last estimator must be transformers (i.e., they must have a `fit_transform()` method).  When you call the pipeline's `fit()` method, it calls `fit_transform()` sequentially on all transformers, passing the output of each call as the parameter to the next call untill it reaches the final estimator, for which it calls the `fit()` mehod.  The pipeline exposes the same methods as the final estimator, which in the example below is a `StandardScaler` so it has a `fit_transform()` method, which is what I use instead.


```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

housing_num_tr = num_pipeline.fit_transform(housing_num)
```

Sci-Kit Learn also has the `ColumnTransformer` which allows you to apply pipelines to different columns, which can be defined using pandas column names.  This is very helpful for applying different transformations to numeric or categorical columns, as in the example below.


```python
from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
])

housing_prepared = full_pipeline.fit_transform(housing)
```

# Select and Train a Model

First, I'll fit a linear regression model to the data.


```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

#checking performance with rmse
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
```




    68628.19819848922



Ok an rmse of $68,000 isn't very good when you consider most district's `median_housing_value` range between $120,000 and $265,000.  This model is clearly underfitting the data. Now I'm going to a more powerful model, `DecisionTreeRegressor()`.


```python
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

#and evaluating rmse on the training set
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
```




    0.0



Ok, an rmse of 0 clearly tells us that this model is massively overfit.  It's no surprise really since the model was trained on the entire training set and also evaluated on the same set.  In order to remedy the overfitting data, I need to split the training data into a training and validation set.

## Better Evaluation Using Cross-Validation

One way to accomplish this is to use `train_test_split()` again to split the training data into a training set and a validation set for model evaluation.  An alternative and often agreed upon better method is to use Sci-Kit Learn's K-fold cross-validation feature.  The following code splits the training data into 10 distinct folds, then it trains and evaluates the Decision Tree model 10 times, picking a different fold for evaluation every time and training on the other 9 folds.  The result is an array containing the 10 evaluation scores.


```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                        scoring = "neg_mean_squared_error", cv=10)
#note that sklearn's cv features expect a utility function (lower is better) so the scoring function is actually the negative of MSE
tree_rmse_scores = np.sqrt(-scores)

#and displaying the scores
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())

display_scores(tree_rmse_scores)
```

    Scores: [69650.86819788 66756.26581791 71535.93489896 69736.26957968
     71483.70073511 75912.86258227 68667.89253772 71555.82238666
     76933.30336158 70814.62338016]
    Mean: 71304.75434779239
    Standard Deviation: 2934.9825767828343


Just as we suspected above, the model is overfitting the training data so badly that it's performing worse on the validation set than the linear regression model.  I'm now going to try a couple more models below, `RandomForestRegressor` and `SVR`.


```python
#random forest
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
#evaluating rmse on the training set
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
#evaluating rmse using cv
scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                        scoring = "neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-scores)
display_scores(forest_rmse_scores)
```

    Scores: [49389.1546339  47464.22092432 49460.28008398 52289.23877996
     49604.96647304 53340.67810141 48438.42561798 47934.50166784
     52948.66234675 50116.11123813]
    Mean: 50098.62398673056
    Standard Deviation: 1974.0066738469416



```python
forest_rmse
```




    18822.598421559




```python
#support vector machine
from sklearn.svm import SVR
svm_reg = SVR()
svm_reg.fit(housing_prepared, housing_labels)
#evaluating rmse on the training set
housing_predictions = svm_reg.predict(housing_prepared)
svm_mse = mean_squared_error(housing_labels, housing_predictions)
svm_rmse = np.sqrt(svm_mse)
#evaluating rmse using cv
scores = cross_val_score(svm_reg, housing_prepared, housing_labels,
                        scoring = "neg_mean_squared_error", cv=10)
svm_rmse_scores = np.sqrt(-scores)
display_scores(svm_rmse_scores)
```

    Scores: [111389.0681902  119541.25938571 116957.62830414 120447.19932481
     117618.15904234 122309.10351544 117634.40230741 121469.713921
     120343.01369623 118017.12860651]
    Mean: 118572.66762937943
    Standard Deviation: 2936.877586794944



```python
svm_rmse
```




    118580.68301157995



The random forest is far better than the decision tree regressor though it is still overfitting the data.  Possible solutions are to simplify the model, constrain it (i.e., regularize it), or get a lot more training data.  The SVM model I tried performs very poorly.  I'm not entirely sure why, but I likely didn't choose the most appropriate kernel.  I'll now move forward with a couple models into model tuning.

## Fine-Tune Your Model

Sci-Kit Learn has a couple built in classes to optmize hyperparameters: `GridSearchCV` and `RandomizedSearchCV`.  In addition there are other libraries to do more optimized hyperparameter tuning, including Optuna, but I'm just looking into the Sci-Kit Learn options here.

### GridSearchCV

In order to use `GridSearchCV` you supply it with one or more dictionaries of hyperparameter values and it tries all combinations of the values with cv scoring.  For example, the param grid below tells `GridSearchCV` to first evaluate all possible combinations of the first dict (3x4 = 12) and then try all possible combinations in the second grid (2x3 = 6), leading to 18 total combinations.  Since I specified 5 fold cv, that will lead to 5 x 18 = 90 rounds of training.


```python
from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
]
forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(
    forest_reg, param_grid, cv=5,
    scoring='neg_mean_squared_error',
    return_train_score=True
)
grid_search.fit(housing_prepared, housing_labels)
```




    GridSearchCV(cv=5, estimator=RandomForestRegressor(),
                 param_grid=[{'max_features': [2, 4, 6, 8],
                              'n_estimators': [3, 10, 30]},
                             {'bootstrap': [False], 'max_features': [2, 3, 4],
                              'n_estimators': [3, 10]}],
                 return_train_score=True, scoring='neg_mean_squared_error')



You can get the best params directly using `.best_params_`.  Since 30 is the maximum `n_estimators` I evaluated, I should probably search again using even higher values.  You can also get the best estimator directly using `.best_estimator_` and of course the individual cv scores using `.cv_results_`.  It's also good to note here that if `GridSearchCV` is initialized with `refit=True`, which is the default, then once it finds the best estimator using cv, it retrains on the whole training set.


```python
grid_search.best_params_
```




    {'max_features': 6, 'n_estimators': 30}




```python
grid_search.best_estimator_
```




    RandomForestRegressor(max_features=6, n_estimators=30)




```python
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
```

    64096.428979611126 {'max_features': 2, 'n_estimators': 3}
    55370.45067516444 {'max_features': 2, 'n_estimators': 10}
    52828.40432351358 {'max_features': 2, 'n_estimators': 30}
    59520.862308977245 {'max_features': 4, 'n_estimators': 3}
    52712.49612037059 {'max_features': 4, 'n_estimators': 10}
    50721.04115644037 {'max_features': 4, 'n_estimators': 30}
    59086.637312506704 {'max_features': 6, 'n_estimators': 3}
    52225.11921182246 {'max_features': 6, 'n_estimators': 10}
    49915.38635628875 {'max_features': 6, 'n_estimators': 30}
    59908.18635411256 {'max_features': 8, 'n_estimators': 3}
    51902.7112591161 {'max_features': 8, 'n_estimators': 10}
    50073.94232440752 {'max_features': 8, 'n_estimators': 30}
    61806.362753358546 {'bootstrap': False, 'max_features': 2, 'n_estimators': 3}
    54274.47829225807 {'bootstrap': False, 'max_features': 2, 'n_estimators': 10}
    60373.16787959317 {'bootstrap': False, 'max_features': 3, 'n_estimators': 3}
    53027.58261195693 {'bootstrap': False, 'max_features': 3, 'n_estimators': 10}
    57811.522049925974 {'bootstrap': False, 'max_features': 4, 'n_estimators': 3}
    51808.8671353706 {'bootstrap': False, 'max_features': 4, 'n_estimators': 10}


I should also mention here that you can treat data preparation steps as hyperparameters as well, for example the `add_bedrooms_per_room` option we added to the `CombinedAttributesAdder` above.  `GridSearchCV` will then automatically find out whether or not to add a feature that we were unsure about.  Similarly, we could also use it to find the best way to handle outliers, missing features, feature selection, etc.

### RandomozedSearchCV

In the example below I'm going to take the information from the previous tuning step and use `RandomizedSearchCV` to further refine the model.  First, I'm going to build a pipeline to include my data preparation steps so that I can include the `add_bedrooms_per_room` hyperparameter and I'll center the ranges of the above hyperparameters around their best values from above.


```python
#defining a full pipeline with preprocessing steps and model training
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

ct = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
])
rfr_pipeline = Pipeline([
    ('ct', ct),
    ('rfr', RandomForestRegressor())
])

#define hyperparameter grid to be randomly searched
param_grid = [
    {'rfr__n_estimators': range(20, 40), 'rfr__max_features': range(5, 10), 'ct__num__attribs_adder__add_bedrooms_per_room': [True, False]}
]

#run RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
rfr_search = RandomizedSearchCV(
    rfr_pipeline, param_grid, cv=5, n_iter=20,
    scoring='neg_mean_squared_error',
    return_train_score=True
)
rfr_search.fit(housing, housing_labels)

cvres = rfr_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
```

    50688.647636169 {'rfr__n_estimators': 21, 'rfr__max_features': 8, 'ct__num__attribs_adder__add_bedrooms_per_room': True}
    50034.38735935574 {'rfr__n_estimators': 33, 'rfr__max_features': 5, 'ct__num__attribs_adder__add_bedrooms_per_room': True}
    49979.657619467835 {'rfr__n_estimators': 38, 'rfr__max_features': 8, 'ct__num__attribs_adder__add_bedrooms_per_room': True}
    49755.706801222084 {'rfr__n_estimators': 38, 'rfr__max_features': 9, 'ct__num__attribs_adder__add_bedrooms_per_room': False}
    49895.671695998695 {'rfr__n_estimators': 39, 'rfr__max_features': 9, 'ct__num__attribs_adder__add_bedrooms_per_room': False}
    49348.89105889358 {'rfr__n_estimators': 36, 'rfr__max_features': 7, 'ct__num__attribs_adder__add_bedrooms_per_room': False}
    49885.774942888136 {'rfr__n_estimators': 27, 'rfr__max_features': 5, 'ct__num__attribs_adder__add_bedrooms_per_room': False}
    50354.94859316871 {'rfr__n_estimators': 28, 'rfr__max_features': 9, 'ct__num__attribs_adder__add_bedrooms_per_room': True}
    49708.4061229694 {'rfr__n_estimators': 26, 'rfr__max_features': 6, 'ct__num__attribs_adder__add_bedrooms_per_room': False}
    49916.181465486436 {'rfr__n_estimators': 26, 'rfr__max_features': 9, 'ct__num__attribs_adder__add_bedrooms_per_room': False}
    49920.672697449874 {'rfr__n_estimators': 39, 'rfr__max_features': 7, 'ct__num__attribs_adder__add_bedrooms_per_room': True}
    49930.48957724172 {'rfr__n_estimators': 29, 'rfr__max_features': 6, 'ct__num__attribs_adder__add_bedrooms_per_room': True}
    50150.01917026861 {'rfr__n_estimators': 22, 'rfr__max_features': 8, 'ct__num__attribs_adder__add_bedrooms_per_room': False}
    49993.32026255731 {'rfr__n_estimators': 33, 'rfr__max_features': 9, 'ct__num__attribs_adder__add_bedrooms_per_room': True}
    50324.41116608029 {'rfr__n_estimators': 22, 'rfr__max_features': 8, 'ct__num__attribs_adder__add_bedrooms_per_room': True}
    49647.94486617563 {'rfr__n_estimators': 36, 'rfr__max_features': 7, 'ct__num__attribs_adder__add_bedrooms_per_room': True}
    49547.70058617279 {'rfr__n_estimators': 37, 'rfr__max_features': 5, 'ct__num__attribs_adder__add_bedrooms_per_room': False}
    50210.83668911258 {'rfr__n_estimators': 24, 'rfr__max_features': 9, 'ct__num__attribs_adder__add_bedrooms_per_room': False}
    49701.72186612198 {'rfr__n_estimators': 34, 'rfr__max_features': 6, 'ct__num__attribs_adder__add_bedrooms_per_room': True}
    49696.795226234186 {'rfr__n_estimators': 31, 'rfr__max_features': 5, 'ct__num__attribs_adder__add_bedrooms_per_room': False}


It looks like I've improved my model an additional $300 or so by using the best combinations of hyperparams.  It's also interesting to note that the best combination did not include the bedrooms_per_room feature, which means it's likely worthless.  I'll also note here that you can use `RandomizedSearchCV` or `GridSearchCV` to evaluate model algorithms as well.  I'll provide an example below using both `RandomForestRegressor` as well as `SVR`.


```python
#include a general model placeholder
mod_pipeline = Pipeline([
    ('ct', ct),
    ('mod', RandomForestRegressor())
])

#define hyperparameter grid to be randomly searched
param_grid = [
    {
        'mod': [RandomForestRegressor()],
        'mod__n_estimators': range(30, 40), 'mod__max_features': range(5, 10), 
        'ct__num__attribs_adder__add_bedrooms_per_room': [True, False]
    },
    {
        'mod': [SVR()],
        'mod__kernel': ['linear', 'rbf'], 'mod__C': [1, 10, 100],
        'ct__num__attribs_adder__add_bedrooms_per_room': [True, False]
    }
]

#run RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
mod_search = RandomizedSearchCV(
    mod_pipeline, param_grid, cv=5, n_iter=20,
    scoring='neg_mean_squared_error',
    return_train_score=True
)
mod_search.fit(housing, housing_labels)

cvres = mod_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
```

    49866.763374310045 {'mod__n_estimators': 37, 'mod__max_features': 9, 'mod': RandomForestRegressor(max_features=6, n_estimators=38), 'ct__num__attribs_adder__add_bedrooms_per_room': True}
    49556.5981387067 {'mod__n_estimators': 33, 'mod__max_features': 6, 'mod': RandomForestRegressor(max_features=6, n_estimators=38), 'ct__num__attribs_adder__add_bedrooms_per_room': False}
    49887.59172937205 {'mod__n_estimators': 33, 'mod__max_features': 9, 'mod': RandomForestRegressor(max_features=6, n_estimators=38), 'ct__num__attribs_adder__add_bedrooms_per_room': True}
    112973.43813971068 {'mod__kernel': 'linear', 'mod__C': 1, 'mod': SVR(), 'ct__num__attribs_adder__add_bedrooms_per_room': False}
    49725.011264272915 {'mod__n_estimators': 39, 'mod__max_features': 9, 'mod': RandomForestRegressor(max_features=6, n_estimators=38), 'ct__num__attribs_adder__add_bedrooms_per_room': True}
    49691.77323079302 {'mod__n_estimators': 30, 'mod__max_features': 5, 'mod': RandomForestRegressor(max_features=6, n_estimators=38), 'ct__num__attribs_adder__add_bedrooms_per_room': False}
    49888.7748159535 {'mod__n_estimators': 33, 'mod__max_features': 7, 'mod': RandomForestRegressor(max_features=6, n_estimators=38), 'ct__num__attribs_adder__add_bedrooms_per_room': True}
    49429.38487359406 {'mod__n_estimators': 38, 'mod__max_features': 6, 'mod': RandomForestRegressor(max_features=6, n_estimators=38), 'ct__num__attribs_adder__add_bedrooms_per_room': False}
    49869.091111047564 {'mod__n_estimators': 38, 'mod__max_features': 6, 'mod': RandomForestRegressor(max_features=6, n_estimators=38), 'ct__num__attribs_adder__add_bedrooms_per_room': True}
    50074.34181027114 {'mod__n_estimators': 36, 'mod__max_features': 5, 'mod': RandomForestRegressor(max_features=6, n_estimators=38), 'ct__num__attribs_adder__add_bedrooms_per_room': True}
    116125.14865684793 {'mod__kernel': 'rbf', 'mod__C': 10, 'mod': SVR(), 'ct__num__attribs_adder__add_bedrooms_per_room': True}
    84653.68946568908 {'mod__kernel': 'linear', 'mod__C': 10, 'mod': SVR(), 'ct__num__attribs_adder__add_bedrooms_per_room': True}
    49988.259377928225 {'mod__n_estimators': 33, 'mod__max_features': 5, 'mod': RandomForestRegressor(max_features=6, n_estimators=38), 'ct__num__attribs_adder__add_bedrooms_per_room': True}
    49558.43344033735 {'mod__n_estimators': 37, 'mod__max_features': 8, 'mod': RandomForestRegressor(max_features=6, n_estimators=38), 'ct__num__attribs_adder__add_bedrooms_per_room': False}
    49503.607643476025 {'mod__n_estimators': 34, 'mod__max_features': 5, 'mod': RandomForestRegressor(max_features=6, n_estimators=38), 'ct__num__attribs_adder__add_bedrooms_per_room': False}
    49825.12874956652 {'mod__n_estimators': 36, 'mod__max_features': 8, 'mod': RandomForestRegressor(max_features=6, n_estimators=38), 'ct__num__attribs_adder__add_bedrooms_per_room': True}
    49703.23680947157 {'mod__n_estimators': 38, 'mod__max_features': 7, 'mod': RandomForestRegressor(max_features=6, n_estimators=38), 'ct__num__attribs_adder__add_bedrooms_per_room': True}
    50256.06603238222 {'mod__n_estimators': 31, 'mod__max_features': 9, 'mod': RandomForestRegressor(max_features=6, n_estimators=38), 'ct__num__attribs_adder__add_bedrooms_per_room': True}
    49678.984418478 {'mod__n_estimators': 32, 'mod__max_features': 5, 'mod': RandomForestRegressor(max_features=6, n_estimators=38), 'ct__num__attribs_adder__add_bedrooms_per_room': False}
    50292.04106290782 {'mod__n_estimators': 31, 'mod__max_features': 5, 'mod': RandomForestRegressor(max_features=6, n_estimators=38), 'ct__num__attribs_adder__add_bedrooms_per_room': True}



```python
mod_search.best_params_
```




    {'mod__n_estimators': 38,
     'mod__max_features': 6,
     'mod': RandomForestRegressor(max_features=6, n_estimators=38),
     'ct__num__attribs_adder__add_bedrooms_per_room': False}



## Evaluate Your System on the Test Set


```python
final_model = rfr_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

final_predictions = final_model.predict(X_test)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print("RMSE on Test set:", final_rmse)
```

    RMSE on Test set: 47218.11203561605


Sometimes a point estimate of error is not good enough.  Below I'm generating a 95% confidence interval.


```python
from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                        loc=squared_errors.mean(),
                        scale=stats.sem(squared_errors)))

```




    array([45251.39368577, 49106.12566586])


