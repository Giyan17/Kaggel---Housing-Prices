#Data preprocessing

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset_original = pd.read_csv('train.csv' , header='infer')
dataset = pd.read_csv('train.csv')

#Empty entries
null_train=dataset.isnull().sum()

## Replace using mean # (LotFrontage) ##
mean = dataset['LotFrontage'].mean() 
dataset['LotFrontage'].fillna(mean, inplace=True)


## Replace  # (Alley) ##
dataset['Alley'].fillna('ANone', inplace=True)


## Replace  # (MasVnrType) ##
dataset['MasVnrType'].fillna('None', inplace=True)


## Replace  # (MasVnrArea ) ##
dataset['MasVnrArea'].fillna( 0 , inplace=True)


## Replace  # (FireplaceQu ) ##
dataset['FireplaceQu'].fillna( 'Anone' , inplace=True)


## Replace  # (GarageType ) ##
dataset['GarageType'].fillna( 'none' , inplace=True)


## Replace  # (GarageYrBlt ) ##
dataset['GarageYrBlt'].fillna( '0' , inplace=True)

## Replace  # (GarageFinish ) ##
dataset['GarageFinish'].fillna( 'none' , inplace=True)


## Replace  # (GarageQual  ) ##
dataset['GarageQual'].fillna( 'none' , inplace=True)


## Replace  # (GarageCond  ) ##
dataset['GarageCond'].fillna( 'none' , inplace=True)


## Replace  # (PoolQC    ) ##
dataset['PoolQC'].fillna( 'none' , inplace=True)


## Replace  # (Fence   ) ##
dataset['Fence'].fillna( 'none' , inplace=True)


## Replace  # (MiscFeature ) ##
dataset['MiscFeature'].fillna( 'none' , inplace=True)


## Replace  # (BsmtQual ) ##
dataset['BsmtQual'].fillna( 'none' , inplace=True)


## Replace  # (BsmtCond ) ##
dataset['BsmtCond'].fillna( 'none' , inplace=True)


## Replace  # (BsmtExposure) ##
dataset['BsmtExposure'].fillna( 'none' , inplace=True)


## Replace  # (BsmtFinType1 ) ##
dataset['BsmtFinType1'].fillna( 'none' , inplace=True)


## Replace  # (BsmtFinType2 ) ##
dataset['BsmtFinType2'].fillna( 'none' , inplace=True)


## Replace  # (Electrical) ##
dataset['Electrical'].fillna( 'SBrkr' , inplace=True)

#Totoal Empty entries
dataset.isnull().sum().sum()


####################################################
X = dataset.iloc[:, 0:80].values
y = dataset.iloc[:, 80].values



# Now we will encode the dataset categorical columns
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2]) #MSZoning
X[:, 5] = labelencoder_X.fit_transform(X[:, 5]) #street
X[:, 6] = labelencoder_X.fit_transform(X[:, 6]) #Alley
X[:, 7] = labelencoder_X.fit_transform(X[:, 7]) #LotShape
X[:, 8] = labelencoder_X.fit_transform(X[:, 8]) #LandContour
X[:, 9] = labelencoder_X.fit_transform(X[:, 9]) #Utilities
X[:, 10] = labelencoder_X.fit_transform(X[:, 10]) #LotConfig
X[:, 11] = labelencoder_X.fit_transform(X[:, 11]) #LandSlope
X[:, 12] = labelencoder_X.fit_transform(X[:, 12]) #Neighborhood
X[:, 13] = labelencoder_X.fit_transform(X[:, 13]) #Condition1
X[:, 14] = labelencoder_X.fit_transform(X[:, 14]) #Condition2
X[:, 15] = labelencoder_X.fit_transform(X[:, 15]) #BldgType
X[:, 16] = labelencoder_X.fit_transform(X[:, 16]) #HouseStyle
X[:, 21] = labelencoder_X.fit_transform(X[:, 21]) #RoofStyle
X[:, 22] = labelencoder_X.fit_transform(X[:, 22]) #RoofMatl
X[:, 23] = labelencoder_X.fit_transform(X[:, 23]) #Exterior1st
X[:, 24] = labelencoder_X.fit_transform(X[:, 24]) #Exterior2nd
X[:, 25] = labelencoder_X.fit_transform(X[:, 25]) #MasVnrType
X[:, 25] = labelencoder_X.fit_transform(X[:, 25]) #MasVnrType
X[:, 27] = labelencoder_X.fit_transform(X[:, 27]) #ExterQual
X[:, 28] = labelencoder_X.fit_transform(X[:, 28]) #ExterCond
X[:, 29] = labelencoder_X.fit_transform(X[:, 29]) #Foundation
X[:, 30] = labelencoder_X.fit_transform(X[:, 30]) #Bsmtqual
X[:, 31] = labelencoder_X.fit_transform(X[:, 31]) #BsmtCond
X[:, 32] = labelencoder_X.fit_transform(X[:, 32]) #BsmtExposure
X[:, 33] = labelencoder_X.fit_transform(X[:, 33]) #BsmtFinType1
X[:, 35] = labelencoder_X.fit_transform(X[:, 35]) #BsmtFinType2
X[:, 39] = labelencoder_X.fit_transform(X[:, 39]) #Heating
X[:, 40] = labelencoder_X.fit_transform(X[:, 40]) #HeatingQC
X[:, 41] = labelencoder_X.fit_transform(X[:, 41]) #CentralAir
X[:, 42] = labelencoder_X.fit_transform(X[:, 42]) #Electrical
X[:, 53] = labelencoder_X.fit_transform(X[:, 53]) #KitchenQual
X[:, 55] = labelencoder_X.fit_transform(X[:, 55]) #Functional
X[:, 57] = labelencoder_X.fit_transform(X[:, 57]) #FireplaceQu
X[:, 58] = labelencoder_X.fit_transform(X[:, 58]) #GarageType
X[:, 60] = labelencoder_X.fit_transform(X[:, 60]) #GarageFinish
X[:, 63] = labelencoder_X.fit_transform(X[:, 63]) #GarageQual
X[:, 64] = labelencoder_X.fit_transform(X[:, 64]) #GarageCond
X[:, 65] = labelencoder_X.fit_transform(X[:, 65]) #PavedDrive
X[:, 72] = labelencoder_X.fit_transform(X[:, 72]) #PoolQC
X[:, 73] = labelencoder_X.fit_transform(X[:, 73]) #Fence
X[:, 74] = labelencoder_X.fit_transform(X[:, 74]) #MiscFeature
X[:, 78] = labelencoder_X.fit_transform(X[:, 78]) #SaleType
X[:, 79] = labelencoder_X.fit_transform(X[:, 79]) #SaleCondition

X_train = X[: , 1:80]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y = sc_y.fit_transform(y.reshape(-1, 1))

#########################
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y)
#########################

#########################
# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y)
#########################

#########################
# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y)
#########################

#########################
# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y)
#########################

######################         Test Set      ###############################
#Importing test Dataset
dataset_test = pd.read_csv('test.csv' )
null_test1=dataset_test.isnull().sum()

## Replace  # (MSZoning) ##
dataset_test['MSZoning'].fillna('RL', inplace=True)

## Replace using mean # (LotFrontage) ##
mean = dataset_test['LotFrontage'].mean() 
dataset_test['LotFrontage'].fillna(mean, inplace=True)

## Replace  # (Alley) ##
dataset_test['Alley'].fillna('ANone', inplace=True)

## Replace  # (Utilities) ##
dataset_test['Utilities'].fillna('None', inplace=True)

## Replace  # (Exterior1st) ##
dataset_test['Exterior1st'].fillna('None', inplace=True)

## Replace  # (Exterior2nd) ##
dataset_test['Exterior2nd'].fillna('None', inplace=True)

## Replace  # (MasVnrType) ##
dataset_test['MasVnrType'].fillna('None', inplace=True)

## Replace  # (MasVnrArea ) ##
dataset_test['MasVnrArea'].fillna( 0 , inplace=True)

## Replace  # (BsmtQual ) ##
dataset_test['BsmtQual'].fillna( 'TA' , inplace=True)

## Replace  # (BsmtCond ) ##
dataset_test['BsmtCond'].fillna( 'TA' , inplace=True)

## Replace  # (BsmtExposure ) ##
dataset_test['BsmtExposure'].fillna( 'No' , inplace=True)

## Replace  # (BsmtFinType1 ) ##
dataset_test['BsmtFinType1'].fillna( 'Unf' , inplace=True)

## Replace  # (BsmtFinSF1 ) ##
dataset_test['BsmtFinSF1'].fillna( 0 , inplace=True)

## Replace  # (BsmtFinType2 ) ##
dataset_test['BsmtFinType2'].fillna( 'Unf' , inplace=True)

## Replace  # (BsmtFinSF2 ) ##
dataset_test['BsmtFinSF2'].fillna( 0 , inplace=True)

## Replace  # (BsmtUnfSf ) ##
mean_BsmtUnfSf = dataset_test['BsmtUnfSF'].mean() 
dataset_test['BsmtUnfSF'].fillna( mean_BsmtUnfSf , inplace=True)

## Replace  # (TotalBsmtSF) ##
mean_TotalBsmtSF = dataset_test['TotalBsmtSF'].mean() 
dataset_test['TotalBsmtSF'].fillna( mean_TotalBsmtSF , inplace=True)

## Replace  # (BsmtFullBath ) ##
dataset_test['BsmtFullBath'].fillna( 0 , inplace=True)

## Replace  # (BsmtHalfBath ) ##
dataset_test['BsmtHalfBath'].fillna( 0 , inplace=True)

## Replace  # (KitchenQual) ##
dataset_test['KitchenQual'].fillna( 'TA' , inplace=True)

## Replace  # (Functional) ##
dataset_test['Functional'].fillna( 'Typ' , inplace=True)


## Replace  # (FireplaceQu ) ##
dataset_test['FireplaceQu'].fillna( 'Anone' , inplace=True)


## Replace  # (GarageType ) ##
dataset_test['GarageType'].fillna( 'none' , inplace=True)


## Replace  # (GarageYrBlt ) ##
dataset_test['GarageYrBlt'].fillna( '0' , inplace=True)

## Replace  # (GarageFinish ) ##
dataset_test['GarageFinish'].fillna( 'none' , inplace=True)

## Replace  # (GarageCars ) ##
dataset_test['GarageCars'].fillna( 0 , inplace=True)

## Replace  # (GarageArea ) ##
dataset_test['GarageArea'].fillna( 0 , inplace=True)

## Replace  # (GarageQual  ) ##
dataset_test['GarageQual'].fillna( 'none' , inplace=True)


## Replace  # (GarageCond  ) ##
dataset_test['GarageCond'].fillna( 'none' , inplace=True)


## Replace  # (PoolQC    ) ##
dataset_test['PoolQC'].fillna( 'none' , inplace=True)


## Replace  # (Fence   ) ##
dataset_test['Fence'].fillna( 'none' , inplace=True)

## Replace  # (MiscFeature ) ##
dataset_test['MiscFeature'].fillna( 'none' , inplace=True)


## Replace  # (SaleType ) ##
dataset_test['SaleType'].fillna( 'WD' , inplace=True)




#Totoal Empty entries
dataset_test.isnull().sum().sum()


X_test = dataset_test.iloc[:, 0:80].values


# Encoding categorical data(test)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_test[:, 2] = labelencoder_X.fit_transform(X_test[:, 2]) #MSZoning
X_test[:, 5] = labelencoder_X.fit_transform(X_test[:, 5]) #street
X_test[:, 6] = labelencoder_X.fit_transform(X_test[:, 6]) #Alley
X_test[:, 7] = labelencoder_X.fit_transform(X_test[:, 7]) #LotShape
X_test[:, 8] = labelencoder_X.fit_transform(X_test[:, 8]) #LandContour
X_test[:, 9] = labelencoder_X.fit_transform(X_test[:, 9]) #Utilities
X_test[:, 10] = labelencoder_X.fit_transform(X_test[:, 10]) #LotConfig
X_test[:, 11] = labelencoder_X.fit_transform(X_test[:, 11]) #LandSlope
X_test[:, 12] = labelencoder_X.fit_transform(X_test[:, 12]) #Neighborhood
X_test[:, 13] = labelencoder_X.fit_transform(X_test[:, 13]) #Condition1
X_test[:, 14] = labelencoder_X.fit_transform(X_test[:, 14]) #Condition2
X_test[:, 15] = labelencoder_X.fit_transform(X_test[:, 15]) #BldgType
X_test[:, 16] = labelencoder_X.fit_transform(X_test[:, 16]) #HouseStyle
X_test[:, 21] = labelencoder_X.fit_transform(X_test[:, 21]) #RoofStyle
X_test[:, 22] = labelencoder_X.fit_transform(X_test[:, 22]) #RoofMatl
X_test[:, 23] = labelencoder_X.fit_transform(X_test[:, 23]) #Exterior1st
X_test[:, 24] = labelencoder_X.fit_transform(X_test[:, 24]) #Exterior2nd
X_test[:, 25] = labelencoder_X.fit_transform(X_test[:, 25]) #MasVnrType
X_test[:, 25] = labelencoder_X.fit_transform(X_test[:, 25]) #MasVnrType
X_test[:, 27] = labelencoder_X.fit_transform(X_test[:, 27]) #ExterQual
X_test[:, 28] = labelencoder_X.fit_transform(X_test[:, 28]) #ExterCond
X_test[:, 29] = labelencoder_X.fit_transform(X_test[:, 29]) #Foundation
X_test[:, 30] = labelencoder_X.fit_transform(X_test[:, 30]) #Bsmtqual
X_test[:, 31] = labelencoder_X.fit_transform(X_test[:, 31]) #BsmtCond
X_test[:, 32] = labelencoder_X.fit_transform(X_test[:, 32]) #BsmtExposure
X_test[:, 33] = labelencoder_X.fit_transform(X_test[:, 33]) #BsmtFinType1
X_test[:, 35] = labelencoder_X.fit_transform(X_test[:, 35]) #BsmtFinType2
X_test[:, 39] = labelencoder_X.fit_transform(X_test[:, 39]) #Heating
X_test[:, 40] = labelencoder_X.fit_transform(X_test[:, 40]) #HeatingQC
X_test[:, 41] = labelencoder_X.fit_transform(X_test[:, 41]) #CentralAir
X_test[:, 42] = labelencoder_X.fit_transform(X_test[:, 42]) #Electrical
X_test[:, 53] = labelencoder_X.fit_transform(X_test[:, 53]) #KitchenQual
X_test[:, 55] = labelencoder_X.fit_transform(X_test[:, 55]) #Functional
X_test[:, 57] = labelencoder_X.fit_transform(X_test[:, 57]) #FireplaceQu
X_test[:, 58] = labelencoder_X.fit_transform(X_test[:, 58]) #GarageType
X_test[:, 60] = labelencoder_X.fit_transform(X_test[:, 60]) #GarageFinish
X_test[:, 63] = labelencoder_X.fit_transform(X_test[:, 63]) #GarageQual
X_test[:, 64] = labelencoder_X.fit_transform(X_test[:, 64]) #GarageCond
X_test[:, 65] = labelencoder_X.fit_transform(X_test[:, 65]) #PavedDrive
X_test[:, 72] = labelencoder_X.fit_transform(X_test[:, 72]) #PoolQC
X_test[:, 73] = labelencoder_X.fit_transform(X_test[:, 73]) #Fence
X_test[:, 74] = labelencoder_X.fit_transform(X_test[:, 74]) #MiscFeature
X_test[:, 78] = labelencoder_X.fit_transform(X_test[:, 78]) #SaleType
X_test[:, 79] = labelencoder_X.fit_transform(X_test[:, 79]) #SaleCondition

X_test1 = X_test[:, 1:80]

# Feature Scaling
X_test1 = sc_X.fit_transform(X_test1)

# Predicting the Test set results
y_pred = regressor.predict(X_test1)
y_pred = sc_y.inverse_transform(y_pred)





submission = pd.DataFrame({
        "Id": dataset_test["Id"],
        "SalePrice": y_pred
    })

submission.to_csv('RandomForest2.csv', index=False)






















