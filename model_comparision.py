import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

#Load Dataset
data=load_breast_cancer(as_frame=True)
df=data.frame
df.head()

#Split features and target
X=df.drop('target',axis=1)
y=df['target']

# Train-test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#bulid RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
rf_pred=rf.predict(X_test)
print("accuracy score:",accuracy_score(y_test,rf_pred))
print("classification report:",classification_report(y_test,rf_pred))

#build XGBClassifier
xgb=XGBClassifier()
xgb_fit=xgb.fit(X_train,y_train)
xgb_pred=xgb.predict(X_test)
print("accuracy score:",accuracy_score(y_test,xgb_pred))
print("classification report:",classification_report(y_test,xgb_pred))

#bulid lightbgm
lgbm=LGBMClassifier()
lgbm_fit=lgbm.fit(X_train,y_train)
lgbm_pred=lgbm.predict(X_test)
print("accuracy score:",accuracy_score(y_test,lgbm_pred))
print("classification report:",classification_report(y_test,lgbm_pred))

#Result Comparision
comparision=pd.DataFrame({
    "model":[
        "Random Forest","XGBOOST","LightBGM"
    ],
    "accuracy":[
        accuracy_score(y_test,rf_pred),
        accuracy_score(y_test,xgb_pred),
        accuracy_score(y_test,lgbm_pred)

    ]
})
comparision