import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

dic = {'alcohol':'ALCEVER', 'marijuana':'MJEVER', 'opioid':'OPEVER', 'nicotine':'NICEVR'}

for i in ['alcohol', 'marijuana', 'opioid', 'nicotine']:
    filename = '/Users/andyliu/Documents/LR_' + i + '.csv'

    df = pd.read_csv(filename, header=0)
    df = df.dropna()

    X = df.loc[:, df.columns != dic[i]]
    y = df.loc[:, df.columns == dic[i]].astype(np.int16)

    os = SMOTE(random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    columns = X_train.columns
    os_data_X,os_data_y=os.fit_sample(X_train, y_train.values.ravel())
    os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
    os_data_y= pd.DataFrame(data=os_data_y,columns=[dic[i]])
    
    logreg = LogisticRegression(solver='lbfgs')
    rfe = RFE(logreg, 15)
    rfe = rfe.fit(os_data_X,os_data_y.values.ravel())
    print(i + " results:")
    rfe_approved = (rfe.support_).tolist()
    rfe_indices = []
    for i in range(0, len(rfe_approved)):
        if rfe_approved[i]:
            rfe_indices.append(i)
            
    X = os_data_X.iloc[:,rfe_indices]
    y = os_data_y

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) #this line breaks
    logreg = LogisticRegression(solver='lbfgs')
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
    print(classification_report(y_test, y_pred))

