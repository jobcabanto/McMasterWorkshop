
"""
https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic.html#sphx-glr-auto-examples-linear-model-plot-logistic-py

"""

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd, csv

class PSWorkshopQualifiers(object):
    
    def __init__(self, score, train):

        # Object-oriented programming variable initialization

        self.scoreset, self.trainset = pd.read_csv(score), pd.read_csv(train)
        self.features, self.target, self.scoringFeatures = [], [], []

    def prepareTrainSet(self, ordEnc = OrdinalEncoder()):

        # Quantifying categorical variables
    
        self.trainset["Gender"] = ordEnc.fit_transform(self.trainset[["Gender"]])
        self.trainset["hadVehicleClaimInPast"] = ordEnc.fit_transform(self.trainset[["hadVehicleClaimInPast"]])
        self.trainset["vehicleStatus"] = ordEnc.fit_transform(self.trainset[["vehicleStatus"]])
        self.scoreset["Gender"] = ordEnc.fit_transform(self.scoreset[["Gender"]])
        self.scoreset["hadVehicleClaimInPast"] = ordEnc.fit_transform(self.scoreset[["hadVehicleClaimInPast"]])
        self.scoreset["vehicleStatus"] = ordEnc.fit_transform(self.scoreset[["vehicleStatus"]])

        # Normalizing all quantitative variables

        self.trainset = pd.DataFrame(MinMaxScaler().fit_transform(self.trainset.values))
        self.scoreset = pd.DataFrame(MinMaxScaler().fit_transform(self.scoreset.values))

        return self.scoreset, self.trainset

    def unpackData(self):

        # Extracting from Training Set

        for i in range(len(self.trainset)):
            tempFeatures = [self.trainset.iloc[i][0]]
            self.target.append(self.trainset.iloc[i][1])
            for j in range(2, 14):
                tempFeatures.append(self.trainset.iloc[0][j])
            self.features.append(tempFeatures)

        # Extracting from Scoring Set

        for i in range(len(self.scoreset)):
            tempFeatures = [self.scoreset.iloc[i][0]]
            for j in range(1, 13):
                tempFeatures.append(self.scoreset.iloc[0][j])
            self.scoringFeatures.append(tempFeatures)

        return self.features, self.target, self.scoringFeatures

    def modelData(self):

        # Applying logistic regression to various sets

        X_train, X_test, Y_train, Y_test = train_test_split(self.features, self.target, test_size = 0.20, random_state = None)
        model = XGBClassifier(objective ='binary:logistic', random_state = None)
        model.fit(X_train, Y_train)
        print(classification_report(Y_test, model.predict(X_test)))
        print(model.predict(self.scoringFeatures))
        
    def convertData(self):

        # Write predictions into .csv file*

        with open("", 'w') as file:
            writer = csv.writer(file)
        pass

client = PSWorkshopQualifiers(r"", 
                            r"")
client.prepareTrainSet()
client.unpackData()
client.modelData()
