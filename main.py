
"""
https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic.html#sphx-glr-auto-examples-linear-model-plot-logistic-py

"""

from sklearn.preprocessing import OrdinalEncoder, Normalizer, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt, csv

class PSWorkshopQualifiers(object):
    
    def __init__(self, score, train):

        # Object-oriented programming variable initialization

        self.scoreset, self.trainset = pd.read_csv(score), pd.read_csv(train)
        self.features, self.target = [], []

    def prepareData(self, ordEnc = OrdinalEncoder()):

        # Quantifying categorical variables
    
        self.scoreset["Gender"] = ordEnc.fit_transform(self.scoreset[["Gender"]])
        self.scoreset["hadVehicleClaimInPast"] = ordEnc.fit_transform(self.scoreset[["hadVehicleClaimInPast"]])
        self.scoreset["hasMortgage"] = ordEnc.fit_transform(self.scoreset[["hasMortgage"]])
        self.scoreset["vehicleStatus"] = ordEnc.fit_transform(self.scoreset[["vehicleStatus"]])
        self.trainset["Gender"] = ordEnc.fit_transform(self.trainset[["Gender"]])
        self.trainset["hadVehicleClaimInPast"] = ordEnc.fit_transform(self.trainset[["hadVehicleClaimInPast"]])
        self.trainset["hasMortgage"] = ordEnc.fit_transform(self.trainset[["hasMortgage"]])
        self.trainset["vehicleStatus"] = ordEnc.fit_transform(self.trainset[["vehicleStatus"]])
        print(self.scoreset)
        print(self.trainset)

        # Normalizing all quantitative variables

        """
        
        Write a for loop iterating through variables and use Normalize().fit_transform(i) function 

        """

        return self.scoreset, self.trainset

    def unpackData(self):

        # Creating training and test sets*

        for i in range(len(self.trainset)):
            tempFeatures = [self.trainset.iloc[i][0]]
            self.target.append = self.trainset.iloc[i][1]
            for j in range(2, 15):
                tempFeatures.append(self.trainset.iloc[0][j])
            self.features.append(tempFeatures)

        return self.features, self.target

    def modelData(self, c = 1e5):

        # Applying linear and logistic regression the various sets*

        X_train, X_test, Y_train, Y_test = train_test_split(self.features, self.target, test_size = 0.25, random_state = None)
        clf = LogisticRegression(c)
        clf.fit(X_train, Y_train)
        ols = LinearRegression()
        ols.fit(X_train, Y_train)
        pass

    def convertData(self):

        # Write predictions into .csv file

        with open("", 'w') as file:
            writer = csv.writer(file)
        pass

client = PSWorkshopQualifiers(r"C:\Users\JC\OneDrive\Documents\University - Level II\mathage\ScoringDataset_2023Qualification.csv", 
                            r"C:\Users\JC\OneDrive\Documents\University - Level II\mathage\TrainingDataset_2023Qualification.csv")
client.prepareData()


