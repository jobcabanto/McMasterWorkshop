
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd

class PSWorkshopQualifiers(object):
    
    def __init__(self, score, train, submit):

        # Object-oriented programming variable initialization

        self.scoreset, self.trainset, self.submit = pd.read_csv(score), pd.read_csv(train), pd.read_csv(submit)
        self.features, self.target, self.scoringFeatures, self.predictions = [], [], [], []

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

    def unpackData(self, tempFeatures = []):

        # Extracting from Training Set

        for i in range(len(self.trainset)):
            tempFeatures = []
            self.target.append(self.trainset.loc[i][1])
            for j in range(2, 15):
                tempFeatures.append(self.trainset.loc[i][j])
            self.features.append(tempFeatures)

            # Oversampling method

            if self.target[-1] == 1 or self.target[-1] == "1":
                for k in range(6):
                    self.features.append(tempFeatures)
                    self.target.append(self.trainset.loc[i][1])

        # Extracting from Scoring Set

        for i in range(len(self.scoreset)):
            tempFeatures = []
            for j in range(1, 13):
                tempFeatures.append(self.trainset.loc[i][j])
            self.scoringFeatures.append(tempFeatures)

        return self.features, self.target, self.scoringFeatures

    def modelData(self):

        # Applying logistic regression the various sets

        X_train, X_test, Y_train, Y_test = train_test_split(self.features, self.target, test_size = 0.20, random_state = 36, shuffle = True)
        model = XGBClassifier(objective = 'binary:logistic', n_estimators = 1000, random_state = 36).fit(X_train, Y_train)
        # print(classification_report(Y_test, model.predict(X_test)))
        self.predictions = model.predict(self.scoringFeatures)
        # print(self.predictions[0:15])
        # print(len(self.predictions))
        
    def sendData(self):

        # Write predictions into .csv file
        for i in range(len(self.predictions)):
            self.submit.loc[i, "predictedResponseVariable"] = self.predictions[i]
        self.submit.to_csv("SubmissionExample_2023Qualification.csv", index=False)

client = PSWorkshopQualifiers(r"C:\Users\JC\OneDrive\Documents\University - Level II\mathage\ScoringDataset_2023Qualification.csv", 
                            r"C:\Users\JC\OneDrive\Documents\University - Level II\mathage\TrainingDataset_2023Qualification.csv",
                            r"C:\Users\JC\OneDrive\Documents\University - Level II\mathage\SubmissionExample_2023Qualification.csv")
client.prepareTrainSet()
client.unpackData()
client.modelData()
client.sendData()

