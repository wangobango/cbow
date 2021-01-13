import pandas as pd

class SetReader:

    def __init__(self):
        self.trainPath = './train'
        self.testPath = './test-A'
        self.dev0Path = './dev-0'
        self.dev1Path = './dev-1'
        self.pandas = pd

    def readTrainSet(self):
        return pd.read_csv(self.trainPath + '/train.tsv', sep = '\t', names=['dateFrom', 'dateTo', 'lead', 'code', 'data'])

    def readDev0Train(self):
        return pd.read_csv(self.dev0Path + '/in.tsv', sep = '\t', names=['dateFrom', 'dateTo', 'data'])

    def readDev0Expected(self):
        return pd.read_csv(self.dev0Path + '/expected.tsv', error_bad_lines=False, engine="python")