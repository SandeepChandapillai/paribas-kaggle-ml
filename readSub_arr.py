import pandas as pd

submission = pd.read_csv('submission.csv')
preds = submission["PredictedProb"]

print(preds)

