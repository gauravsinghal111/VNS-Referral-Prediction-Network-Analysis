# Model evaluation (time-based hold-out)

Train: 2013 features -> 2014 label. Test: 2014 features -> 2015 label. All numbers below are from the test year only.

- Neurologists in train: 671 (24.1% positive)
- Neurologists in test: 701 (13.8% positive) - so a random list of 100 would contain about 14 future referrers

## All active neurologists

| Model                                     |   Precision@100 |   Recall@100 |   ROC-AUC |   PR-AUC |   Brier |
|:------------------------------------------|----------------:|-------------:|----------:|---------:|--------:|
| Baseline: rank by patient volume          |           0.340 |        0.351 |     0.788 |    0.357 | nan     |
| Baseline: rank by current ties to targets |           0.680 |        0.701 |     0.923 |    0.761 | nan     |
| Logistic regression                       |           0.650 |        0.670 |     0.925 |    0.728 |   0.081 |
| XGBoost                                   |           0.660 |        0.680 |     0.917 |    0.682 |   0.085 |

## New-referrer opportunity: neurologists with no tie to a target in 2014 (n = 538, 15 positives)

| Model                            |   Positives found in top 50 |   ROC-AUC |   PR-AUC | PR-AUC 95% CI   |
|:---------------------------------|----------------------------:|----------:|---------:|:----------------|
| Baseline: rank by patient volume |                           4 |     0.732 |    0.091 | 0.03 - 0.26     |
| Logistic regression              |                           4 |     0.777 |    0.088 | 0.04 - 0.20     |
| XGBoost                          |                           6 |     0.727 |    0.196 | 0.07 - 0.47     |
