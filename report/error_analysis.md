# Error analysis (XGBoost, 2015 hold-out)

## Top drivers (mean |SHAP|)

| Feature                                            |   Mean |SHAP| |
|:---------------------------------------------------|--------------:|
| Patients sent to other doctors                     |         0.838 |
| Patients sent to epilepsy/neurosurgery             |         0.836 |
| Already shares patients with epilepsy/neurosurgery |         0.777 |
| Patients received from other doctors               |         0.304 |
| Years since NPI issued                             |         0.263 |
| Total patients shared (all doctors)                |         0.227 |
| Neuro network: PageRank (influence)                |         0.206 |
| Epilepsy/neurosurgery specialists nearby (ZIP3)    |         0.190 |
| Number of distinct partner doctors                 |         0.183 |
| Neuro network: outgoing ties                       |         0.174 |
| Neuro network: incoming ties                       |         0.111 |
| Neuro network: betweenness (bridge role)           |         0.088 |
| Practises in a major metro                         |         0.061 |
| General neurologist                                |         0.010 |
| Child neurologist                                  |         0.000 |

## Who the model gets right and wrong (medians)

| Median value                                       |   True positives in top 100 (n=66) |   False positives in top 100 (n=34) |   Missed referrers outside top 100 (n=31) |
|:---------------------------------------------------|-----------------------------------:|------------------------------------:|------------------------------------------:|
| Total patients shared (all doctors)                |                          4196.5000 |                           4102.0000 |                                 2988.0000 |
| Number of distinct partner doctors                 |                            89.5000 |                             94.0000 |                                   69.0000 |
| Already shares patients with epilepsy/neurosurgery |                             2.0000 |                              1.0000 |                                    1.0000 |
| Epilepsy/neurosurgery specialists nearby (ZIP3)    |                            43.0000 |                             81.0000 |                                   74.0000 |
| Neuro network: PageRank (influence)                |                             0.0024 |                              0.0020 |                                    0.0013 |
| Practises in a major metro                         |                             1.0000 |                              1.0000 |                                    1.0000 |

## Performance by specialty

| specialty                       |   Neurologists |   Positive rate |   ROC-AUC |   PR-AUC |
|:--------------------------------|---------------:|----------------:|----------:|---------:|
| Neurology                       |            629 |           0.126 |     0.915 |    0.640 |
| Clinical Neurophysiology        |             23 |           0.217 |     1.000 |    1.000 |
| Neurocritical Care              |             14 |           0.286 |     0.925 |    0.893 |
| Vascular Neurology              |             13 |           0.385 |     0.900 |    0.810 |
| Neuromuscular Medicine          |             10 |           0.100 |     1.000 |    1.000 |
| Child Neurology                 |              7 |           0.143 |     1.000 |    1.000 |
| Behavioral Neurology            |              4 |           0.500 |     1.000 |    1.000 |
| Neurodevelopmental Disabilities |              1 |           0.000 |   nan     |  nan     |

## Performance by metro vs non-metro

| metro             |   Neurologists |   Positive rate |   ROC-AUC |   PR-AUC |
|:------------------|---------------:|----------------:|----------:|---------:|
| Houston           |            253 |           0.083 |     0.917 |    0.677 |
| Dallas-Fort Worth |            192 |           0.141 |     0.932 |    0.677 |
| Non-metro         |            116 |           0.259 |     0.915 |    0.754 |
| Austin            |             68 |           0.147 |     0.829 |    0.534 |
| San Antonio       |             60 |           0.117 |     0.876 |    0.708 |
| El Paso           |             12 |           0.167 |     1.000 |    1.000 |
