# Model Card

This a model that looks at different poulation features and tries to predict whether they make +50K or less than 50K a year.
The model looks at different categorical features like: "workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country". and it also looks at some numerical.

## Model Details

This is a linear regression model that was trained on the above features.
the model.

Model Metrics:
Overall Accuracy: 0.79
Overall Precision: 0.74
Overall Recall: 0.27
Overall F1-Score: 0.39

## Intended Use

in the model repo, there's an api running on heroku that allows users to pass 15 features about the user.

## Training Data

the original dataset is of shape: (32561, 15)
the training set had a shape of: (26048, 15)

## Evaluation Data

For validation we used a set of shape: (6513, 15)

## Metrics

We're looking at 4 different metrics:
accuracy
precision
recall
fbeta

## Ethical Considerations

## Caveats and Recommendations

Some recommendations:

- make the dataset more diverse and inclusive to reduce bias
- include more feature to improve the model accuracy and improve the model recall rates.

the model showed some baises towards so data slices, below I'll list down the feature slices with their accuracy metrics:

accuracy for feature workclass with value State-gov: 0.7686567164179104
accuracy for feature workclass with value Local-gov: 0.7596371882086168
accuracy for feature workclass with value Federal-gov: 0.675531914893617
accuracy for feature workclass with value Private: 0.8099977880999779
accuracy for feature workclass with value Self-emp-not-inc: 0.7901960784313725
accuracy for feature workclass with value ?: 0.8913043478260869
accuracy for feature workclass with value Self-emp-inc: 0.6121495327102804
accuracy for feature workclass with value Never-worked: 1.0
accuracy for feature workclass with value Without-pay: 1.0
accuracy for feature education with value Bachelors: 0.6808712121212122
accuracy for feature education with value Masters: 0.599483204134367
accuracy for feature education with value Assoc-voc: 0.749034749034749
accuracy for feature education with value HS-grad: 0.8352831071595694
accuracy for feature education with value 7th-8th: 0.9262295081967213
accuracy for feature education with value 11th: 0.9527896995708155
accuracy for feature education with value Some-college: 0.8376487053883834
accuracy for feature education with value 9th: 0.940677966101695
accuracy for feature education with value Preschool: 0.9166666666666666
accuracy for feature education with value Assoc-acdm: 0.82
accuracy for feature education with value 1st-4th: 0.9444444444444444
accuracy for feature education with value 10th: 0.9171270718232044
accuracy for feature education with value 12th: 0.9186046511627907
accuracy for feature education with value Doctorate: 0.4691358024691358
accuracy for feature education with value 5th-6th: 0.9206349206349206
accuracy for feature education with value Prof-school: 0.6371681415929203
accuracy for feature marital-status with value Never-married: 0.9477434679334917
accuracy for feature marital-status with value Widowed: 0.9162790697674419
accuracy for feature marital-status with value Married-civ-spouse: 0.6412719443524346
accuracy for feature marital-status with value Divorced: 0.8997695852534562
accuracy for feature marital-status with value Separated: 0.9360730593607306
accuracy for feature marital-status with value Married-spouse-absent: 0.9523809523809523
accuracy for feature marital-status with value Married-AF-spouse: 0.3333333333333333
accuracy for feature occupation with value Adm-clerical: 0.8649025069637883
accuracy for feature occupation with value Prof-specialty: 0.6596244131455399
accuracy for feature occupation with value Exec-managerial: 0.6397590361445783
accuracy for feature occupation with value Craft-repair: 0.7982779827798278
accuracy for feature occupation with value Sales: 0.7815344603381015
accuracy for feature occupation with value ?: 0.8915989159891599
accuracy for feature occupation with value Other-service: 0.9448698315467075
accuracy for feature occupation with value Handlers-cleaners: 0.933852140077821
accuracy for feature occupation with value Machine-op-inspct: 0.8983451536643026
accuracy for feature occupation with value Tech-support: 0.7461139896373057
accuracy for feature occupation with value Farming-fishing: 0.8977272727272727
accuracy for feature occupation with value Protective-serv: 0.6870229007633588
accuracy for feature occupation with value Transport-moving: 0.8178807947019867
accuracy for feature occupation with value Priv-house-serv: 1.0
accuracy for feature occupation with value Armed-Forces: 0.5
accuracy for feature relationship with value Not-in-family: 0.903951367781155
accuracy for feature relationship with value Unmarried: 0.9381294964028777
accuracy for feature relationship with value Husband: 0.6474118529632408
accuracy for feature relationship with value Other-relative: 0.9459459459459459
accuracy for feature relationship with value Own-child: 0.96751968503937
accuracy for feature relationship with value Wife: 0.5620915032679739
accuracy for feature race with value White: 0.7864826102545716
accuracy for feature race with value Asian-Pac-Islander: 0.7525252525252525
accuracy for feature race with value Black: 0.9032258064516129
accuracy for feature race with value Other: 0.875
accuracy for feature race with value Amer-Indian-Eskimo: 0.819672131147541
accuracy for feature sex with value Female: 0.8863530507685142
accuracy for feature sex with value Male: 0.7540082455336693
