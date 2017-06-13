Scripts for Kaggle Quora Duplicate Questions competition

https://www.kaggle.com/c/quora-question-pairs

Final ranking #79 with private leaderbord score of 0.14137

Most features were adopted from Kaggle kernels and discussions for this competition: 

https://www.kaggle.com/jturkewitz/magic-features-0-03-gain

https://www.kaggle.com/tour1st/magic-feature-v2-0-045-gain

https://www.kaggle.com/act444/lb-0-158-xgb-handcrafted-leaky

https://www.kaggle.com/selfishgene/shallow-benchmark-0-31675-lb

https://www.kaggle.com/skihikingkevin/magic-feature-v2-krzy-new-idea

https://www.kaggle.com/c/quora-question-pairs/discussion/32313

https://www.kaggle.com/c/quora-question-pairs/discussion/33371

https://www.kaggle.com/c/quora-question-pairs/discussion/31284

Other features were contributed by teammates

https://www.kaggle.com/stys

https://www.kaggle.com/avideret

https://www.kaggle.com/aelphy

https://www.kaggle.com/velika12

Abhishek's features and some teammates' features were provided as CSV files only,
so these features are not available here.


```
# Compute features
PYTHONPATH='.' python features/counters.py
PYTHONPATH='.' python features/linear.py
etc..

# Train XGBoost and create submission
PYTHONPATH='.' python models/xgb.py --conf=configs/baseline20.conf

```

