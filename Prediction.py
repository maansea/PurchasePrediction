#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ctypes import pointer
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def LabelEncoder(input_value, feats):
    feat_value=list(1+np.arange(len(feats)))
    feat_key=feats
    feat_dict=dict(zip(feat_key,feat_value))
    print(feat_dict)
    print(input_value)
    value=feat_dict[input_value]
    return value


def get_prediction(data,model):
    """
     predict the class of given data pointe
    """
    return model.predict(data)
    print(model.predict(data))


# In[ ]:




