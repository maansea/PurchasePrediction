import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from Prediction import get_prediction, LabelEncoder
from load_model import get_model


rf_model = get_model(model_path = r'Purchase_Prediction.pkl')

st.set_page_config(page_title="Customer Purchase Prediction App for an E-Commerce Setup",
                   page_icon="🚧", layout="wide")

#creating option list for dropdown menu
options_Operating_Systems = ['OS_1', 'OS_2', 'OS_3', 'OS_4', 'OS_5', 'OS_6','OS_7', 'OS_8']
options_Browser = ['Browser_1', 'Browser_2', 'Browser_3', 'Browser_4', 'Browser_5', 'Browser_6', 'Browser_7', 'Browser_8',
                  'Browser_9', 'Browser_10', 'Browser_11', 'Browser_12', 'Browser_13']
options_Region = ['Region1', 'Region2', 'Region3', 'Region4', 'Region5', 'Region6', 'Region7', 'Region8', 'Region9']
options_Traffic_Type = ['TrafficType1', 'TrafficType2', 'TrafficType3', 'TrafficType4', 'TrafficType5', 'TrafficType6', 'TrafficType7'
                       'TrafficType8', 'TrafficType9', 'TrafficType10', 'TrafficType11', 'TrafficType12', 'TrafficType13']
options_Visitor_Type = ['Returning_Visitor', 'New_Visitor', 'Other']
options_Weekend = ['True', 'False']
options_Month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', ]



#creating sliders

options_Administrative = list(range(0,30))
options_Administrative_Duration = list(np.arange(0, 3000, 0.1))
options_Informational = list(range(0,20))
options_Informational_Duration = list(np.arange(0,2000, 0.2))
options_Product_Related = list(range(0,600))
options_Product_Related_Duration = list(np.arange(0,30000, 0.1))
options_Bounce_Rate = list(np.arange(0.0,0.2, 0.0001))
options_Exit_Rate = list(np.arange(0.0,0.2, 0.0001))
options_Page_Values = list(np.arange(0.0,400.0, 0.001))
options_Special_Days = list(np.arange(0.0, 1.0, 0.2))



st.markdown("<h1 style='text-align: center;'>Customer Purchase Prediction App for an E-Commerce Setup 🚧</h1>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):
      
        st.subheader("Enter the input for following features:")
        Administrative = st.slider("Select Administrattive: ", 0, 30, value = 3, format = "%d")
        Administrative_duration = st.slider("Select Administrative Duration: ", 0, 3000, value = 32, format = "%d")
        Informational = st.slider("Select Informational: ", 0, 30, value = 7, format = "%d")
        Informational_duration = st.slider("Select Informational Duration: ", 0, 2000, value= 125, format = "%d")
        Product_Related = st.slider("Select Product Related: ", 0, 600, value = 27, format = "%d")
        Product_Related_Duration = st.slider("Select  Product Related Duration: ", 0, 30000, value = 603, format = "%d")
        Bounce_Rate = st.slider("Select Bounce Rate: ", 0.0, 0.2, value = 0.0003, format = "%d" )
        Exit_Rate = st.slider("Select Exit Rate: ", 0.0, 0.2, value = 0.00043, format = "%d" )
        Page_Values = st.slider("Select Page Values: ", 0, 400, value = 51.7563, format = "%d")
        Special_Days = st.slider("Select Special days: ", 0.0, 1.0, value = 0.4, format = "%d")
        Operating_Systems = st.selectbox("Select the Operating System: ", options=options_Operating_Systems) 
        Browser = st.selectbox("Select the Browser: ", options=options_Browser)
        Region = st.selectbox("Select the Region: ", options_Region)
        Traffic_Type = st.selectbox("Select the Type of Traffic: ", options_Traffic_Type)
        Visitor_Type = st.selectbox("Select the Type of Traffic: ", options_Visitor_Type)
        Weekend = st.selectbox("Select True if it was a weekend or false if not: ", options_Weekend)
        Month = st.selectbox("Select the month: ", options_Month)
        
        
        submit = st.form_submit_button("Predict")
        
        
    if submit:
         Administrative =  LabelEncoder(Administrative, options_Administrative)
         Administrative_duration = LabelEncoder(Administrative_duration, options_Administrative_Duration) 
         Informational = LabelEncoder(Informational, options_Informational)
         Informational_duration =  LabelEncoder(Informational_duration, options_Informational_Duration)
         Product_Related = LabelEncoder(Product_Related, options_Product_Related)
         Product_Related_Duration = LabelEncoder(Product_Related_Duration, options_Product_Related_Duration) 
         Exit_Rate = LabelEncoder(Exit_Rate, options_Exit_Rate)
         Bounce_Rate = LabelEncoder(Bounce_Rate, options_Bounce_Rate)
         Page_Values = LabelEncoder(Page_Values, options_Page_Values) 
         Special_Days = LabelEncoder(Special_Days, options_Special_Days)
         Operating_Systems = LabelEncoder(Operating_Systems, options_Operating_Systems)
         Browser = LabelEncoder(Browser, options_Browser) 
         Region = LabelEncoder(Region, options_Region)
         Traffic_Type = LabelEncoder(Traffic_Type, options_Traffic_Type)
         Visitor_Type = LabelEncoder(Visitor_Type, options_Visitor_Type)
         Weekend = LabelEncoder(Weekend, options_Weekend)
         Month = LabelEncoder(Month, options_Month)
            
            
         data = np.array([Administrative,Administrative_duration, Informational, Informational_duration, Product_Related,
                            Product_Related_Duration, Exit_Rate, Bounce_Rate, Page_Values, Special_Days, Operating_Systems, 
                            Browser, Region, Traffic_Type, Visitor_Type, Weekend, Month ]).reshape(-1,1)
         pred = get_prediction(data = data, model = rf_model)
         if pred[0] == 0:
          pred = "Consumer will not buy"
         else:
          pred = "Consumer will buy"
                
            
            
         st.write(f"The customer purchase intent prediction :  {pred}")
  

if __name__ == '__main__':
    main()
        
   
        
        
        
        


