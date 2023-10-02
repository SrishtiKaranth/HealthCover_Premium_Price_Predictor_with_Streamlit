import numpy as np
import pickle 
import streamlit as st
import requests
import json
import streamlit_lottie
from streamlit_lottie import st_lottie

loaded_model=pickle.load(open('trained_model_xgb.sav','rb'))
loaded_model2=pickle.load(open('trained_model_rf.sav','rb'))
loaded_scaler = pickle.load(open("scaler.pkl", 'rb'))

def predict_premium_price(single_obs, loaded_model_xgb, loaded_model_rf, loaded_scaler):
    single_obs = np.asarray(single_obs)
    single_obs_reshaped = single_obs.reshape(1, -1)
    
    standardized_input = loaded_scaler.transform(single_obs_reshaped)
    
    pred_xgb = loaded_model_xgb.predict(standardized_input)
    predicted_pricex = pred_xgb[0]
    
    pred_rf = loaded_model_rf.predict(standardized_input)
    predicted_price = pred_rf[0]
    
    average_price = (predicted_price + predicted_pricex) / 2
    
    return predicted_price, predicted_pricex, average_price


def main():

    st.set_page_config(page_title="HealthCover || Premium Price Prediction",page_icon="💰",layout="wide")
    with st.container():
        st.write('#') 
        st.write('#') 
        st.write('#') 
        left_column,right_column=st.columns([1,1])
        with right_column:


            st.title("HealthCover Premium Predictor")
            st.subheader("Your Health, Your Price: Unlocking the Mystery of Medical Insurance Premiums💸")
            st.markdown(
                        """
                        <div style="text-align: justify;color:grey;">
                        Navigating the complex world of medical insurance can be challenging, with premiums varying widely based on factors such as age, medical history, and lifestyle. With HealthCover Premium Predictor, you gain the power to make informed decisions. Our models leverage cutting-edge data analysis and machine learning techniques to provide you with accurate premium estimates, ensuring you get the coverage you need at a price that suits your budget.
                        Simply input your health details, and let our algorithms do the rest. We're committed to bringing transparency and affordability to healthcare, one prediction at a time.
                        </div>
                        """,
                        unsafe_allow_html=True,
                                )
            st.write('#')
            st.write("---")
            st.write('#')
        
        with left_column:
            def load_lottiefile(filepath:str):
                with open(filepath,"r") as f:
                    return json.load(f)  # Corrected indentation
            lottie_coding=load_lottiefile("animation_ln99l4tt.json")  # Corrected indentation
            st_lottie(
                lottie_coding,
                speed=1,
                reverse=False,
                loop=True,
                quality="high",
                # renderer="svg",
                height="450px",
                width="600px",
                key=None,
                )
            st.write('#')  # This line should be outside of the with block
            st.write('#') 
    with st.container():
        st.write('#')
        st.title("HealthCover Premium price prediction model 🪙")
        st.write("Enter the following details to help us predict your medical insurance premium cost.")
        st.write("##")
        st.write("##")

            #getting input data from user

        age=st.text_input('Enter Age')
        diabetes=st.text_input('Does customer have Diabetes? 1/0(y/n)')
        bloodPressureProblems=st.text_input('Does customer have BloodPressure Problems? 1/0(y/n)')
        anyTransplants=st.text_input('Has customer had any organ transplants? 1/0(y/n)')
        anyChronicDiseases=st.text_input('Does customer have any chronic disease? 1/0(y/n)')
        height=st.text_input('Enter Height')
        weight=st.text_input('Enter Weight')
        knownAllergies	=st.text_input('Does customer have any known allergies? 1/0(y/n)')
        historyOfCancerInFamily=st.text_input('Does customers family have any history of Cancer? 1/0(y/n)')
        numberOfMajorSurgeries=st.text_input('Enter number of major surgeries undergone')

        prices=''

        if st.button("Predict Results"):
            prices=predict_premium_price([age,diabetes,bloodPressureProblems,anyTransplants,anyChronicDiseases,height,weight,knownAllergies,historyOfCancerInFamily,numberOfMajorSurgeries],loaded_model,loaded_model2,loaded_scaler)

            predicted_price, predicted_pricex, average_price = prices  # Unpack the returned values

            st.success(f"The first Premium price prediction is: {predicted_price:.2f}")
            st.success(f"The second Premium price prediction is: {predicted_pricex:.2f}")
            st.success(f"The Average Premium price predicted is: {average_price:.2f}")


        
        st.write("This model uses different Regression models such as RandomForestRegressor, XGBoostRegressor and has an accuracy over 85%.")
        st.write("#")
        st.write("---")


if __name__ == '__main__':
    main()