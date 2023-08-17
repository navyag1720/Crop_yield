import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split




st.set_option('deprecation.showPyplotGlobalUse', False)


data = pd.read_csv('Crop_recommendation.csv')


st.sidebar.title("Crop Yield Prediction")


st.sidebar.markdown("Explore and analyze crop yield data")





def display_crops_by_condition(selected_crops,condition, threshold, comparison_type,avg_comparison_type=None):
    if comparison_type == "Greater than":
        st.write(f"Crops which require a very high ratio of {condition} content in soil:")
        selected_crops = data[data[condition] > threshold]['label'].unique()
    elif comparison_type == "Less than":
        st.write(f"Crops which require a very low ratio of {condition} content in soil:")
        selected_crops = data[data[condition] < threshold]['label'].unique()
    elif comparison_type == "Equal to average":
        st.write(f"Crops with equal to average {condition} content in soil:")
        selected_crops = data[data[condition] == threshold]['label'].unique()


    st.table(pd.DataFrame({"Crops": selected_crops}))






condition_options = ['Select a condition', 'N', 'P', 'K', 'rainfall', 'temperature', 'humidity', 'ph']
condition = st.sidebar.selectbox("Select Condition", condition_options, index=0)
selected_crops = [] 

if condition != 'Select a condition':
    min_threshold = int(data[condition].min())
    max_threshold = int(data[condition].max())
    
    threshold = st.sidebar.slider("Select Threshold Value", min_value=min_threshold, max_value=max_threshold)
else:
    threshold = None

comparison_type_options = ['Greater than', 'Less than', 'Equal to average']
comparison_type = st.sidebar.radio("Select Comparison Type", comparison_type_options)

if condition != 'Select a condition' and threshold and comparison_type:
    result_placeholder = st.empty()
    result_placeholder.text("Loading... Please wait.")
    display_crops_by_condition(selected_crops,condition, threshold, comparison_type)

    result_placeholder.empty()









def summary(crops=None):
    if crops is None:
        crops = list(data['label'].value_counts().index)

    st.write(f"Summary for {crops} - Minimum, Average, and Maximum")
    st.write("--------------------------------------------------")

    
    x = data[data['label'].isin(crops)]

    if x.empty:
        st.write("No data available for the selected crops.")
    else:
        z = x.drop(['label'], axis=1)

        
        summary_data = pd.DataFrame({
            'Crop': crops
        })

        for i in z:
            summary_data[f'Minimum {i}'] = x[i].min()
            summary_data[f'Average {i}'] = x[i].mean()
            summary_data[f'Maximum {i}'] = x[i].max()

        
        st.table(summary_data)

if 'selected_crops' not in st.session_state:
    st.session_state.selected_crops = list(data['label'].value_counts().index)

st.subheader("MIN,MAX,AVG FOR SELECTED CROP")
selected_crops_placeholder = st.empty()
selected_crops = st.multiselect("Select Crops", data['label'].value_counts().index, st.session_state.selected_crops)
y = data['label']
x = data.drop(['label'], axis=1)

if st.button("Select Crops"):
    st.session_state.selected_crops = selected_crops


selected_crops_placeholder.write(f"Selected Crops: {st.session_state.selected_crops}")
summary(st.session_state.selected_crops)

st.sidebar.subheader("Model Selection")
selected_model = st.sidebar.selectbox("Select Model", ['Logistic Regression', 'Random Forest'])

if selected_model == 'Logistic Regression':
    model = LogisticRegression()
else:  
    model = RandomForestClassifier()

model.fit(x, y)



y_predict = model.predict(x)







st.header("Crop Prediction")
st.write("Enter the climatic conditions below and click the 'Predict' button to get the suggested crop.")


n = st.slider("Nitrogen (N)", min_value=0, max_value=100, value=50)
p = st.slider("Phosphorus (P)", min_value=0, max_value=100, value=50)
k = st.slider("Potassium (K)", min_value=0, max_value=100, value=50)
temperature = st.slider("Temperature", min_value=0, max_value=100, value=25)
humidity = st.slider("Humidity", min_value=0, max_value=100, value=60)
ph = st.slider("pH", min_value=0, max_value=14, value=7)
rainfall = st.slider("Rainfall", min_value=0, max_value=400, value=200)


if st.button("Predict"):
    prediction = model.predict([[n, p, k, temperature, humidity, ph, rainfall]])
    st.write("The suggested crop for the given climatic condition is:", prediction[0])
