import streamlit as st
import joblib
import pandas as pd

st.title('🚢Titanic survival Pridiction App')

#Load model and encoders
model = joblib.load('best_model.pkl')
LEC_Embarked = joblib.load('LEC_Embarked.pkl')
LEC_Sex = joblib.load('LEC_Sex.pkl')

# User input field
pclass = st.selectbox('Passengerd Class', [1,2,3])
sex = st.selectbox('Sex',['male','female'])
age = st.number_input('Age', 0, 100, 30)
sibsp = st.number_input('Siblings/Spouses Abroad', 0, 10, 0)
parch = st.number_input('Parents/Children Abroad', 0, 10, 0)
fare = st.number_input('fare', 0.0,600.0,32.0)
embarked = st.selectbox('Embarked',['C','Q','S'])


# Encode user inputs
sex_encoded = LEC_Sex.transform([sex])[0]
embarked_encoded = LEC_Embarked.transform([embarked])[0]

# create input DataFrame
input_data = pd.DataFrame([{
    'Pclass':pclass,
    'Sex':sex_encoded,
    'Age':age,
    'SibSp':sibsp,
    'Parch':parch,
    'Fare':fare,
    'Embarked':embarked_encoded
}])

if st.button('predict'):
    prediction = model.predict(input_data)[0]
    if prediction ==1:
        st.success('✅The perssenger Survived')
    else:
        st.error('❌The Perssenger did not Survived')