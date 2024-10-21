from sklearn.calibration import LabelEncoder
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lime import lime_tabular



df = pd.read_csv('Salary Prediction of Data Professions.csv')
df.dropna()

label_encoder = LabelEncoder()
df['SEX'] = label_encoder.fit_transform(df['SEX'])


X = df[['AGE', 'RATINGS', 'SEX']]
y = df['SALARY']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


st.title("Salary Prediction Dashboard :bar_chart: :chart_with_upwards_trend:")
st.markdown("Predict Salary based on Age, Ratings, and Gender")

tab1, tab2, tab3 = st.tabs(["Data :clipboard:", "Global Performance :weight_lifter:", "Local Performance :bicyclist:"])

with tab1:
    st.header("Salary Dataset")
    st.write(df)
    

with tab2:
    st.header("Model Performance Metrics")
    st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
    st.write(f"R^2 Score: {r2_score(y_test, y_pred):.2f}")
    plt.figure(figsize=(10, 6))
    plt.hist(df['AGE'], bins=15, color='blue', alpha=0.5, edgecolor='black', label='Age Distribution')

    plt.scatter(df['AGE'].iloc[y_test.index], y_test, color='blue', label='Real Values', alpha=0.6)
    plt.scatter(df['AGE'].iloc[y_test.index], y_pred, color='orange', label='Predicted Values', alpha=0.6)

    plt.title('Age Distribution and Salary Estimates (Random Forest Regression)')
    plt.xlabel('Age')
    plt.ylabel('Average Salary')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    

with tab3:
    st.header("Input Features for Prediction")
    age = st.slider("Age", min_value=int(df['AGE'].min()), max_value=int(df['AGE'].max()))
    ratings = st.slider("Ratings", min_value=float(df['RATINGS'].min()), max_value=float(df['RATINGS'].max()))
    sex = st.selectbox("Gender", options=['M', 'F'])
    sex_encoded = 1 if sex == 'M' else 0

    if st.button("Predict Salary"):
        input_data = np.array([[age, ratings, sex_encoded]])
        
        
        if np.any(np.isnan(input_data)) or np.any(np.isinf(input_data)):
            st.error("Input data contains NaN or infinite values.")
            st.stop()  #return ise yaramadi
        
        prediction = model.predict(input_data)
        st.markdown(f"### Predicted Salary: <strong style='color:tomato;'>{prediction[0]:.2f}</strong>", unsafe_allow_html=True)

        #--LIME
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            mode="regression",
            feature_names=['AGE', 'RATINGS', 'SEX'],
            training_labels=y_train.values,
            discretize_continuous=True, 
            discretizer='entropy'  #discretezer ise yaramadi
        )
        
        explanation = explainer.explain_instance(input_data[:], model.predict)
        interpretation_fig = explanation.as_pyplot_figure()
        st.pyplot(interpretation_fig, use_container_width=True)
