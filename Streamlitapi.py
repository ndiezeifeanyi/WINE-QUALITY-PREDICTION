#Load the model
import pickle
import streamlit as st
Model= pickle.load(open("C:/Users/NDIEZE/Documents/Model Deployment M4ACE/WineQuality.pkl", "rb"))
S_cale= pickle.load(open("C:/Users/NDIEZE/Documents/Model Deployment M4ACE/WineQualityScaler.pkl", "rb"))

#Define input variable
def main():
    st.title("Wine Quality Taste")
    #inputs for model predictions
    fixed_acidity = st.number_input("fixed acidity")
    volatile_acidity = st.number_input("volatile acidity")
    citric_acid = st.number_input("citric acid")
    residual_sugar = st.number_input("residual sugar")
    chlorides = st.number_input("chlorides")
    free_sulfur_dioxide = st.number_input("free sulfur dioxide")
    total_sulfur_dioxide = st.number_input("total sulfur dioxide")
    density = st.number_input("density")
    pH = st.number_input("pH")
    sulphates = st.number_input("sulphates")
    alcohol = st.number_input("alcohol")

    #Code Prediction
    if st.button("Check Wine Quality"):
        # Scale the input features
        input = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]]
        scaled = S_cale.transform(input)
        prediction = Model.predict(scaled)
        output = round(prediction[0], 2)
        st.success("The predicted wine quality is {}".format(output))
if __name__ == "__main__":
    main()
