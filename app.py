import streamlit as st
import pickle
import numpy as np
model=pickle.load(open('logreg.pkl','rb'))

def predict(Recency,Frequency,Monetary,Time):
    input=np.array([[Recency,Frequency,Monetary,Time]]).astype(np.float64)
    prediction=model.predict(input)
    return prediction

def main():
    st.title("Streamlit")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Predict Blood Donation </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    Recency=st.text_input("Months since the last donation")
    Frequency=st.text_input("Total number of donation")
    Monetary=st.text_input("Total blood donated in c.c.")
    Time=st.text_input(" Months since the first donation")

    if st.button("Predict"):
        output=predict(Recency,Frequency,Monetary,Time)
        st.success('The value is {}'.format(output))

if __name__=='__main__':
    main()
