import streamlit as st
st.write("Hello World")

st.sidebar.header("Header")

def user_input():
    name = st.sidebar.text_input("Name")
    age = st.sidebar.number_input("Age", min_value=0, max_value=99)
    sexe = st.sidebar.selectbox("Sexe", ["M", "F"])
    rate = st.sidebar.slider("Rate", min_value=0, max_value=5)
    data = {"name": name, "age": age, "sexe" : sexe, "rate": rate}
    return data

def fn(data):
    return data

data = user_input()
st.subheader("User Input")
st.write(data)

output = fn(data)
st.subheader("Output") 
st.write(output)
