import pandas as pd
import numpy as np
import streamlit as st
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
st.title("PREDICTION MODEL")
def generate_sidebar_from_csv(df):
    
    df_features = df.iloc[:, :-1]
    df_target = df.iloc[:, -1]
    
    
    feature_values = []
    for col in df_features.columns:
        col_type = df_features[col].dtype

        if pd.api.types.is_numeric_dtype(col_type):
            min_value = float(df_features[col].min())
            max_value = float(df_features[col].max())
            value = (min_value + max_value) / 2
            feature_values.append(st.sidebar.number_input(f"Select {col}", min_value=min_value, max_value=max_value, value=value))

        elif pd.api.types.is_string_dtype(col_type):
            unique_values = df_features[col].unique()
            selected_value = st.sidebar.selectbox(f"Select {col}", unique_values)
            feature_values.append(selected_value)

        elif pd.api.types.is_bool_dtype(col_type):
            feature_values.append(st.sidebar.checkbox(f"Select {col}", value=bool(df_features[col].iloc[0])))

        elif pd.api.types.is_datetime64_any_dtype(col_type):
            min_date = df_features[col].min().date()
            max_date = df_features[col].max().date()
            feature_values.append(st.sidebar.date_input(f"Select {col}", value=min_date, min_value=min_date, max_value=max_date))

        else:
            feature_values.append(st.sidebar.text_input(f"Input for {col}"))
    
    return feature_values, df_features, df_target

def prediction(X, Y, input_features, mean_value):
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y,random_state=2)
    
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, Y_train)
    
    X_train_prediction = classifier.predict(X_train)
    training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
    st.write('Accuracy score of the training data:', training_data_accuracy)
    
    X_test_prediction = classifier.predict(X_test)
    testing_data_accuracy = accuracy_score(Y_test, X_test_prediction)
    st.write('Accuracy score of the testing data:', testing_data_accuracy)
    
    input_data_as_numpy_array = np.asarray(input_features).reshape(1, -1)
    std_data = scaler.transform(input_data_as_numpy_array)
    st.write('Transformed Data:')
    st.write(std_data)

    prediction = classifier.predict(std_data)
    st.write('The Prediction:', prediction)
    
    if prediction[0] > mean_value:
        st.write('RED')
    else:
        st.write('WHITE')

def main():
    uploaded_file = st.file_uploader('Choose a CSV file', type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        mean_value = df.iloc[:, -1].mean()
        st.write('Mean value of the last column:', mean_value)
        st.sidebar.header('USER INPUTS')
        input_features, df_features, df_target = generate_sidebar_from_csv(df)
        
        X = df_features
        Y = df_target
        
        if st.button('Make Prediction'):
            prediction(X, Y, input_features, mean_value)

if __name__ == "__main__":
    main()

