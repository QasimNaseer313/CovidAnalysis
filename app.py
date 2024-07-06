import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

def load_data(filepath):
    data = pd.read_csv(filepath)
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    return data.dropna(subset=['Date', 'Country'])

def filter_data(df, country, start_date, end_date):
    mask = (df['Country'] == country) & (df['Date'].between(start_date, end_date))
    return df.loc[mask].dropna(subset=['Total_Recovered', 'Total_Deaths', 'Total_Cases'])

def check_missing_data(df):
    return df.isnull().sum()

def check_duplicates(df):
    return df.duplicated().sum()

def perform_eda(df, chart_type):
    # Extend to include more chart options or configurations
    if chart_type == 'Scatter Plot':
        return px.scatter(df, x='Total_Recovered', y='Total_Deaths', size='Total_Recovered', color='Total_Deaths', title="Scatter Plot")
    elif chart_type == 'Line Chart':
        return px.bar(df, x='Date', y='Total_Cases', title="Bar Chart")
    elif chart_type == 'Bar Chart':
        return px.line(df, x='Date', y='Total_Cases', title="Line Chart")
    elif chart_type == 'Heatmap':
        fig = go.Figure(data=go.Heatmap(z=df['Total_Cases'], x=df['Date'], y=df['Country'], colorscale='Viridis'))
        fig.update_layout(title="Heatmap")
        return fig
    elif chart_type == 'Pie Chart':
        return px.pie(df, names='Date', values='Total_Cases', title="Pie Chart")
    return None


def encode_data(df, column, encoding_type):
    if encoding_type == 'One-Hot Encoding':
        # Return the DataFrame with dummy variables created
        return pd.get_dummies(df, columns=[column])
    elif encoding_type == 'Label Encoding':
        # Check if the column is categorical
        if pd.api.types.is_categorical_dtype(df[column]) or df[column].dtype == object:
            df[column] = df[column].astype('category').cat.codes
            return df
        else:
            st.error('Label Encoding can only be applied to categorical data.')
            return df
    else:
        # If the encoding type is not recognized, return the original DataFrame unchanged
        st.error('Selected encoding type is not recognized.')
        return df
        
# Function to prepare data for KNN
def prepare_data_for_knn(df, target, classification):
    if classification:
        median_value = df[target].median()
        df['Category'] = (df[target] > median_value).astype(int)
        X = df.drop([target, 'Category', 'Country', 'Date'], axis=1)
        y = df['Category']
    else:
        X = df.drop([target, 'Country', 'Date'], axis=1)
        y = df[target]

    # Imputing NaN values
    imputer = SimpleImputer(strategy='mean')  # You can change the strategy to 'median' or 'most_frequent' if needed
    X_imputed = imputer.fit_transform(X)
    y = y.dropna()  # Just in case y contains NaNs

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Function to apply KNN
def apply_knn(df, target, neighbors, classification=False):
    X_train, X_test, y_train, y_test = prepare_data_for_knn(df, target, classification)
    model = KNeighborsClassifier(n_neighbors=neighbors) if classification else KNeighborsRegressor(n_neighbors=neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if classification:
        cm = confusion_matrix(y_test, y_pred)
        cmd = ConfusionMatrixDisplay(cm)
        return cmd
    else:
        mse = mean_squared_error(y_test, y_pred)
        return mse


def main():
    st.title('COVID-19 Country Level Analysis')
    df = load_data('covid19_sample_data.csv')
    country_selected = st.sidebar.selectbox('Select a Country', df['Country'].unique())

    # Convert Streamlit date inputs to pandas datetime at input
    start_date = pd.to_datetime(st.sidebar.date_input('Start Date', df['Date'].min()))
    end_date = pd.to_datetime(st.sidebar.date_input('End Date', df['Date'].max()))

    df_filtered = filter_data(df, country_selected, start_date, end_date)

    if st.checkbox("Show raw data"):
        st.write(df_filtered)

    # EDA
    chart_type = st.sidebar.radio('Select Chart Type', ['Scatter Plot', 'Bar Chart', 'Line Chart', 'Heatmap', 'Pie Chart'])
    fig = perform_eda(df_filtered, chart_type)
    st.plotly_chart(fig, use_container_width=True)

    # Data checks
    if st.sidebar.button('Check Missing Values'):
        missing_data = check_missing_data(df_filtered)
        st.sidebar.write("Missing Values:", missing_data)

    if st.sidebar.button('Check Duplicates'):
        duplicate_data = check_duplicates(df_filtered)
        st.sidebar.write("Duplicate Rows:", duplicate_data)

    # Scale data
    if st.sidebar.button('Scale Data'):
        scaler = StandardScaler()
        df[['Total_Recovered', 'Total_Deaths', 'Total_Cases']] = scaler.fit_transform(df[['Total_Recovered', 'Total_Deaths', 'Total_Cases']])
        st.write('Data scaled')

    # Encoding data
    encoding_type = st.sidebar.radio('Encoding Type', ['One-Hot Encoding', 'Label Encoding'])
    column_to_encode = st.sidebar.selectbox('Select Column to Encode', df.columns)

    if st.sidebar.button('Apply Encoding'):
        encoded_df = encode_data(df, column_to_encode, encoding_type)
        if encoded_df is not None:
            st.write("Encoded Data:")
            st.dataframe(encoded_df.head())  # Display the DataFrame on the front end

    
 # User control for setting the number of neighbors
    neighbors = st.sidebar.slider('Select Number of Neighbors (k)', 1, 20, 5)

    # Apply KNN Regression
    if st.sidebar.button('Apply KNN Regression'):
        mse = apply_knn(df_filtered, 'Total_Cases', neighbors)
        st.write(f'MSE: {mse}')

    # Apply KNN Classification
    if st.sidebar.button('Apply KNN Classification'):
        cmd = apply_knn(df_filtered, 'Total_Cases', neighbors, classification=True)
        fig, ax = plt.subplots()
        cmd.plot(ax=ax)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
