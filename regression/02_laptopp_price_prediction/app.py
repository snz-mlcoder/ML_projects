import streamlit as st
import pandas as pd
import joblib

# Load trained models
@st.cache_resource
def load_models():
    return {
        "Linear Regression": joblib.load("model/Linear_Regression.joblib"),
        "Decision Tree": joblib.load("model/Decision_Tree.joblib"),
        "Random Forest": joblib.load("model/Random_Forest.joblib"),
        "AdaBoost": joblib.load("model/AdaBoost.joblib"),
        "XGBoost": joblib.load("model/XGBoost.joblib"),
    }

models = load_models()

# Load dataset to extract valid options per brand
@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned_laptop_data.csv")
    return df

df = load_data()

# Extract conditional options
brand_os = df.groupby("Company")["os"].unique().to_dict()
brand_cpu = df.groupby("Company")["Cpu brand"].unique().to_dict()
brand_gpu = df.groupby("Company")["Gpu brand"].unique().to_dict()
brand_types = df.groupby("Company")["TypeName"].unique().to_dict()

all_os = sorted(df["os"].unique())
all_cpu = sorted(df["Cpu brand"].unique())
all_gpu = sorted(df["Gpu brand"].unique())
all_types = sorted(df["TypeName"].unique())
all_companies = sorted(df["Company"].unique())

# Replace unknowns with dataset defaults
def handle_unknown(value, column_name):
    if value in ['Unknown', "I don't know", None, '']:
        if column_name in df.select_dtypes(include='number').columns:
            return df[column_name].mean()
        else:
            return df[column_name].mode()[0]
    return value

# App UI
st.title("💻 Laptop Price Prediction")

st.sidebar.header("⚙️ Select Prediction Model")
model_name = st.sidebar.selectbox("Model", list(models.keys()))
model = models[model_name]

st.header("📝 Laptop Configuration")

# Brand selection
company = st.selectbox("Brand", ["I don't know"] + all_companies)

# Conditional dropdowns
if company in brand_types:
    type_options = sorted(brand_types[company])
    cpu_options = sorted(brand_cpu.get(company, all_cpu))
    gpu_options = sorted(brand_gpu.get(company, all_gpu))
    os_options = sorted(brand_os.get(company, all_os))
else:
    type_options = all_types
    cpu_options = all_cpu
    gpu_options = all_gpu
    os_options = all_os

typename = st.selectbox("Laptop Type", ["I don't know"] + type_options)
ram = st.slider("RAM (GB)", 2, 64, 8)
weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, step=0.1)
touchscreen = st.selectbox("Touchscreen", ["I don't know", "No", "Yes"])
ips = st.selectbox("IPS Panel", ["I don't know", "No", "Yes"])
ppi = st.number_input("Pixel Density (PPI)", 100, 400, step=1)
cpu = st.selectbox("CPU Brand", ["I don't know"] + cpu_options)
hdd = st.selectbox("HDD (GB)", [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox("SSD (GB)", [0, 128, 256, 512, 1024])
gpu = st.selectbox("GPU Brand", ["I don't know"] + gpu_options)
os = st.selectbox("Operating System", ["I don't know"] + os_options)

# Predict button
if st.button("Predict Price"):
    touchscreen_bin = 1 if touchscreen == 'Yes' else 0 if touchscreen == 'No' else df["TouchScreen"].mode()[0]
    ips_bin = 1 if ips == 'Yes' else 0 if ips == 'No' else df["IPS"].mode()[0]

    # Handle all fields for unknowns
    company = handle_unknown(company, 'Company')
    typename = handle_unknown(typename, 'TypeName')
    cpu = handle_unknown(cpu, 'Cpu brand')
    gpu = handle_unknown(gpu, 'Gpu brand')
    os = handle_unknown(os, 'os')
    weight = handle_unknown(weight, 'Weight')
    ppi = handle_unknown(ppi, 'ppi')
    hdd = handle_unknown(hdd, 'HDD')
    ssd = handle_unknown(ssd, 'SSD')
    ram = handle_unknown(ram, 'Ram')

    input_data = {
        "Company": [company],
        "TypeName": [typename],
        "Ram": [ram],
        "Weight": [weight],
        "TouchScreen": [touchscreen_bin],
        "IPS": [ips_bin],
        "ppi": [ppi],
        "Cpu brand": [cpu],
        "HDD": [hdd],
        "SSD": [ssd],
        "Gpu brand": [gpu],
        "os": [os]
    }

    input_df = pd.DataFrame(input_data)
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Laptop Price: {int(prediction):,} $")
