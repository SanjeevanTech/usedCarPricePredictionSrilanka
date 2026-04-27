# 🚗 Used Car Price Prediction — Sri Lanka

A machine learning web application that predicts the market price of used vehicles in Sri Lanka. Built with a **Random Forest** model trained on real Sri Lankan car listing data, served through a **Flask** REST API with a modern glassmorphic UI.

---

## 📸 Preview

> Enter vehicle details → Get an estimated market price in LKR Lakhs instantly.

---

## 🗂️ Project Structure

```
2019icts11MlProject/
│
├── data/
│   └── car_price_dataset.csv          # Raw training dataset (9,788 listings)
│
├── backend/
│   ├── app.py                         # Flask API server
│   ├── requirements.txt               # Python dependencies
│   ├── options.json                   # Valid dropdown options (auto-generated)
│   ├── model/
│   │   ├── vehicle_price_model.pkl    # Trained ML model
│   │   ├── preprocessor.pkl           # Feature preprocessor pipeline
│   │   └── label_encoders.pkl         # Label encoders for categorical features
│   └── templates/
│       └── index.html                 # Frontend UI (single-page app)
│
├── vehicle_price_prediction.ipynb     # Jupyter Notebook (EDA + Model Training)
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Machine Learning | Scikit-learn (Random Forest Regressor) |
| Data Processing | Pandas, NumPy |
| Backend API | Python, Flask, Flask-CORS |
| Frontend | HTML5, CSS3 (Glassmorphism), Vanilla JavaScript |
| Model Serialization | Joblib |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### 1. Clone the Repository

```bash
git clone https://github.com/SanjeevanTech/usedCarPricePredictionSrilanka.git
cd usedCarPricePredictionSrilanka
```

### 2. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### 3. Run the Flask App

```bash
cd backend
python app.py
```

The application will start at: **http://127.0.0.1:5000**

### 4. (Optional) Run the Jupyter Notebook

To explore the data analysis and model training:

```bash
pip install jupyter
jupyter notebook
```

Then open `vehicle_price_prediction.ipynb` in your browser.

---

## 🧠 How It Works

1. **Data Collection** — Real used car listings scraped from Sri Lankan vehicle marketplaces.
2. **Preprocessing** — Label encoding for categorical features (Brand, Model, Town), standard scaling for numerical features.
3. **Model Training** — A Random Forest Regressor is trained on 9,788 listings with features like Brand, Model, Year, Engine CC, Mileage, Fuel Type, Gear, and more.
4. **Prediction API** — The trained model is served via a Flask REST API (`/predict` endpoint).
5. **Smart Validation** — The frontend dynamically restricts Fuel Type, Gear, and Engine CC dropdowns based on the selected car model, using a mapping built from the actual training data. This prevents impossible combinations (e.g., Electric Wagon R).

---

## 🔌 API Reference

### `GET /options`
Returns all valid dropdown values for the prediction form.

**Response:**
```json
{
  "Brand": ["SUZUKI", "TOYOTA", ...],
  "ModelsByBrand": { "SUZUKI": ["WAGON R", ...] },
  "FuelByModel": { "WAGON R": ["PETROL"] },
  "GearByModel": { "WAGON R": ["AUTOMATIC", "MANUAL"] },
  ...
}
```

---

### `POST /predict`
Predicts the market price of a vehicle.

**Request Body:**
```json
{
  "Brand": "SUZUKI",
  "Model": "WAGON R STINGRAY TURBO",
  "YOM": 2018,
  "Engine_cc": 660,
  "Gear": "Automatic",
  "Fuel_Type": "Petrol",
  "Millage_KM": 50000,
  "Town": "COLOMBO",
  "Leasing": "No Leasing",
  "Condition": "USED",
  "AIR_CONDITION": "Available",
  "POWER_STEERING": "Available",
  "POWER_MIRROR": "Available",
  "POWER_WINDOW": "Available"
}
```

**Response:**
```json
{
  "estimated_price_lakhs": 59.44
}
```

---

## ✅ Input Validation

The application enforces the following rules to ensure accurate predictions:

| Field | Rule |
|---|---|
| Engine Capacity | Must be 100 – 5,000 cc |
| Mileage | Must be 0 – 300,000 KM |
| Fuel Type | Locked to only fuels seen for that model in training data |
| Gear Type | Locked to only gears seen for that model in training data |
| Brand / Model | Must be selected from known values |

---

## 📊 Model Features

| Feature | Type |
|---|---|
| Brand | Categorical (Label Encoded) |
| Model | Categorical (Label Encoded) |
| Year of Manufacture (YOM) | Numerical |
| Engine Capacity (cc) | Numerical |
| Gear Type | Categorical (One-Hot) |
| Fuel Type | Categorical (One-Hot) |
| Mileage (KM) | Numerical |
| Town | Categorical (Label Encoded) |
| Leasing | Categorical (One-Hot) |
| Condition | Categorical (One-Hot) |
| Air Condition | Categorical (One-Hot) |
| Power Steering | Categorical (One-Hot) |
| Power Mirror | Categorical (One-Hot) |
| Power Window | Categorical (One-Hot) |

---

## 👨‍💻 Author

**Sanjeevan T.**  
BSc (Hons) in Information & Communication Technology  
University of Vavuniya — 2019ICTS11

---

## 📄 License

This project is for academic purposes. All rights reserved.
