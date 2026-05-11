# ⚡ Quick Start Guide - Circuit Fault Detection

## 🚀 Start in 5 Minutes

### Step 1: Backend Setup (Terminal 1)

```bash
# Navigate to backend folder
cd backend

# Install Python packages
pip install -r requirements.txt

# Start Flask server
python app.py
```

You should see:
```
Running on http://127.0.0.1:5000
```

### Step 2: Frontend Setup (Terminal 2)

```bash
# Navigate to frontend folder
cd frontend

# Install packages (first time only)
npm install

# Start React app
npm start
```

Your browser should open to `http://localhost:3000` automatically!

## 📊 Try It Out

### 1. Upload Sample Data
- Click the **"📤 Upload Data"** tab
- Upload: `../sample_data/circuit_training_data.csv`
- You'll see a preview of the data

### 2. Train Models
- Click the **"🧠 Train Model"** tab
- Select target column: **"fault"**
- Keep all 4 models selected (Logistic Regression, Decision Tree, Random Forest, SVM)
- Click **"🚀 Start Training"**
- Wait for training to complete (~10-30 seconds)
- See results with metrics for each model

### 3. Make Predictions
- Click the **"🔮 Predict"** tab
- Enter circuit parameters:
  - voltage: 220
  - current: 5.2
  - temperature: 45
  - resistance: 42.3
  - frequency: 50
  - power: 1144
- Click **"🚀 Predict"**
- See if the circuit has faults!

### 4. View Model Info
- Click the **"ℹ️ Model Info"** tab
- See all models' performance metrics
- View confusion matrices and detailed stats

## 📋 CSV Format for Your Own Data

Create a CSV with these columns:
```
voltage,current,temperature,resistance,frequency,power,fault
220,5.2,45,42.3,50,1144,0
230,7.8,88,28.8,50,1755,1
```

- **voltage**: AC/DC voltage (e.g., 120, 220, 380V)
- **current**: Current in Amps (e.g., 2.5, 5.2, 8.5A)
- **temperature**: Temperature in °C (e.g., 35-100°C)
- **resistance**: Resistance in Ohms (e.g., 20-50Ω)
- **frequency**: Frequency in Hz (e.g., 50 or 60)
- **power**: Power in Watts (e.g., 300-5000W)
- **fault**: 0 (No fault) or 1 (Fault detected)

## 🔧 Troubleshooting

### "Cannot connect to server"
- Make sure Flask is running in Terminal 1
- Check: `http://localhost:5000/api/health`

### "No models available"
- Upload data first before trying to train
- Make sure CSV has correct columns

### "Button is disabled"
- You need to upload data first
- Then train a model
- Then you can make predictions

## 🎓 Understanding the Results

After training, you'll see:

| Metric | Meaning |
|--------|---------|
| **Accuracy** | % of correct predictions overall |
| **Precision** | When it predicts FAULT, how often is it right? |
| **Recall** | Of all the real FAULTS, how many did it catch? |
| **F1 Score** | Balance between Precision & Recall |

## 💡 Tips

1. **More data = Better models** - Train with at least 100+ samples
2. **Balance matters** - Have similar numbers of fault and non-fault samples
3. **Feature scaling** - The app handles this automatically
4. **Model comparison** - Try different models to see which works best for your data

## 📞 Need Help?

Check the README.md for full documentation and API details!

---

**Your Circuit Fault Detection System is Ready! ⚡🔧**
