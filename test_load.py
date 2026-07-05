import joblib
import traceback
try:
    model = joblib.load('models/advanced_xgboost_model.pkl')
    print('Loaded successfully! Type:', type(model))
except Exception as e:
    print('Error loading model:')
    traceback.print_exc()
