import joblib

from src.data import OUTPUT_DIR
from src.modeling import predict_examples

model = joblib.load(OUTPUT_DIR / "co2_final_model.joblib")
predict_examples(model, list(model.feature_names_in_))