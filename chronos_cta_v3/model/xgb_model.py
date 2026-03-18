"""XGBoost alpha model wrapper."""

from xgboost import XGBRegressor


class XGBModel:
    def __init__(self, random_state: int = 42):
        self.model = XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
        )

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)
