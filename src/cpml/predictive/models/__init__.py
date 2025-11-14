from .base import CPModel
from .linear import CPLinearRegressionModel
from .polylinear import CPPolyLinearRegressionModel
from .random_forest import CPRandomForestModel
from .gradient_boost import CPGradientBoostModel
from .lightgbm import CPLightGBModel
from .neural_network import CPNeuralNetwork
from .xgboost_model import CPXGBoostModel

MODEL_CLASSES = [
    CPLinearRegressionModel,
    CPPolyLinearRegressionModel,
    CPRandomForestModel,
    CPGradientBoostModel,
    CPLightGBModel,
    CPNeuralNetwork,
    CPXGBoostModel
]

MODEL_REGISTRY = {
    "linear": CPLinearRegressionModel,
    "poly_linear": CPPolyLinearRegressionModel,
    "random_forest": CPRandomForestModel,
    "gradient_boost": CPGradientBoostModel,
    "lightgbm": CPLightGBModel,
    "nn": CPNeuralNetwork,
    "xgboost": CPXGBoostModel,
}