from enum import Enum

class SupportedModels(Enum):
    LOGISTIC_REGRESSION = "LogisticRegression"
    SGD_CLASSIFIER = "SGDClassifier"
    DECISION_TREE = "DecisionTreeClassifier"
    ADA_BOOST = "AdaBoostClassifier"
    GRADIENT_BOOSTING = "GradientBoostingClassifier"
class HypterParameterSettings(Enum):
    DEFAULT= "default"
    CUSTOM= "custom"