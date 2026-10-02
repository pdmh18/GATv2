from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def fit(self, train, val=None): ...
    @abstractmethod
    def predict_proba(self, X): ...
