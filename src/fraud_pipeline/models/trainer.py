"""Bước 5: Vòng train (khung chung, chưa gắn thuật toán nào)."""
from .base import BaseModel


def train(model: BaseModel, train_data, val_data=None, **kwargs) -> BaseModel:
    # TODO: gắn thuật toán đã chọn
    return model.fit(train_data, val_data)
