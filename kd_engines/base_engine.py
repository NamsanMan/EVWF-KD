import torch.nn as nn
from abc import ABC, abstractmethod

class BaseKDEngine(nn.Module, ABC):
    """모든 KD 엔진의 기본이 되는 추상 클래스"""
    def __init__(self, teacher, student, **kwargs):
        super().__init__()
        self.teacher = teacher
        self.student = student

    @abstractmethod
    def compute_losses(self, imgs, masks, device):
        """
        KD 손실을 계산하는 핵심 메서드.
        모든 하위 클래스는 이 메서드를 반드시 구현해야 합니다.
        반환값: loss 딕셔너리 ({"total": ..., "ce_student": ..., ...})
        """
        raise NotImplementedError

    def get_extra_parameters(self):
        """
        KD 엔진 자체가 학습해야 할 파라미터가 있을 경우 반환합니다.
        (예: Adaptation Layer의 가중치)
        """
        return []