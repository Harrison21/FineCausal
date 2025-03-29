# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss with capability to emphasize main task

    Params：
        num: int，the number of loss
        main_task_weight: float, additional weight for the main task (first loss)
        x: multi-task loss
    Examples：
        loss1=1  # AQA loss (main task)
        loss2=2  # TAS loss
        loss3=3  # Mask loss
        awl = AutomaticWeightedLoss(3, main_task_weight=5.0)
        loss_sum = awl(loss1, loss2, loss3)
    """

    def __init__(self, num=2, main_task_weight=1.0):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)
        self.main_task_weight = (
            main_task_weight  # Additional weight for the main task (AQA loss)
        )

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            # Apply additional weight of 5.0 to the first loss (AQA loss)
            task_weight = self.main_task_weight if i == 0 else 1.0
            loss_sum += task_weight * (
                0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
            )
        return loss_sum
