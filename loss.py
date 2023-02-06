import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # predictions shape: [n, 1470]
        # 1470 = S * S * (C + B * 5) = [S, S, 30] = [S, S, C+B*5]

        # target shape: [n, S, S, 25]

        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5)
        # predictions new shape: [n, S, S, 30]

        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25]) # boxes shapes: [n, S, S, 4]
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25]) # iou shape: [n, S, S, 1]
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0) # ious shape: [2, n, S, S, 1]
        iou_maxes, best_box = torch.max(ious, dim=0) # iou_maxes shape and best_box shapes: [n, S, S, 1]
        exists_box = target[..., 20].unsqueeze(3) # identity of obj i (if there is an object in cell i)
        # shape: [n, S, S, 1], and it is gonna be 0 or 1, if there is an object in cell

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #
        box_predictions = exists_box * (
            best_box * predictions[..., 26:30]
            + (1 - best_box) * predictions[..., 21:25]
        ) # shape: [n, S, S, 4]

        box_targets = exists_box * target[..., 21:25]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # (N, S, S, 4) -> (N*S*S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )


        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #
        pred_box = (
            best_box * predictions[..., 25:26] + (1 - best_box) * predictions[..., 20:21] # Getting the probability of the best box
        ) 

        # (N*S*S)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),  # Exists box is for identity
            torch.flatten(exists_box * target[..., 20:21]),
        )


        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #
        # (N, S, S, 1) -> (N, S*S)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # =================== #
        #   FOR CLASS LOSS    #
        # =================== #

        # (N, S, S, 20) -> (N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], start_dim=-2),
            torch.flatten(exists_box * target[..., :20], start_dim=-2),
        )

        loss = (
            self.lambda_coord * box_loss # First two rows os loss in paper
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss