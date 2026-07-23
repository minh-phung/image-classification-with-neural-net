import torch

LOSS = {
    "BCE_with_logit": torch.nn.BCEWithLogitsLoss(reduction = 'sum')
}