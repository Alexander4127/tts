import torch
import torch.nn as nn
import torch.nn.functional as F

from .adaptor import Predictor
from .utils import create_alignment


class LengthRegulator(nn.Module):
    """ Length Regulator """
    def __init__(self, encoder_dim, alpha, filter_size, kernel_size, dropout):
        super().__init__()
        self.alpha = alpha
        self.duration_predictor = Predictor(encoder_dim, filter_size, kernel_size, dropout)

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(output, (0, 0, 0, mel_max_length - output.size(1), 0, 0))
        return output

    def forward(self, x, target=None, mel_max_length=None):
        duration_predictor_output = self.duration_predictor(x)
        if target is None:
            # inference
            duration_predictor_output = F.relu(duration_predictor_output * self.alpha + 0.5).long()
            out = self.LR(x, duration_predictor_output)
            return out, torch.stack([torch.Tensor([i + 1 for i in range(out.shape[1])])]).long().to(x.device)

        # train
        out = self.LR(x, target, mel_max_length)
        return out, duration_predictor_output
