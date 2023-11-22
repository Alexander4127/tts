import torch
import torch.nn as nn


class FastSpeechLoss(nn.Module):
    def __init__(self, **kwargs):
        """
        Construct loss for FastSpeech2 model
        """
        super().__init__(**kwargs)
        self.mse = nn.MSELoss()

    def __call__(self, mel, mel_target,
                 duration_pred, duration_target,
                 pitch_pred, pitch_target,
                 energy_pred, energy_target,
                 **kwargs):
        mel_loss = self.mse(mel, mel_target)
        duration_predictor_loss = self.mse(duration_pred, torch.log1p(duration_target.float()))
        pitch_predictor_loss = self.mse(pitch_pred, torch.log1p(pitch_target.float()))
        energy_predictor_loss = self.mse(energy_pred, torch.log1p(energy_target.float()))
        return mel_loss, duration_predictor_loss, pitch_predictor_loss, energy_predictor_loss
