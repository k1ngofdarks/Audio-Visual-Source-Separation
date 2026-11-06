import torch
from torch import nn


class AudioEncoder(nn.Module):
    """Encode complex STFT features with a 2D convolution."""

    def __init__(self, n_fft, hop_length, win_length, out_channels, **kwargs):
        """
        Args:
            n_fft (int): FFT size.
            hop_length (int): step between STFT frames, in samples.
            win_length (int): Hann window length, in samples.
            out_channels (int): number of output feature channels.
        """
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.out_channels = out_channels

        self.register_buffer("window", torch.hann_window(win_length))
        self.conv = nn.Conv2d(
            2, self.out_channels, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): audio waveforms of shape (B, L).
        Returns:
            Tensor: features of shape (B, out_channels, T, F), where
                T is the number of STFT frames and F = n_fft // 2 + 1.
        """
        x = torch.stft(
            x,
            window=self.window,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True,
        )
        x = torch.stack([x.real, x.imag], dim=1)
        x = x.transpose(-1, -2)
        x = self.conv(x)
        return x
