from typing import Union, List, Tuple

import torch
from torch import nn
from transformers.utils import logging

from .layers import ResNet1DLayer, Enc_eeg, Proj_eeg, ResidualAdd, FlattenHead
from .atm.atm_s import iTransformer

logger = logging.get_logger(__name__)


class BrainMLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        intermediate_size: int = None,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.proj_in = nn.Linear(in_channels, hidden_size)

        self.layers = nn.ModuleList(
            [
                ResNet1DLayer(
                    embed_dim=hidden_size,
                    hidden_dim=(
                        intermediate_size
                        if intermediate_size is not None
                        else hidden_size
                    ),
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        x = self.proj_in(x.reshape(x.shape[0], -1).contiguous())

        for layer in self.layers:
            x = layer(x)

        return x


class NICE(nn.Module):
    def __init__(self, c_num: int, embedding_dim: int = 1440):
        super().__init__()
        self.enc_eeg = Enc_eeg(c_num=c_num)
        self.proj_eeg = Proj_eeg(embedding_dim=embedding_dim)

    def forward(self, x):
        eeg_embedding = self.enc_eeg(x)
        out = self.proj_eeg(eeg_embedding)

        return out


class BaseModel(nn.Module):
    def __init__(
        self,
        z_dim: int,
        c_num: int,
        timesteps: Union[List[int], Tuple[int, int]],
        embedding_dim: int = 1440,
    ):
        super().__init__()

        self.backbone = None
        self.project = nn.Sequential(
            FlattenHead(),
            nn.Linear(embedding_dim, z_dim),
            ResidualAdd(
                nn.Sequential(nn.GELU(), nn.Linear(z_dim, z_dim), nn.Dropout(0.5))
            ),
            nn.LayerNorm(z_dim),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.backbone(x)
        x = self.project(x)
        return x


class Shallownet(BaseModel):
    def __init__(
        self,
        z_dim: int,
        c_num: int,
        timesteps: Union[List[int], Tuple[int, int]],
        embedding_dim: int = 1440,
    ):
        super().__init__(z_dim, c_num, timesteps, embedding_dim)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (c_num, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.Dropout(0.5),
        )


class Deepnet(BaseModel):
    def __init__(
        self,
        z_dim: int,
        c_num: int,
        timesteps: Union[List[int], Tuple[int, int]],
        embedding_dim: int = 1400,
    ):
        super().__init__(z_dim, c_num, timesteps, embedding_dim=embedding_dim)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 25, (1, 10), (1, 1)),
            nn.Conv2d(25, 25, (c_num, 1), (1, 1)),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(25, 50, (1, 10), (1, 1)),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(50, 100, (1, 10), (1, 1)),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(100, 200, (1, 10), (1, 1)),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
        )


class EEGnet(BaseModel):
    def __init__(
        self,
        z_dim: int,
        c_num: int,
        timesteps: Union[List[int], Tuple[int, int]],
        embedding_dim: int = 1248,
    ):
        super().__init__(z_dim, c_num, timesteps, embedding_dim=embedding_dim)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), (1, 1)),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (c_num, 1), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(16, 16, (1, 16), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            # nn.AvgPool2d((1, 2), (1, 2)),
            nn.Dropout2d(0.5),
        )


class TSconv(BaseModel):
    def __init__(
        self,
        z_dim: int,
        c_num: int,
        timesteps: Union[List[int], Tuple[int, int]],
        embedding_dim=1440,
    ):
        super().__init__(z_dim, c_num, timesteps, embedding_dim)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (c_num, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )


class EEGProjectLayer(nn.Module):
    def __init__(
        self,
        z_dim: int,
        c_num: int,
        timesteps: Union[List[int], Tuple[int, int]],
        drop_proj=0.3,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps

        self.input_dim = self.c_num * (self.timesteps[1] - self.timesteps[0])
        proj_dim = z_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, proj_dim),
            ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(proj_dim, proj_dim),
                    nn.Dropout(drop_proj),
                )
            ),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, x):
        x = x.view(x.shape[0], self.input_dim)
        x = self.model(x)
        return x


class ATM(nn.Module):
    def __init__(
        self,
        seq_len: int = 250,
        c_num: int = 17,
        dropout: float = 0.25,
        z_dim: int = 1024,
    ):
        super().__init__()

        self.encoder = iTransformer(seq_len=seq_len, c_num=c_num, dropout=dropout)
        self.enc_eeg = Enc_eeg(c_num=c_num)
        self.proj_eeg = Proj_eeg(proj_dim=z_dim)

    def forward(self, x, subject_ids):
        x = self.encoder(x, None, subject_ids)
        # print(f'After attention shape: {x.shape}')
        # print("x", x.shape)
        # x = self.subject_wise_linear[0](x)
        # print(f'After subject-specific linear transformation shape: {x.shape}')
        eeg_embedding = self.enc_eeg(x)

        out = self.proj_eeg(eeg_embedding)
        return out


class BrainEncoder(nn.Module):
    def __init__(
        self,
        backbone: str,
        brain_channels: int = 17,
        num_layers: int = 4,
        brain_sequence_length: int = 250,
        embed_dim: int = 1024,
        extra_dim: int = 1440,  # for compatiblility to other models
        dropout: float = 0.0,
    ):
        super().__init__()

        backbone = backbone.lower()

        if backbone == "brain_mlp":
            in_channels = int(brain_channels * brain_sequence_length)
            self.encoder = BrainMLP(
                in_channels=in_channels,
                hidden_size=embed_dim,
                intermediate_size=embed_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif backbone == "nice":
            self.encoder = NICE(
                c_num=brain_channels, embedding_dim=extra_dim
            )  # 1440, 1040
        elif backbone == "shallow_net":
            self.encoder = Shallownet(
                z_dim=embed_dim,
                c_num=brain_channels,
                timesteps=None,
                embedding_dim=extra_dim,  # 1440, 1040
            )
        elif backbone == "deep_net":
            self.encoder = Deepnet(
                z_dim=embed_dim,
                c_num=brain_channels,
                timesteps=None,
                embedding_dim=extra_dim,  # 1400, 800
            )
        elif backbone == "eeg_net":
            self.encoder = EEGnet(
                z_dim=embed_dim,
                c_num=brain_channels,
                timesteps=None,
                embedding_dim=extra_dim,  # 1248, 864
            )
        elif backbone == "ts_conv":
            self.encoder = TSconv(
                z_dim=embed_dim,
                c_num=brain_channels,
                timesteps=None,
                embedding_dim=extra_dim,  # 1440, 1040
            )
        elif backbone == "brain_projection":
            self.encoder = EEGProjectLayer(
                z_dim=embed_dim,
                c_num=brain_channels,
                timesteps=[0, brain_sequence_length],
                drop_proj=dropout,  # use 0.3 following UBP's implementation
            )
        elif backbone == "atm":
            self.encoder = ATM(
                z_dim=embed_dim,
                seq_len=brain_sequence_length,
                c_num=brain_channels,
                dropout=dropout,  # use 0.25 following ATM's implementation
            )
        else:
            raise ValueError(f"Invalid brain backbone: {backbone}")

    def forward(
        self, brain_signals: torch.Tensor, subject_ids: torch.Tensor
    ) -> torch.Tensor:
        additional_kwargs = {}
        if isinstance(self.encoder, ATM):
            additional_kwargs.update({"subject_ids": subject_ids})

        brain_embeds = self.encoder(brain_signals, **additional_kwargs)

        return brain_embeds
