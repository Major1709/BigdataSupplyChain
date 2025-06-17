import torch
import torch.nn as nn

class SupplyChainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.market_emb = nn.Embedding(5, 4)
        self.ship_emb = nn.Embedding(4, 4)
        self.country_emb = nn.Embedding(3597, 16)
        self.segment_emb = nn.Embedding(563, 8)

        # total embeddings + 3 features numériques
        self.fc1 = nn.Linear(4 + 4 + 16 + 8 + 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)  # 1 sortie pour binaire

    def forward(self, x_cat, x_num):
        e1 = self.market_emb(x_cat[:, 0])
        e2 = self.ship_emb(x_cat[:, 1])
        e3 = self.country_emb(x_cat[:, 2])
        e4 = self.segment_emb(x_cat[:, 3])
        x = torch.cat([e1, e2, e3, e4, x_num], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)  # pas de sigmoid ici (BCEWithLogitsLoss s'en charge)
