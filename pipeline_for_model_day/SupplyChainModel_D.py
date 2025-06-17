import torch
import torch.nn as nn

class SupplyChainModel_D(nn.Module):
    def __init__(self):
        super().__init__()
        self.market_emb = nn.Embedding(5, 5)
        self.ship_emb = nn.Embedding(4, 4)
        self.order_city_emb = nn.Embedding(3597, 1799)
        self.customer_city_emb = nn.Embedding(563, 282)

        # total embeddings + 3 features numériques
        self.fc1 = nn.Linear(5 + 4 + 1799 + 282 + 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x_cat, x_num):
        e1 = self.market_emb(x_cat[:, 0])
        e2 = self.ship_emb(x_cat[:, 1])
        e3 = self.order_city_emb(x_cat[:, 2])
        e4 = self.customer_city_emb(x_cat[:, 3])
        x = torch.cat([e1, e2, e3, e4, x_num], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)  # pas de sigmoid ici (BCEWithLogitsLoss s'en charge)
