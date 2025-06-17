
import torch
from torch.utils.data import Dataset

class SupplyChainDataset(Dataset):
    def __init__(self, dataframe):
        super(SupplyChainDataset, self).__init__()
        self.X_cat = dataframe[["market_index", "shipping_mode_index", "order_city_index", "customer_city_index"]].values
        self.X_num = dataframe[["product_card_id", "days_for_shipment_(scheduled)"]].values
        self.y = dataframe["late_delivery_risk"].values  # 0 ou 1

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "cat": torch.tensor(self.X_cat[idx], dtype=torch.long),
            "num": torch.tensor(self.X_num[idx], dtype=torch.float32),
            "target": torch.tensor(self.y[idx], dtype=torch.float32)  # binaire → float
       
        }
