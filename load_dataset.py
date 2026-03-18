import kaggle
import pandas as pd

# Télécharge et dézippe le CSV dans le dossier courant
kaggle.api.dataset_download_files(
    "mczielinski/bitcoin-historical-data", path="./data", unzip=True
)

# Charge le CSV
df = pd.read_csv("./data/btcusd_1-min_data.csv")
print(df.tail())  # Affiche les dernières lignes (les plus récentes)
