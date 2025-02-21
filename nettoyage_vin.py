import pandas as pd 
import unicodedata

def remove_accents(text):
    if isinstance(text, str) :
        return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')

# Charger les données
df_vin = pd.read_csv(r"D:\Diginamic\2. Cours\34. Conception Et Developpement D'une IA\vin.csv")

# Suppression des accents
for col in df_vin.columns:
    if df_vin[col].dtype == 'object' :
        df_vin[col] = df_vin[col].apply(remove_accents)

# Correction de equilibre
df_vin = df_vin.replace("Vin euilibre", "Vin equilibre")

# Suppression de la première colonne
df_vin = df_vin.iloc[:, 1:]
        
# Génération nouveau CSV
df_vin.to_csv("data_vin.csv", index=False, encoding='utf-8')

print(df_vin.index)