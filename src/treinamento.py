import pandas as pd
import joblib as jb 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
#teste
# tranfere os dados do arquivo csv para um DataFrame
df = pd.read_csv("data/Atividade_Cap10_produtos_agricolas.csv")

#separando features e labels
X = df.drop("label", axis=1)
y = df["label"]

#dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) #20% para teste
#garante a divisão como sempre a mesma

scaler = StandardScaler()#padroniza os dados
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print("Dados preparados e escalados :D")
print(f"Treinamento (80)%: {X_train_scaled.shape}, Teste(20%): {X_test_scaled.shape}")

#dicionário para armazenar os resultados
resultados = {}

print("Iniciando treinamento do modelo...")

#definindo o modelo 
model = RandomForestClassifier(random_state=42)
#quando for testar um modelo só, usar apenas o nome do modelo sem os parênteses


#treinando o modelo
#treinamento
model.fit(X_train_scaled, y_train)
#teste
y_pred = model.predict(X_test_scaled)
#avaliar
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo RandomForestClassifier: {accuracy:.4f}")



jb.dump(model, "models/random_forest_model.pkl")
print("Modelo salvo em 'models/random_forest_model.pkl'")

jb.dump(scaler,"models/scaler.pkl")
print("tradutor scaler salvo")