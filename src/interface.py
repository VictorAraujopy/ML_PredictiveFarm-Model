import joblib as jb 
import numpy as np
import sys
import os


#define o caminho pra pasta models
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(script_path))

#define o caminho dos modelos
MODEL_PATH = os.path.join(project_root, "models", "random_forest_model.pkl")
SCALER_PATH = os.path.join(project_root, "models", "scaler.pkl")

print("--- 🤖 Assistente FarmTech Iniciando ---")

#vai tentar carregar os arquivos

try:
    model = jb.load(MODEL_PATH)
    scaler = jb.load(SCALER_PATH)
    print("✅ Modelo e Scaler carregados com sucesso!")
    print("------------------------------------------")

except FileNotFoundError:
    print("erro arquivos não encontrados")


def fazer_predicao(IntputsUser):
    #converter input do usuario para um array legivel pelo scaler
    input_array = np.array(IntputsUser).reshape(1, -1) 
    #aplica tradução
    input_scaled = scaler.transform(input_array)
    # faz o chute do modelo
    predicao_num = model.predict(input_scaled)

    return predicao_num[0]

if __name__ == "__main__":
    print("\n--- 🤖 Assistente de Recomendação Agrícola FarmTech ---")
    print("Insira os 7 valores do solo para receber a recomendação.")
    print("--------------------------------------------------")
    
    try:
        # Coleta os 7 inputs do usuário
        n = float(input("   1. Nível de Nitrogênio (N): "))
        p = float(input("   2. Nível de Fósforo (P): "))
        k = float(input("   3. Nível de Potássio (K): "))
        temp = float(input("   4. Temperatura (°C): "))
        umid = float(input("   5. Umidade do Ar (%): "))
        ph = float(input("   6. Nível de pH do solo: "))
        chuva = float(input("   7. Precipitação (mm): "))
        
        # Cria a lista de inputs na ordem correta
        dados_do_usuario = [n, p, k, temp, umid, ph, chuva]
        
        # Chama a função de predição que criamos no Bloco 2
        cultura_recomendada = fazer_predicao(dados_do_usuario)
        
        print("\n=======================================================")
        print(f"   🏆 A cultura ideal para estas condições é: >> {cultura_recomendada.upper()} <<")
        print("=======================================================")

    except ValueError:
        print("\n❌ Erro: Por favor, insira apenas números.")
    except KeyboardInterrupt:
        print("\n👋 Saindo do assistente...")


