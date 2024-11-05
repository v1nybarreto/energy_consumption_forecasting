import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D
import keras_tuner as kt
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Função para criar diretórios
def criar_diretorio(caminho):
    """Cria um diretório se ele não existir.
    
    Args:
        caminho (str): O caminho do diretório a ser criado.
    """
    if not os.path.exists(caminho):
        os.makedirs(caminho)
    elif not os.path.isdir(caminho):
        raise NotADirectoryError(f"{caminho} não é um diretório")

# Função para carregar e processar dados
def carregar_dados(file_path):
    """Carrega e processa o conjunto de dados de consumo de energia.
    
    Args:
        file_path (str): O caminho do arquivo CSV.

    Returns:
        pd.DataFrame: DataFrame processado.
        MinMaxScaler: Objeto scaler ajustado aos dados.
    """
    try:
        dados = pd.read_csv(file_path, sep=';')
    except FileNotFoundError:
        logging.error(f"Arquivo não encontrado: {file_path}")
        raise

    try:
        dados['Data'] = pd.to_datetime(dados['Data'], format='%d/%m/%Y')
        dados['Hora'] = pd.to_datetime(dados['Hora'], format='%H:%M').dt.time
        dados['Datetime'] = dados.apply(lambda row: pd.Timestamp(row['Data'].date()) + pd.Timedelta(hours=row['Hora'].hour, minutes=row['Hora'].minute), axis=1)
        dados = dados.drop(columns=['Data', 'Hora'])
        dados['Consumo'] = pd.to_numeric(dados['Consumo'], errors='coerce')
        dados = dados.dropna(subset=['Consumo'])
        dados = dados.sort_values('Datetime')
        dados['Hora'] = dados['Datetime'].dt.hour
        dados['Dia'] = dados['Datetime'].dt.day
        dados['Dia_da_Semana'] = dados['Datetime'].dt.dayofweek
        dados['Mes'] = dados['Datetime'].dt.month
        dados['Ano'] = dados['Datetime'].dt.year

        # Remover outliers
        Q1 = dados['Consumo'].quantile(0.25)
        Q3 = dados['Consumo'].quantile(0.75)
        IQR = Q3 - Q1
        outliers_iqr = dados[(dados['Consumo'] < (Q1 - 1.5 * IQR)) | (dados['Consumo'] > (Q3 + 1.5 * IQR))]
        media = dados['Consumo'].mean()
        desvio_padrao = dados['Consumo'].std()
        dados['Z-Score'] = (dados['Consumo'] - media) / desvio_padrao
        outliers_zscore = dados[(dados['Z-Score'] > 3) | (dados['Z-Score'] < -3)]
        outliers_combinados = pd.concat([outliers_iqr, outliers_zscore]).drop_duplicates()
        dados = dados[~dados.index.isin(outliers_combinados.index)]

        # Normalizar os dados de consumo
        scaler = MinMaxScaler()
        dados['Consumo'] = scaler.fit_transform(dados[['Consumo']])
        
        return dados, scaler
    except Exception as e:
        logging.error(f"Erro ao processar dados: {e}")
        raise

# Função para criar sequências de tempo
def criar_sequencias(dados, seq_length):
    """Cria sequências de tempo com informações de sazonalidade.
    
    Args:
        dados (pd.DataFrame): DataFrame com os dados de entrada.
        seq_length (int): Comprimento da sequência.

    Returns:
        np.array, np.array: Arrays com as sequências de entrada (X) e os valores alvo (y).
    """
    xs = []
    ys = []
    for i in range(len(dados) - seq_length - 4):
        x = dados.iloc[i:(i + seq_length)][['Consumo', 'Hora', 'Dia', 'Dia_da_Semana', 'Mes', 'Ano']].values
        y = dados.iloc[(i + seq_length):(i + seq_length + 4)]['Consumo'].values
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Função para construir modelos LSTM
def construir_modelo_lstm(hp):
    """Constrói um modelo LSTM com hiperparâmetros ajustáveis.
    
    Args:
        hp (kt.HyperParameters): Hiperparâmetros ajustáveis.

    Returns:
        keras.Model: Modelo LSTM compilado.
    """
    modelo = Sequential()
    modelo.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32), return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    modelo.add(Dropout(hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))
    modelo.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32)))
    modelo.add(Dense(4, activation='linear'))
    modelo.compile(optimizer='adam', loss='mse')
    return modelo

# Função para construir modelos GRU
def construir_modelo_gru(hp):
    """Constrói um modelo GRU com hiperparâmetros ajustáveis.
    
    Args:
        hp (kt.HyperParameters): Hiperparâmetros ajustáveis.

    Returns:
        keras.Model: Modelo GRU compilado.
    """
    modelo = Sequential()
    modelo.add(GRU(units=hp.Int('units', min_value=32, max_value=512, step=32), return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    modelo.add(Dropout(hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))
    modelo.add(GRU(units=hp.Int('units', min_value=32, max_value=512, step=32)))
    modelo.add(Dense(4, activation='linear'))
    modelo.compile(optimizer='adam', loss='mse')
    return modelo

# Função para construir modelos CNN-LSTM
def construir_modelo_cnn_lstm(hp):
    """Constrói um modelo CNN-LSTM com hiperparâmetros ajustáveis.
    
    Args:
        hp (kt.HyperParameters): Hiperparâmetros ajustáveis.

    Returns:
        keras.Model: Modelo CNN-LSTM compilado.
    """
    modelo = Sequential()
    modelo.add(Conv1D(filters=hp.Int('filters', min_value=32, max_value=128, step=16), kernel_size=hp.Choice('kernel_size', values=[3, 5, 7]), activation='relu', input_shape=(X.shape[1], X.shape[2])))
    modelo.add(MaxPooling1D(pool_size=2))
    modelo.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32)))
    modelo.add(Dense(4, activation='linear'))
    modelo.compile(optimizer='adam', loss='mse')
    return modelo

# Função para avaliar o modelo
def avaliar_modelo(modelo, X_teste, y_teste, scaler):
    """Avalia o modelo com métricas de desempenho.
    
    Args:
        modelo (keras.Model): O modelo treinado.
        X_teste (np.array): Dados de teste.
        y_teste (np.array): Valores reais de teste.
        scaler (MinMaxScaler): Objeto scaler para inversão da normalização.

    Returns:
        tuple: MSE, MAE e MAPE das previsões.
    """
    previsoes = modelo.predict(X_teste)
    previsoes_inversas = scaler.inverse_transform(previsoes)
    y_teste_inverso = scaler.inverse_transform(y_teste)
    
    mse = mean_squared_error(y_teste_inverso, previsoes_inversas)
    mae = mean_absolute_error(y_teste_inverso, previsoes_inversas)
    mape = np.mean(np.abs((y_teste_inverso - previsoes_inversas) / y_teste_inverso)) * 100
    
    return mse, mae, mape

# Função para plotar previsões
def plotar_previsoes(previsoes, proximas_horas):
    """Plota as previsões de consumo de energia para as próximas horas.
    
    Args:
        previsoes (dict): Dicionário com as previsões dos modelos.
        proximas_horas (list): Lista com as horas futuras para o eixo x.
    """
    plt.figure(figsize=(14, 7))
    for nome, previsao in previsoes.items():
        plt.plot(proximas_horas, previsao, label=f'Previsão {nome}', marker='o')
    plt.xlabel('Hora')
    plt.ylabel('Consumo de Energia')
    plt.title('Previsões de Consumo de Energia para as Próximas 4 Horas')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Função para salvar modelos
def salvar_modelo(modelo, nome_arquivo):
    """Salva o modelo treinado em um arquivo.
    
    Args:
        modelo (keras.Model): O modelo treinado.
        nome_arquivo (str): O nome do arquivo para salvar o modelo.
    """
    modelo.save(nome_arquivo)
    logging.info(f"Modelo salvo como {nome_arquivo}")

# Função para carregar modelos
def carregar_modelo(nome_arquivo):
    """Carrega um modelo treinado de um arquivo.
    
    Args:
        nome_arquivo (str): O nome do arquivo do modelo a ser carregado.

    Returns:
        keras.Model: O modelo carregado.
    """
    from keras.models import load_model
    modelo = load_model(nome_arquivo)
    logging.info(f"Modelo carregado de {nome_arquivo}")
    return modelo

# Exemplo de uso das funções
base_dir = '/mnt/data/energy_consumption_tuning'
criar_diretorio(base_dir)

file_path = '/mnt/data/Historico_de_Consumo_2019_2020.csv'
dados, scaler = carregar_dados(file_path)
seq_length = 24
X, y = criar_sequencias(dados, seq_length)

# Configuração do tuner
tuner_lstm = kt.Hyperband(construir_modelo_lstm, objective='val_loss', max_epochs=10, factor=3, directory=os.path.join(base_dir, 'energy_consumption_lstm'), project_name='energy_consumption_lstm')
tuner_gru = kt.Hyperband(construir_modelo_gru, objective='val_loss', max_epochs=10, factor=3, directory=os.path.join(base_dir, 'energy_consumption_gru'), project_name='energy_consumption_gru')
tuner_cnn_lstm = kt.Hyperband(construir_modelo_cnn_lstm, objective='val_loss', max_epochs=10, factor=3, directory=os.path.join(base_dir, 'energy_consumption_cnn_lstm'), project_name='energy_consumption_cnn_lstm')

# Validação cruzada em série temporal
tscv = TimeSeriesSplit(n_splits=5)
resultados = {'lstm': [], 'gru': [], 'cnn_lstm': []}

for train_index, test_index in tscv.split(X):
    X_treino, X_teste = X[train_index], X[test_index]
    y_treino, y_teste = y[train_index], y[test_index]
    
    for tuner, nome in zip([tuner_lstm, tuner_gru, tuner_cnn_lstm], ['lstm', 'gru', 'cnn_lstm']):
        tuner.search(X_treino, y_treino, epochs=10, validation_data=(X_teste, y_teste), verbose=2)
        melhor_modelo = tuner.get_best_models(num_models=1)[0]
        perda = melhor_modelo.evaluate(X_teste, y_teste)
        resultados[nome].append(perda)

# Obter a média das perdas de validação para cada modelo
perda_media_lstm = np.mean(resultados['lstm'])
perda_media_gru = np.mean(resultados['gru'])
perda_media_cnn_lstm = np.mean(resultados['cnn_lstm'])

logging.info(f"Perda Média de Validação LSTM: {perda_media_lstm}")
logging.info(f"Perda Média de Validação GRU: {perda_media_gru}")
logging.info(f"Perda Média de Validação CNN-LSTM: {perda_media_cnn_lstm}")

# Obter os melhores modelos
melhor_modelo_lstm = tuner_lstm.get_best_models(num_models=1)[0]
melhor_modelo_gru = tuner_gru.get_best_models(num_models=1)[0]
melhor_modelo_cnn_lstm = tuner_cnn_lstm.get_best_models(num_models=1)[0]

# Avaliar e comparar os modelos com métricas adicionais
mse_lstm, mae_lstm, mape_lstm = avaliar_modelo(melhor_modelo_lstm, X_teste, y_teste, scaler)
mse_gru, mae_gru, mape_gru = avaliar_modelo(melhor_modelo_gru, X_teste, y_teste, scaler)
mse_cnn_lstm, mae_cnn_lstm, mape_cnn_lstm = avaliar_modelo(melhor_modelo_cnn_lstm, X_teste, y_teste, scaler)

logging.info(f"LSTM - MSE: {mse_lstm}, MAE: {mae_lstm}, MAPE: {mape_lstm}")
logging.info(f"GRU - MSE: {mse_gru}, MAE: {mae_gru}, MAPE: {mape_gru}")
logging.info(f"CNN-LSTM - MSE: {mse_cnn_lstm}, MAE: {mae_cnn_lstm}, MAPE: {mape_cnn_lstm}")

# Obter as últimas 24 horas de dados para criar a sequência de entrada para a previsão
X_ultimas = X[-1].reshape(1, seq_length, -1)

# Gerar previsões
previsoes = {}
for nome, modelo in {'LSTM': melhor_modelo_lstm, 'GRU': melhor_modelo_gru, 'CNN-LSTM': melhor_modelo_cnn_lstm}.items():
    previsao = modelo.predict(X_ultimas)
    previsoes[nome] = scaler.inverse_transform(previsao).flatten()
    logging.info(f"Previsões do modelo {nome} para as próximas 4 horas: {previsoes[nome]}")

# Definir as próximas horas para o eixo x
proximas_horas = ['00:00', '01:00', '02:00', '03:00']

# Plotar previsões
plotar_previsoes(previsoes, proximas_horas)

# Salvar os melhores modelos
salvar_modelo(melhor_modelo_lstm, os.path.join(base_dir, 'melhor_modelo_lstm.h5'))
salvar_modelo(melhor_modelo_gru, os.path.join(base_dir, 'melhor_modelo_gru.h5'))
salvar_modelo(melhor_modelo_cnn_lstm, os.path.join(base_dir, 'melhor_modelo_cnn_lstm.h5'))