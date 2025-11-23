# API para Detectar Fraudes️

Solução de Machine Learning para detectar fraudes em cartões. O sistema funciona tanto para predições em tempo real quanto processamento de grandes volumes.

## ️Tecnologias utilizadas

* **Python 3**
* **FastAPI**
* **Scikit-learn**
* **Pandas**
* **Docker**

---

## Como rodar o projeto

### 1. Instalação

Instale as dependências do e baixe o dataset automaticamente:

```bash
pip install -r requirements.txt

python -m src.setup_data
```

### 2. Treinamento do modelo

Para gerar o artefato do modelo, execute:

```bash
python -m src.training
```

### 3. Iniciar a API

Para iniciar o servidor, execute:

```bash
uvicorn src.main:app --reload    
```

A API está documentada em: `https://localhost:8000/docs`

### Testar a API pelo Swagger:

1. Acesse `http://localhost:8000/docs`.
2. 
2. Selecione ``POST /predict``.

3. Clique em "Try it out".

4. Insira o JSON, exemplo:

```json
{
  "Time": 0,
  "Amount": 100.0,
  "V_features": {
    "V1": 0.5, "V2": 0.5, "V3": 0.5, "V4": 0.5, "V5": 0.5,
    "V6": 0.5, "V7": 0.5, "V8": 0.5, "V9": 0.5, "V10": 0.5,
    "V11": 0.5, "V12": 0.5, "V13": 0.5, "V14": 0.5, "V15": 0.5,
    "V16": 0.5, "V17": 0.5, "V18": 0.5, "V19": 0.5, "V20": 0.5,
    "V21": 0.5, "V22": 0.5, "V23": 0.5, "V24": 0.5, "V25": 0.5,
    "V26": 0.5, "V27": 0.5, "V28": 0.5
  }
}
```

### 4. Processamento em Lote (Batch)

Para processar um CSV grande, use o script batch.py:

```bash
python -m src.batch data/creditcard.csv predicoes_batch.csv
```

As predições serão salvas em `predicoes_batch.csv`.

### Testes Automatizados

Para rodar os testes, execute:

```bash
pytest
```

---

### 5. Docker

Para construir e rodar o container Docker, execute:

```bash
docker build -t mlproject .

docker run -p 8080:8080 mlproject
```







