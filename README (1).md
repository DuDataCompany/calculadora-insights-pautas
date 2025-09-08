# Temperatura da Pauta & Forecast (Streamlit)

App em Streamlit para:
- Calcular a **temperatura da pauta** (0–100) com análise de drivers e gauge.
- Gerar **forecast** de conversas a partir de um **CSV**, escolhendo automaticamente entre Holt-Winters e Regressão Linear + dummies de dia da semana.

## 🚀 Como rodar localmente

```bash
# 1) Clone o repositório e entre na pasta
git clone <seu-fork-ou-repo>.git
cd streamlit-temperatura-pauta

# 2) Crie e ative um ambiente virtual (opcional, recomendado)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3) Instale dependências
pip install -r requirements.txt

# 4) Rode o app
streamlit run app.py
```

## ☁️ Deploy no Streamlit Cloud

1. Suba este código no seu GitHub (visível ao Streamlit Cloud).
2. Em https://share.streamlit.io/ , crie um novo app apontando para o repo/branch e arquivo principal `app.py`.
3. Na primeira execução, o Streamlit instalará os pacotes do `requirements.txt`.

## 🧩 Estrutura

- `app.py` – código principal do app (duas abas: 🌡️ Temperatura da Pauta, 📈 Forecast (CSV)).
- `requirements.txt` – dependências.
- `README.md` – instruções e guia de deploy.
- `.gitignore` – padrões Python/venv/cache.

## 📄 Entrada do Forecast

- **CSV** com ao menos duas colunas: uma de **datas** e outra **numérica** (volume).
- Escolha as colunas via UI após upload.
- Marque `Datas no formato dia/mês/ano?` se for o padrão pt-BR (`dayfirst=True`).

## 🧪 Notas

- O gauge usa **cinza** (`#808080`) para a barra.
- Os quatro gauges adicionais (Menções, Engajamentos, Sentimento, Brandfit) mostram **progresso vs meta (0–100)**.
- O forecast usa holdout de 28 dias para escolher o melhor entre Holt-Winters e Regressão Linear com dummies de dia da semana.
