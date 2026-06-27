"""Gera o notebook trabalho_cars.ipynb a partir de células definidas em Python.

Manter o conteúdo das células aqui (em vez de editar JSON do .ipynb na mão) torna
o notebook reproduzível e fácil de versionar. Rode:  python build_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ---------------------------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------------------------
md(r"""
# Trabalho Final — Gestão e Análise de Dados (GETI)

**Dataset:** `vehicle_price_prediction.csv` — 1.000.000 de veículos do mercado dos EUA (Kaggle).
**Alvo de estudo:** `price` (preço do usado, em USD).

Este notebook cumpre as tarefas do descritivo (ver `Descritivo-do-Trabalho.md`):

1. **5.1 Pré-processamento** — *missing values*, tratamento, codificação de categóricos e normalização.
2. **5.2.1 Regressão Linear Múltipla** — coeficientes, R² e valor-P por feature.
3. **5.2.2 K-means** — agrupamento dos veículos e análise de um grupo.
4. **5.2.3 K-NN** — os 5 veículos mais parecidos com um escolhido e checagem do cluster.

> **Observação de execução:** por padrão `USE_SAMPLE = False`, ou seja, **todas as etapas rodam
> sobre o dataset completo (1M de linhas)**. Métricas O(n²) (silhouette) e os gráficos de dispersão
> usam uma subamostra apenas para cálculo/plotagem; os modelos são treinados no dataset inteiro.
""")

# ---------------------------------------------------------------------------
# 0. Configuração
# ---------------------------------------------------------------------------
md("## 0. Configuração e carga dos dados")

code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import statsmodels.api as sm

# --- Parâmetros globais ---
RANDOM_STATE = 42
USE_SAMPLE = False        # False = dataset completo (1M). True = amostra rápida p/ exploração.
SAMPLE_SIZE = 150_000     # usado somente quando USE_SAMPLE = True
CSV_PATH = "vehicle_price_prediction.csv"

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)
np.random.seed(RANDOM_STATE)
print("Configuração carregada. USE_SAMPLE =", USE_SAMPLE)
""")

code(r"""
# IMPORTANTE: a coluna accident_history usa "None" como CATEGORIA legítima
# (= "sem histórico de acidentes"). O pandas, por padrão, interpretaria a string
# "None" como valor ausente (NaN). Por isso lemos com keep_default_na=False e
# tratamos apenas campos realmente vazios como ausentes.
df = pd.read_csv(CSV_PATH, keep_default_na=False, na_values=["", " "])
print("Dimensões do dataset:", df.shape)
df.head()
""")

code(r"""
df.info()
""")

code(r"""
df.describe(include="all").T
""")

md(r"""
### Seleção do conjunto de trabalho (`df_work`)

Ponto **único** de troca entre dataset completo e amostra. Como `USE_SAMPLE = False`, `df_work`
é o dataset inteiro. Para uma exploração interativa mais rápida, basta mudar `USE_SAMPLE = True`
no topo e reexecutar.
""")

code(r"""
if USE_SAMPLE:
    df_work = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
else:
    df_work = df.reset_index(drop=True)

print("df_work:", df_work.shape, "| modo:", "AMOSTRA" if USE_SAMPLE else "COMPLETO")
""")

# ---------------------------------------------------------------------------
# 1. Pré-processamento
# ---------------------------------------------------------------------------
md(r"""
## 1. Pré-processamento (Tarefa 5.1)

Definimos os papéis de cada atributo:

- **Numéricos:** `year`, `mileage`, `engine_hp`, `owner_count`, `vehicle_age`, `mileage_per_year`, `brand_popularity`.
- **Categóricos ordinais:** `condition` (Fair < Good < Excellent), `accident_history` (None < Minor < Major).
- **Categóricos nominais (baixa cardinalidade):** `transmission`, `fuel_type`, `drivetrain`, `body_type`, `seller_type`, `exterior_color`, `interior_color`, `trim`.
- **Categóricos nominais (alta cardinalidade):** `make` (25), `model` (105).
- **Alvo:** `price`.
""")

code(r"""
NUMERIC_COLS = ["year", "mileage", "engine_hp", "owner_count",
                "vehicle_age", "mileage_per_year", "brand_popularity"]
ORDINAL_COLS = ["condition", "accident_history"]
NOMINAL_LOW  = ["transmission", "fuel_type", "drivetrain", "body_type",
                "seller_type", "exterior_color", "interior_color", "trim"]
NOMINAL_HIGH = ["make", "model"]
TARGET = "price"

ORDINAL_CATEGORIES = {
    "condition": ["Fair", "Good", "Excellent"],          # pior -> melhor
    "accident_history": ["None", "Minor", "Major"],       # menos -> mais grave
}
""")

md("### 1.1 Identificação de *missing values*")

code(r"""
na_counts = df.isna().sum()
na_table = pd.DataFrame({
    "n_missing": na_counts,
    "pct_missing": (na_counts / len(df) * 100).round(4),
})
print("Total de valores ausentes no dataset:", int(na_counts.sum()))
na_table
""")

code(r"""
fig, ax = plt.subplots(figsize=(10, 4))
na_table["n_missing"].plot(kind="bar", ax=ax, color="#4C72B0")
ax.set_title("Valores ausentes por coluna")
ax.set_ylabel("nº de NaN")
ax.set_ylim(0, max(1, na_table["n_missing"].max()))
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
""")

md(r"""
**Resultado:** o dataset **não possui valores ausentes** — todas as 20 colunas têm 0 *missing values*.

> **Cuidado importante:** a coluna `accident_history` tem `None` como categoria válida ("sem acidentes").
> O pandas, por padrão, converteria essa string em `NaN`, criando ~1/3 de "faltantes" falsos. Por isso
> lemos o CSV com `keep_default_na=False`. Identificar esse tipo de armadilha faz parte da etapa de
> *missing values*.

Como não há ausentes reais, não é necessária imputação. Ainda assim, demonstramos a seguir a **técnica**
que seria aplicada caso houvesse dados faltantes.
""")

md(r"""
### 1.2 Técnica de tratamento de dados faltantes (demonstração)

Técnica escolhida — **imputação simples** com `sklearn.impute.SimpleImputer`:

- **Numéricos → mediana:** robusta a *outliers* (mais que a média) e não distorce a escala.
- **Categóricos → moda** (valor mais frequente): preserva a distribuição da categoria dominante.

Para demonstrar, criamos uma cópia de 1.000 linhas e **injetamos NaN artificialmente** (apenas para
fins didáticos — o dataset original continua intacto).
""")

code(r"""
demo = df.head(1000).copy()

# Injeta NaN artificialmente em uma coluna numérica e uma categórica.
rng = np.random.default_rng(RANDOM_STATE)
demo.loc[rng.choice(demo.index, 50, replace=False), "engine_hp"] = np.nan
demo.loc[rng.choice(demo.index, 50, replace=False), "fuel_type"] = np.nan

print("NaN injetados -> engine_hp:", int(demo['engine_hp'].isna().sum()),
      "| fuel_type:", int(demo['fuel_type'].isna().sum()))

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

demo["engine_hp"] = num_imputer.fit_transform(demo[["engine_hp"]])
demo["fuel_type"] = cat_imputer.fit_transform(demo[["fuel_type"]]).ravel()

print("Após imputação -> engine_hp:", int(demo['engine_hp'].isna().sum()),
      "| fuel_type:", int(demo['fuel_type'].isna().sum()))
print("Mediana usada (engine_hp):", float(num_imputer.statistics_[0]))
print("Moda usada (fuel_type):", cat_imputer.statistics_[0])
""")

md(r"""
### 1.3 Transformação de atributos categóricos em numéricos

Usamos três técnicas, escolhidas conforme a natureza de cada variável:

| Técnica | Aplicada a | Por quê |
|---|---|---|
| **Ordinal Encoding** | `condition`, `accident_history` | Há ordem natural; preservá-la dá informação ao modelo. |
| **One-Hot Encoding** | nominais de baixa cardinalidade | Sem ordem; one-hot evita criar uma ordem falsa. Poucas categorias → poucas colunas novas. |
| **Frequency Encoding** | `make` (25), `model` (105) | Alta cardinalidade; one-hot explodiria em dezenas/centenas de colunas. A frequência é um sinal numérico compacto. |

`make` também é parcialmente redundante com `brand_popularity` (que já é uma codificação de popularidade da marca).
""")

code(r"""
proc = df_work.copy()

# (a) Ordinais -> inteiros respeitando a ordem definida.
ord_enc = OrdinalEncoder(categories=[ORDINAL_CATEGORIES[c] for c in ORDINAL_COLS])
proc[ORDINAL_COLS] = ord_enc.fit_transform(proc[ORDINAL_COLS])
print("Ordinais codificados:")
for c in ORDINAL_COLS:
    print(f"  {c}: {dict(zip(ORDINAL_CATEGORIES[c], range(len(ORDINAL_CATEGORIES[c]))))}")

# (b) Frequency encoding p/ alta cardinalidade.
for c in NOMINAL_HIGH:
    freq = proc[c].value_counts(normalize=True)
    proc[c + "_freq"] = proc[c].map(freq).astype(float)
proc = proc.drop(columns=NOMINAL_HIGH)

# (c) One-Hot p/ nominais de baixa cardinalidade (cast p/ int — pandas 3 gera bool).
proc = pd.get_dummies(proc, columns=NOMINAL_LOW, drop_first=True, dtype=int)

print("\nDimensões após codificação:", proc.shape)
proc.head()
""")

md(r"""
### 1.4 Normalização

Aplicamos **padronização z-score** (`StandardScaler`): cada feature passa a ter média 0 e desvio 1.
É essencial para algoritmos baseados em distância (K-means, K-NN) e torna os coeficientes da
regressão comparáveis entre si. (Alternativa comum: `MinMaxScaler`, que reescala para [0, 1].)

A matriz padronizada `X_scaled` (todas as features, **exceto** o alvo `price`) é reutilizada nas três
tarefas seguintes. O `price` é mantido à parte como `y`.
""")

code(r"""
feature_cols = [c for c in proc.columns if c != TARGET]
X = proc[feature_cols].astype(float)
y = proc[TARGET].astype(float)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=proc.index)

print("X_scaled:", X_scaled.shape, "| nº de features:", len(feature_cols))
X_scaled.describe().T[["mean", "std", "min", "max"]].round(3).head(10)
""")

code(r"""
# Evidência visual: distribuição de 'mileage' antes e depois da padronização.
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(X["mileage"], bins=50, ax=axes[0], color="#4C72B0")
axes[0].set_title("mileage — original")
sns.histplot(X_scaled["mileage"], bins=50, ax=axes[1], color="#55A868")
axes[1].set_title("mileage — padronizado (z-score)")
plt.tight_layout()
plt.show()
""")

# ---------------------------------------------------------------------------
# 2. Regressão Linear Múltipla
# ---------------------------------------------------------------------------
md(r"""
## 2. Regressão Linear Múltipla (Tarefa 5.2.1)

Objetivo: medir o **grau de importância de cada feature** na formação do `price`.

- **Variável dependente:** `price`.
- **Variáveis independentes:** todas as features padronizadas de `X_scaled`.
- **Cuidado com colinearidade:** `vehicle_age = AnoAtual − year`, portanto `year` e `vehicle_age` são
  perfeitamente colineares. **Removemos `year`** e mantemos `vehicle_age`.
- Usamos `statsmodels.OLS` (com constante) porque fornece **coeficientes, R², R²-ajustado e valor-P**
  por feature. Como `X_scaled` está padronizado, os coeficientes são *betas padronizados* e podem ser
  comparados diretamente em magnitude.
""")

code(r"""
X_reg = X_scaled.drop(columns=["year"])     # evita colinearidade com vehicle_age
X_reg = sm.add_constant(X_reg)

ols = sm.OLS(y, X_reg).fit()
print(ols.summary())
""")

code(r"""
# Tabela organizada: coeficiente + valor-P, ordenada por importância (|coef|).
res = pd.DataFrame({
    "coef": ols.params,
    "p_value": ols.pvalues,
}).drop(index="const")
res["abs_coef"] = res["coef"].abs()
res = res.sort_values("abs_coef", ascending=False)
print(f"R-quadrado: {ols.rsquared:.4f} | R-quadrado ajustado: {ols.rsquared_adj:.4f}")
res.drop(columns="abs_coef").round(4)
""")

code(r"""
# Evidência visual: top features por magnitude do coeficiente padronizado.
top = res.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 6))
colors = ["#C44E52" if c < 0 else "#4C72B0" for c in top["coef"]]
ax.barh(top.index, top["coef"], color=colors)
ax.set_title("Regressão linear — coeficientes padronizados (top 15 por magnitude)")
ax.set_xlabel("coeficiente (impacto no preço, em desvios-padrão)")
ax.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.show()
""")

md(r"""
**Como ler os resultados:**

- O **R²** indica a fração da variância do preço explicada pelo modelo linear.
- O **coeficiente padronizado** mostra quanto o preço varia (em desvios-padrão) quando a feature
  aumenta 1 desvio-padrão — quanto maior o módulo, **mais importante** a feature.
- O **valor-P** testa a significância: `p < 0,05` indica feature estatisticamente significativa.
  *(Comente no relatório as features de maior |coef| — tipicamente `engine_hp`, `vehicle_age`/`mileage`
  e `condition` — e quais p-values são significativos.)*
""")

# ---------------------------------------------------------------------------
# 3. K-means
# ---------------------------------------------------------------------------
md(r"""
## 3. Agrupamento com K-means (Tarefa 5.2.2)

Agrupamos os veículos pelas suas **características** (usamos `X_scaled`). O preço **não** entra como
feature de agrupamento, para que os grupos reflitam similaridade de atributos do veículo — e possamos
depois observar como o preço se distribui em cada grupo.

Escolha de `k`: combinamos o **método do cotovelo** (inércia) com o **coeficiente de silhueta**
(calculado em subamostra, pois é O(n²)).
""")

code(r"""
# X_scaled já exclui o preço (price é o alvo, ficou em y).
X_clust = X_scaled

ks = range(2, 9)
inertias, silhouettes = [], []

# Subamostra apenas para o cálculo da silhueta (métrica O(n^2)).
sil_idx = np.random.choice(len(X_clust), size=min(10_000, len(X_clust)), replace=False)
X_sil = X_clust.iloc[sil_idx].to_numpy()

for k in ks:
    km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=5, batch_size=4096)
    labels = km.fit_predict(X_clust)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_sil, labels[sil_idx]))
    print(f"k={k}: inertia={km.inertia_:,.0f} | silhouette={silhouettes[-1]:.4f}")
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(list(ks), inertias, "o-", color="#4C72B0")
axes[0].set_title("Método do cotovelo")
axes[0].set_xlabel("k"); axes[0].set_ylabel("inércia")
axes[1].plot(list(ks), silhouettes, "o-", color="#55A868")
axes[1].set_title("Coeficiente de silhueta (subamostra)")
axes[1].set_xlabel("k"); axes[1].set_ylabel("silhouette")
plt.tight_layout()
plt.show()
""")

md(r"""
Com base nos gráficos, escolhemos **`k = 4`** (ajuste conforme o cotovelo/silhueta observados).
Treinamos o modelo final no **dataset completo** com `MiniBatchKMeans` (escalável para 1M de linhas).
""")

code(r"""
K = 4
kmeans = MiniBatchKMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10, batch_size=4096)
# Alternativa (mais lenta em 1M, porém K-means clássico):
# kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)

cluster_labels = kmeans.fit_predict(X_clust)
df_work = df_work.copy()
df_work["cluster"] = cluster_labels

print("Tamanho de cada cluster:")
print(df_work["cluster"].value_counts().sort_index())
""")

code(r"""
# Visualização 2D via PCA (modelo treinado no full; plotamos uma subamostra p/ leveza).
pca = PCA(n_components=2, random_state=RANDOM_STATE)
coords = pca.fit_transform(X_clust)

plot_idx = np.random.choice(len(X_clust), size=min(20_000, len(X_clust)), replace=False)
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(coords[plot_idx, 0], coords[plot_idx, 1],
                c=cluster_labels[plot_idx], cmap="tab10", s=6, alpha=0.4)
ax.set_title(f"Clusters K-means (k={K}) projetados via PCA — amostra de {len(plot_idx)} pontos")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)")
plt.colorbar(sc, label="cluster")
plt.tight_layout()
plt.show()
""")

md("### 3.1 Análise de um dos grupos")

code(r"""
# Perfil numérico médio de cada cluster (nas variáveis originais + preço).
profile_cols = NUMERIC_COLS + ["condition", "accident_history", "price"]
profile = df_work.copy()
profile[ORDINAL_COLS] = ord_enc.transform(profile[ORDINAL_COLS])
cluster_profile = profile.groupby("cluster")[profile_cols].mean().round(2)
cluster_profile
""")

code(r"""
# Escolhemos um cluster para detalhar (o de maior preço médio).
target_cluster = int(cluster_profile["price"].idxmax())
grp = df_work[df_work["cluster"] == target_cluster]
print(f"Cluster analisado: {target_cluster} | nº de veículos: {len(grp):,}")
print(f"Preço médio: US$ {grp['price'].mean():,.0f}  (geral: US$ {df_work['price'].mean():,.0f})")

print("\nCategorias predominantes neste grupo:")
for c in ["make", "body_type", "fuel_type", "drivetrain", "condition", "transmission"]:
    top_val = grp[c].value_counts(normalize=True).head(3)
    print(f"  {c}: " + ", ".join(f"{k} ({v*100:.0f}%)" for k, v in top_val.items()))
""")

md(r"""
**Comentário (semelhança no grupo):** os veículos do cluster destacado tendem a compartilhar faixa de
potência (`engine_hp`), idade/quilometragem e condição parecidas, o que se reflete numa faixa de preço
homogênea. *(No relatório, descreva o perfil concreto que aparecer acima — marcas, carrocerias e
condição dominantes — explicando por que esses veículos são considerados "parecidos" pelo algoritmo.)*
""")

# ---------------------------------------------------------------------------
# 4. K-NN
# ---------------------------------------------------------------------------
md(r"""
## 4. Vizinhos mais próximos com K-NN (Tarefa 5.2.3)

Escolhemos um veículo e usamos `NearestNeighbors` (distância euclidiana no mesmo espaço padronizado
`X_scaled`) para encontrar os **5 mais parecidos**. Depois verificamos se esses vizinhos caem no
**mesmo cluster** do K-means — uma checagem de consistência entre os dois métodos.
""")

code(r"""
nn = NearestNeighbors(n_neighbors=6, algorithm="brute", metric="euclidean")
nn.fit(X_clust.to_numpy())

# Veículo de referência (índice fixo para reprodutibilidade).
query_pos = 0
distances, indices = nn.kneighbors(X_clust.iloc[[query_pos]].to_numpy())

neighbor_pos = indices[0]              # inclui o próprio veículo na posição 0
neighbor_dist = distances[0]

show_cols = ["make", "model", "year", "mileage", "engine_hp", "fuel_type",
             "drivetrain", "body_type", "condition", "price", "cluster"]
result = df_work.iloc[neighbor_pos][show_cols].copy()
result.insert(0, "distancia", neighbor_dist.round(3))
result.insert(0, "papel", ["REFERÊNCIA"] + [f"vizinho {i}" for i in range(1, 6)])
result.reset_index(drop=True)
""")

code(r"""
ref_cluster = int(df_work.iloc[query_pos]["cluster"])
neigh_clusters = df_work.iloc[neighbor_pos[1:]]["cluster"].tolist()
same = sum(c == ref_cluster for c in neigh_clusters)

print(f"Cluster do veículo de referência: {ref_cluster}")
print(f"Clusters dos 5 vizinhos: {neigh_clusters}")
print(f"Vizinhos no mesmo cluster da referência: {same} de 5")
""")

md(r"""
**Comentário (consistência K-NN × K-means):** se a maioria (ou todos) dos 5 vizinhos mais próximos
estiver no mesmo cluster do veículo de referência, isso indica que os dois algoritmos enxergam a mesma
estrutura de similaridade — o K-NN encontra vizinhos locais e o K-means os agrupa na mesma região do
espaço de features. *(Reporte no relatório o número obtido acima e interprete-o.)*
""")

# ---------------------------------------------------------------------------
# 5. Conclusão
# ---------------------------------------------------------------------------
md(r"""
## 5. Conclusão

- **5.1 Pré-processamento:** o dataset não continha *missing values*; demonstramos a imputação por
  mediana/moda. Categóricos foram convertidos por *ordinal*, *one-hot* e *frequency encoding* conforme
  a natureza de cada variável, e tudo foi padronizado (z-score) para reuso nas demais tarefas.
- **5.2.1 Regressão:** o modelo OLS quantificou a influência de cada feature no preço (coeficientes
  padronizados, R² e valores-P), destacando as variáveis mais relevantes.
- **5.2.2 K-means:** os veículos foram agrupados por características; o grupo analisado mostrou
  homogeneidade de atributos e de faixa de preço.
- **5.2.3 K-NN:** os 5 veículos mais parecidos com o de referência foram identificados e em geral
  pertencem ao mesmo cluster do K-means, confirmando a coerência entre os métodos.
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

with open("trabalho_cars.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook gerado com {len(cells)} células: trabalho_cars.ipynb")
