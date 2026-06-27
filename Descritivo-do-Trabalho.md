# Gestão e Análise de Dados — GETI

**Professor:** André Luis Maciel Leme
**Instituição:** Instituto Federal de São Paulo — Câmpus Bragança Paulista
**Atividade:** Trabalho Final de Semestre

> Documento de referência gerado a partir de `Descritivo do Trabalho CARS.pdf` para consulta rápida durante o desenvolvimento.

---

## 1. Objetivo

- Através de prática sobre um dataset, este trabalho busca consolidar alguns dos conhecimentos sobre análise de dados apresentados ao longo do semestre.
- A atividade pode ser feita em **grupos de até 5 integrantes**. **Todos** devem fazer o upload do relatório.

## 2. O que entregar no Moodle

- O grupo deve entregar um **documento único em formato de relatório** (leiaute à sua escolha) com as evidências solicitadas no descritivo das tarefas.

## 3. Dataset

- O arquivo contém **1 milhão de registros** sobre carros no mercado dos Estados Unidos. Foi escolhido pelo volume de dados, tanto em instâncias quanto em features.
- A descrição dos atributos está no **Anexo** (ver seção final).
- Origem: Kaggle — https://www.kaggle.com/datasets/metawave/vehicle-price-prediction
- Arquivo local: `vehicle_price_prediction.csv`

## 4. Detalhamento da Atividade

- Consiste em abordagens de análise de dados mostradas na disciplina. **Sugestão:** usar o **Orange Data Mining**, mas outras ferramentas podem ser usadas.
- O resultado a entregar é um **relatório** demonstrando as tarefas realizadas com **evidências** (prints de tela, gráficos, etc.).
- Devem ser incluídos **comentários e descritivos** sobre a aplicação das técnicas e os resultados obtidos.

---

## 5. Tarefas com o Arquivo

### 5.1 Pré-processamento

- [ ] Identificar a existência de *missing values*.
- [ ] Descrever a técnica utilizada para resolver os casos de dados faltantes.
- [ ] Transformar os atributos categóricos em numéricos e **explicar a técnica** utilizada.
- [ ] Aplicar algum tipo de **normalização** nos dados transformados.

### 5.2 Algoritmos de Aprendizado de Máquina

#### 5.2.1 Regressão Linear Múltipla

- Objetivo: demonstrar o **grau de importância das features** na formação do preço (`price`) do veículo.
- A escolha de quais variáveis independentes serão usadas fica a seu critério.
- [ ] Mostrar o resultado da regressão: **coeficientes** encontrados, **R-Quadrado** e **valor-P** de cada feature.

#### 5.2.2 K-means

- Algoritmo de agrupamento que mostra como os cálculos classificaram as instâncias para formar os grupos.
- O **número de grupos** fica a seu critério.
- [ ] Separar **um dos grupos** e comentar a semelhança entre os veículos.

#### 5.2.3 K-NN

- [ ] Escolher um dos veículos e usar o K-NN para mostrar os **5 mais parecidos** com ele.
- [ ] Localizar esse veículo nos agrupamentos do item anterior (5.2.2) e verificar se os semelhantes estão **no mesmo cluster**.
- [ ] Mostrar os resultados no relatório.

---

## Anexo — Descrição dos Atributos do Dataset

| id | nome | descrição |
|----|------|-----------|
| 1 | `make` | Fabricante ou marca do veículo (ex.: Ford, Toyota). |
| 2 | `model` | Modelo específico do veículo (ex.: F-150, Camry). |
| 3 | `year` | Ano de fabricação do veículo. |
| 4 | `mileage` | Distância total percorrida pelo veículo, em milhas. |
| 5 | `engine_hp` | Potência (cavalos) do motor do veículo. |
| 6 | `transmission` | Tipo de transmissão (Automatic ou Manual). |
| 7 | `fuel_type` | Tipo de combustível (ex.: Gasoline, Diesel, Electric). |
| 8 | `drivetrain` | Tipo de tração (ex.: FWD, RWD, AWD). |
| 9 | `body_type` | Estilo de carroceria (ex.: SUV, Sedan, Pickup Truck). |
| 10 | `exterior_color` | Cor primária do exterior do veículo. |
| 11 | `interior_color` | Cor primária do interior do veículo. |
| 12 | `owner_count` | Número de donos anteriores do veículo. |
| 13 | `accident_history` | Histórico de acidentes registrado (None, Minor ou Major). |
| 14 | `seller_type` | Tipo de vendedor (Dealer ou Private). |
| 15 | `condition` | Condição geral do veículo (Excellent, Good ou Fair). |
| 16 | `trim` | Nível de acabamento (trim) específico do modelo. |
| 17 | `vehicle_age` | Idade do veículo em anos, calculada como `Current Year - year`. |
| 18 | `mileage_per_year` | Média de milhas rodadas por ano. |
| 19 | `brand_popularity` | Pontuação da popularidade da marca, baseada em sua frequência no dataset. |
| 20 | `price` | Preço (usado) do veículo em USD. **← variável alvo** |

### Resumo dos tipos de atributo

- **Categóricos:** `make`, `model`, `transmission`, `fuel_type`, `drivetrain`, `body_type`, `exterior_color`, `interior_color`, `accident_history`, `seller_type`, `condition`, `trim`
- **Numéricos:** `year`, `mileage`, `engine_hp`, `owner_count`, `vehicle_age`, `mileage_per_year`, `brand_popularity`, `price`
- **Alvo (target):** `price`
