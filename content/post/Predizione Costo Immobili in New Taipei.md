---
title: Predizione Costo Immobili in New Taipei
date: 2020-01-27
tags: ["markdown"]
image : "post/img/Taipei.jpg"
Description  : "Predizione del valore degli immobili tramite reti neurali artificiali..."
---
# Presentazione

Avvio presentazione fullscreen

<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vQQ9YFAeAOYVbj1g1othIDZNdbPMM19OZVNMqRoCbclZP2-STve8wKDzbFDEoTO26xFUEfSAJnl6p1q/embed?start=false&loop=false&delayms=3000" frameborder="0" width="820" height="498" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>

# Notebook

## 2\. Raccolta dati

## ► 2.1. Identificazione delle fonti dati

## ► 2.2. Selezione delle fonti dati

```
# installo il pachetto per aprire il file xlsx
!pip install openpyxl


import pandas as pd

# Adatto l'output stampato a schermo alla larghezza attuale della finestra
from IPython.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))
pd.set_option("display.width", 1000)

# Cambio la palette dei colori standard per adattarli alla palette del sito
import matplotlib

# definire i colori specificati dall'utente
colors = ["#0077b5", "#7cb82f", "#dd5143", "#00aeb3", "#8d6cab", "#edb220", "#262626"]

# cambiare la palette di colori di default
matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=colors)
```

```
Requirement already satisfied: openpyxl in c:\users\wolvi\documents\venv\lib\site-packages (3.0.10)
Requirement already satisfied: et-xmlfile in c:\users\wolvi\documents\venv\lib\site-packages (from openpyxl) (1.1.0)
```

## ► 2.3. Raccolta dei dati

```
# carico le librerie necessarie
import pandas as pd

# Scarico il dataset in formato xlsx e lo carico nel file
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00477/Real%20estate%20valuation%20data%20set.xlsx"
df = pd.read_excel(url)
```

## ► 2.4. Verifica della qualità dei dati

```
import pandas as pd

# Caricamento del dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00477/Real%20estate%20valuation%20data%20set.xlsx"
df = pd.read_excel(url)

# Trasformo la colonna "NO" in indice per il dataset
df = df.set_index("No")

# Traduzione in italiano del nome delle colonne
df.rename(
    columns={
        "X1 transaction date": "Data transazione",
        "X2 house age": "Età della casa",
        "X3 distance to the nearest MRT station": "Distanza MRT vicina",
        "X4 number of convenience stores": "Numero di discount vicini",
        "X5 latitude": "Latitudine",
        "X6 longitude": "Longitudine",
        "Y house price of unit area": "costo al m2",
    },
    inplace=True,
)

# Analisi delle informazioni sul dataset
print("Numero di righe e colonne:     ", df.shape)
print("════════════════════════════════════════════════════════════════════════")
# print("Nomi delle colonne:", df.columns)
print("Tipi di dati delle colonne:\n\n", df.dtypes.to_frame().T)
print("════════════════════════════════════════════════════════════════════════")
print("Valori mancanti nel dataset: \n\n", df.isnull().sum().to_frame().T)
print("════════════════════════════════════════════════════════════════════════")
# Gestione dei valori mancanti
# rimozione delle righe con valori mancanti
# df = df.dropna()
# o sostituzione dei valori mancanti con un valore specifico
# df = df.fillna(0)

df.head()
```

```
Numero di righe e colonne:      (414, 7)
════════════════════════════════════════════════════════════════════════
Tipi di dati delle colonne:

   Data transazione Età della casa Distanza MRT vicina Numero di discount vicini Latitudine Longitudine costo al m2
0          float64        float64             float64                     int64    float64     float64     float64
════════════════════════════════════════════════════════════════════════
Valori mancanti nel dataset: 

    Data transazione  Età della casa  Distanza MRT vicina  Numero di discount vicini  Latitudine  Longitudine  costo al m2
0                 0               0                    0                          0           0            0            0
════════════════════════════════════════════════════════════════════════
```

|  | Data transazione | Età della casa | Distanza MRT vicina | Numero di discount vicini | Latitudine | Longitudine | costo al m2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2012.916667 | 32.0 | 84.87882 | 10 | 24.98298 | 121.54024 | 37.9 |
| 2 | 2012.916667 | 19.5 | 306.59470 | 9 | 24.98034 | 121.53951 | 42.2 |
| 3 | 2013.583333 | 13.3 | 561.98450 | 5 | 24.98746 | 121.54391 | 47.3 |
| 4 | 2013.500000 | 13.3 | 561.98450 | 5 | 24.98746 | 121.54391 | 54.8 |
| 5 | 2012.833333 | 5.0 | 390.56840 | 5 | 24.97937 | 121.54245 | 43.1 |

## ► 2.5. Archiviazione dei dati

```
# Salvataggio del dataset pulito
df.to_csv("dataset_pulito.csv", index=False)
```

## ► 2.6. Documentazione dei dati

La Documentazione dei dati è data, in questo caso dal salvataggio su github del notebook di lavoro, ordinato per sottosezioni, con ulteriori commenti all’interno del codice.

## 3\. Pulizia dei dati

## ► 3.1. Analisi iniziale dei dati

## ► 3.2. Trasformazione dei dati

## ► 3.3. Normalizzazione dei dati

## ► 3.4. Creazione di nuove variabili(non serve in questo caso)

## ► 3.5. Documentazione dei dati

## 4\. Esplorazione dei dati

## ► 4.1 Analisi univariata(unico output è “costo al m2”)

```
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

# Grafici in formato svg


# Analisi della distribuzione dei dati
print("Statistiche descrittive:\n\n", df.describe())
print("════════════════════════════════════════════════════════════════════════")
print("Valori unici per ogni colonna:\n\n", df.nunique().to_frame().T)
print("════════════════════════════════════════════════════════════════════════")
# Correzione dei tipi di dati
#df["colonna_specifica"] = df["colonna_specifica"].astype(int)

# visualizzare la distribuzione dei valori per ogni colonna
df.hist(bins=50, figsize=(20,15))
plt.suptitle("Distribuzione dei valori per ogni colonna \n", fontsize=24)
plt.tight_layout()
plt.show()

# visualizzare la relazione tra costo al m2 e altre colonne
X = df[['Data transazione', 'Età della casa', 'Distanza MRT vicina', 'Numero di discount vicini', 'Latitudine', 'Longitudine']]
Y = df[['costo al m2']]

# creare una griglia di sottografici 2x3 adatta alla larghezza della pagina
fig, axs = plt.subplots(2, 3, figsize=(20, 10))

# tracciare un grafico scatterplot per ogni colonna di X in ogni sottografico
for i in range(2):
    for j in range(3):
        if i*3+j < len(X.columns):
            axs[i, j].scatter(X[X.columns[i*3+j]], Y)
            axs[i, j].set_xlabel(X.columns[i*3+j])
            axs[i, j].set_ylabel('costo al m2')
            
plt.suptitle("Grafico di Dispersione relazionati alla funzione di output", fontsize=24)
plt.tight_layout()
plt.show()

# Crea una figura con griglia 2x3
fig, axs = plt.subplots(2, 3, figsize=(15,10))
'Data transazione', 'Età della casa', 'Distanza MRT vicina', 'Numero di discount vicini', 'Latitudine', 'Longitudine'
# 1. Disegna il boxplot per la colonna 'Data transazione'
df.boxplot(column=["costo al m2"], by='Data transazione', ax=axs[0][0], grid=False)
axs[0][0].set_title('Data transazione')
axs[0][0].xaxis.set_major_locator(MaxNLocator(nbins = 10,min_n_ticks=10))
axs[0][0].xaxis.set_tick_params(rotation=30)


# 2. Disegna il boxplot per la colonna 'Età della casa'
df.boxplot(column=["costo al m2"], by='Età della casa', ax=axs[0][1], grid=False)
axs[0][1].set_title('Età della casa')
axs[0][1].xaxis.set_major_locator(MaxNLocator(nbins = 10,min_n_ticks=10))
axs[0][1].xaxis.set_tick_params(rotation=30)

# 3. Disegna il boxplot per la colonna 'Distanza MRT vicina'
df.boxplot(column=["costo al m2"], by='Distanza MRT vicina', ax=axs[0][2], grid=False)
axs[0][2].set_title('Distanza MRT vicina')
axs[0][2].xaxis.set_major_locator(MaxNLocator(nbins = 10,min_n_ticks=10))
axs[0][2].xaxis.set_tick_params(rotation=30)

# 4. Disegna il boxplot per la colonna 'Numero di discount vicini'
df.boxplot(column=["costo al m2"], by='Numero di discount vicini', ax=axs[1][0], grid=False)
axs[1][0].set_title('Numero di discount vicini')
axs[1][0].xaxis.set_major_locator(MaxNLocator(nbins = 10,min_n_ticks=10))
axs[1][0].xaxis.set_tick_params(rotation=30)

# 5. Disegna il boxplot per la colonna 'Latitudine'
df.boxplot(column=["costo al m2"], by='Latitudine', ax=axs[1][1], grid=False)
axs[1][1].set_title('Latitudine')
axs[1][1].xaxis.set_major_locator(MaxNLocator(nbins = 10,min_n_ticks=10))
axs[1][1].xaxis.set_tick_params(rotation=30)

# 6. Disegna il boxplot per la colonna  'Longitudine'
df.boxplot(column=["costo al m2"], by= 'Longitudine', ax=axs[1][2], grid=False)
axs[1][2].set_title('Longitudine')
axs[1][2].xaxis.set_major_locator(MaxNLocator(nbins = 10,min_n_ticks=10))
axs[1][2].xaxis.set_tick_params(rotation=30)

# Inserisce un titolo per la griglia di plot
plt.suptitle("Analisi statistica del costo al m2 in relazione alle altre caratteristiche", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.show()


# Calcola la correlazione lineare tra le colonne
corr_matrix = df.corr()

# Crea una figura con due sottografici
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Crea una heatmap della matrice di correlazione lineare
cmap = plt.cm.get_cmap('Spectral', 256)
im1 = ax1.imshow(corr_matrix, cmap=cmap)
ax1.set_xticks(np.arange(corr_matrix.shape[1]))
ax1.set_yticks(np.arange(corr_matrix.shape[0]))
ax1.set_xticklabels(corr_matrix.columns, rotation=15, ha='right')
ax1.set_yticklabels(corr_matrix.index)
ax1.set_title("Grafico di Correlazione Lineare fra le variabili \n")
plt.colorbar(im1,ax=ax1)

# Calcola la correlazione non lineare tra le colonne
corr_matrix_non_lineare = df[["Data transazione", "Età della casa", "Distanza MRT vicina", "Numero di discount vicini", "Latitudine", "Longitudine","costo al m2"]].corr(method ='spearman')

# Crea una heatmap della matrice di correlazione non lineare
im2 = ax2.imshow(corr_matrix_non_lineare, cmap=cmap)
ax2.set_xticks(np.arange(corr_matrix_non_lineare.shape[1]))
ax2.set_yticks(np.arange(corr_matrix_non_lineare.shape[0]))
ax2.set_xticklabels(corr_matrix_non_lineare.columns, rotation=15, ha='right')
ax2.set_yticklabels(corr_matrix_non_lineare.index)
ax2.set_title("Grafico di Correlazione non Lineare fra le variabili \n")
plt.colorbar(im2,ax=ax2)

plt.suptitle("Grafici di Correlazione fra le variabili", fontsize=16)
plt.tight_layout()
plt.show()
```

```
Statistiche descrittive:

        Data transazione  Età della casa  Distanza MRT vicina  Numero di discount vicini  Latitudine  Longitudine  costo al m2
count        414.000000      414.000000           414.000000                 414.000000  414.000000   414.000000   414.000000
mean        2013.148953       17.712560          1083.885689                   4.094203   24.969030   121.533361    37.980193
std            0.281995       11.392485          1262.109595                   2.945562    0.012410     0.015347    13.606488
min         2012.666667        0.000000            23.382840                   0.000000   24.932070   121.473530     7.600000
25%         2012.916667        9.025000           289.324800                   1.000000   24.963000   121.528085    27.700000
50%         2013.166667       16.100000           492.231300                   4.000000   24.971100   121.538630    38.450000
75%         2013.416667       28.150000          1454.279000                   6.000000   24.977455   121.543305    46.600000
max         2013.583333       43.800000          6488.021000                  10.000000   25.014590   121.566270   117.500000
════════════════════════════════════════════════════════════════════════
Valori unici per ogni colonna:

    Data transazione  Età della casa  Distanza MRT vicina  Numero di discount vicini  Latitudine  Longitudine  costo al m2
0                12             236                  259                         11         234          232          270
════════════════════════════════════════════════════════════════════════
```

[![](https://automataia.github.io/AutoGnosis/4-progetti/4.1-supervised-learning/4.1.1-regressione-non-lineare/notebook-di-lavoro/out.png "Introduzione")](https://automataia.github.io/AutoGnosis/4-progetti/4.1-supervised-learning/4.1.1-regressione-non-lineare/notebook-di-lavoro/out.png)

[![](https://automataia.github.io/AutoGnosis/4-progetti/4.1-supervised-learning/4.1.1-regressione-non-lineare/notebook-di-lavoro/automata/content/4-Progetti/4.1-Supervised-Learning/4.1.1-Regressione-non-lineare/Notebook-di-lavoro/output_11_1.png)](https://automataia.github.io/AutoGnosis/4-progetti/4.1-supervised-learning/4.1.1-regressione-non-lineare/notebook-di-lavoro/automata/content/4-Progetti/4.1-Supervised-Learning/4.1.1-Regressione-non-lineare/Notebook-di-lavoro/output_11_1.png)

[![](https://automataia.github.io/AutoGnosis/4-progetti/4.1-supervised-learning/4.1.1-regressione-non-lineare/notebook-di-lavoro/automata/content/4-Progetti/4.1-Supervised-Learning/4.1.1-Regressione-non-lineare/Notebook-di-lavoro/output_11_2.png)](https://automataia.github.io/AutoGnosis/4-progetti/4.1-supervised-learning/4.1.1-regressione-non-lineare/notebook-di-lavoro/automata/content/4-Progetti/4.1-Supervised-Learning/4.1.1-Regressione-non-lineare/Notebook-di-lavoro/output_11_2.png)

[![](https://automataia.github.io/AutoGnosis/output_11_3.png)](https://automataia.github.io/AutoGnosis/output_11_3.png)

[![](https://automataia.github.io/AutoGnosis/4-progetti/4.1-supervised-learning/4.1.1-regressione-non-lineare/notebook-di-lavoro/automata/content/4-Progetti/4.1-Supervised-Learning/4.1.1-Regressione-non-lineare/Notebook-di-lavoro/output_11_4.png)](https://automataia.github.io/AutoGnosis/4-progetti/4.1-supervised-learning/4.1.1-regressione-non-lineare/notebook-di-lavoro/automata/content/4-Progetti/4.1-Supervised-Learning/4.1.1-Regressione-non-lineare/Notebook-di-lavoro/output_11_4.png)

```
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

# normalizzazione di X
norm_scaler = MinMaxScaler()
X = norm_scaler.fit_transform(X)

# normalizzazione di y
norm_scaler = MinMaxScaler()
y = norm_scaler.fit_transform(y.values.reshape(-1,1))


# Crea un oggetto RandomForestRegressor
reg = RandomForestRegressor()

# Addestra il modello e seleziona le caratteristiche
reg.fit(X, y)

# Crea un Dataframe con l'importanza delle caratteristiche e i relativi nomi
importance = pd.DataFrame(data={'Feature': X.columns, 'Importance': reg.feature_importances_})

# Ordina i valori per importanza crescente
importance = importance.sort_values('Importance', ascending=False)

# Stampa i valori ordinati
print(importance)
```

```
                     Feature  Importance
2        Distanza MRT vicina    0.583947
1             Età della casa    0.176832
4                 Latitudine    0.091936
5                Longitudine    0.086302
0           Data transazione    0.039450
3  Numero di discount vicini    0.021535
```

-   Analisi predittiva dei dati

```
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd

# Crea un oggetto GradientBoostingRegressor
reg = GradientBoostingRegressor()

# Addestra il modello e seleziona le caratteristiche
reg.fit(X, y)

# Crea un Dataframe con l'importanza delle caratteristiche e i relativi nomi
importance = pd.DataFrame(data={'Feature': X.columns, 'Importance': reg.feature_importances_})

# Ordina i valori per importanza crescente
importance = importance.sort_values('Importance', ascending=False)

# Stampa i valori ordinati
print(importance)
print("════════════════════════════════════════════════════════════════════════")
```

```
                     Feature  Importance
2        Distanza MRT vicina    0.608580
1             Età della casa    0.185034
4                 Latitudine    0.113694
5                Longitudine    0.059041
0           Data transazione    0.024951
3  Numero di discount vicini    0.008699
════════════════════════════════════════════════════════════════════════
```

## ► 4.7 Interpretazione e comunicazione dei risultati

I risultati mostrano che la feature più importante per il costo al metro quadro è la “Distanza MRT vicina”, con un peso del 60,86%. La seconda feature più importante è “Età della casa” con il 18,50%, seguita da “Latitudine” con l'11,37%. Le feature “Longitudine”, “Data transazione” e “Numero di discount vicini” hanno importanze rispettivamente del 5,90%, 2,50% e 0,87%. Si può dedurre che l’ubicazione e la vicinanza ai mezzi di trasporto pubblico sono fattori chiave per determinare il costo al metro quadro di un immobile, seguite dall’età della casa e dalla latitudine.

## 5\. Modellizzazione

## ► 5.1 Selezione del modello

## ► 5.2 Preparazione dei dati

## ► 5.3 Allenamento del modello

-   USO DEL MODELLO **RandomForestRegressor** ══════════════════════════════════════════════════════════

```
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# definisci le colonne come input
X = df[['Data transazione', 'Età della casa', 'Distanza MRT vicina', 'Numero di discount vicini', 'Latitudine', 'Longitudine']]

# definisci la colonna di output
y = df['costo al m2']

# normalizzazione di X e y
#norm_scaler = MinMaxScaler()
#X = norm_scaler.fit_transform(X)
#y = norm_scaler.fit_transform(y.values.reshape(-1,1))

# standardizzazione di X e y
stand_scaler = StandardScaler()
X = stand_scaler.fit_transform(X)
y = stand_scaler.fit_transform(y.values.reshape(-1,1))

# dividi il dataset in train e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Reshape da matrice a vettore dell output y
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

# crea l'oggetto del modello
rf = RandomForestRegressor()

# addestra il modello sui dati di addestramento
rf.fit(X_train, y_train)

# fai le previsioni sui dati di test
y_pred = rf.predict(X_test)

# valuta il modello
score = rf.score(X_test, y_test)
print("Accuratezza del modello: ", score)
```

```
Accuratezza del modello:  0.7116968032216016
```

-   USO DEL MODELLO **GradientBoostingRegressor** ══════════════════════════════════════════════════════════

```
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# definisci le colonne come input
X = df[['Data transazione', 'Età della casa', 'Distanza MRT vicina', 'Numero di discount vicini', 'Latitudine', 'Longitudine']]

# definisci la colonna di output
y = df['costo al m2']

# normalizzazione di X e y
#norm_scaler = MinMaxScaler()
#X = norm_scaler.fit_transform(X)
#y = norm_scaler.fit_transform(y.values.reshape(-1,1))

# standardizzazione di X e y
stand_scaler = StandardScaler()
X = stand_scaler.fit_transform(X)
y = stand_scaler.fit_transform(y.values.reshape(-1,1))

# dividi il dataset in train e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Reshape da matrice a vettore dell output y
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

# crea l'oggetto del modello
gbr = GradientBoostingRegressor()
# addestra il modello sui dati di addestramento
gbr.fit(X_train, y_train)

# fai le previsioni sui dati di test
y_pred = gbr.predict(X_test)

# valuta il modello
score = rf.score(X_test, y_test)

print("Accuratezza del modello: ", score)
```

```
Accuratezza del modello:  0.916166216656722
```