---
title: Predizione correntisti morosi
date: 2020-02-27
tags: ["markdown"]
image : "post/img/Correntisti.jpg"
Description  : "Classificazione del profilo dei correntisti tipo di una banca..."
---
# Presentazione

[Avvio presentazione fullscreen](https://docs.google.com/presentation/d/e/2PACX-1vQInj2CAhdMwHSbBjnv-A-xMn2xA5YnqO2bM21SblUcv2coL92Up7Dp2eZh07X57Ll7AugpCwvXdTmP/pub?start=false&loop=false&delayms=3000)

# Notebook
## 2\. Raccolta dati

## ► 2.1. Identificazione delle fonti dati

## ► 2.2. Selezione delle fonti dati

Il dataset in questione contiene informazioni relative a diversi clienti di una banca. Ogni cliente è identificato da un **ID** univoco, e sono registrati diversi attributi a loro associati.

Tra questi attributi, troviamo il **LIMIT\_BAL**, ovvero l’importo di credito concesso al cliente, espresso in NT dollars. Viene inoltre registrato il **GENERE** del cliente, indicato tramite il valore 1 per il genere maschile e 2 per quello femminile.

Inoltre, viene riportato il livello di **ISTRUZIONE** raggiunto dal cliente, che può essere di diversi tipi, ovvero: graduate school (1), university (2), high school (3), others (4), unknown (5) o sconosciuto (6).

Viene inoltre registrato lo **STATO CIVILE** del cliente, indicato tramite il valore 1 per i clienti sposati, 2 per quelli single e 3 per quelli che hanno un altro stato civile.

Sono inoltre riportati l’**ETA’** del cliente in anni, e il suo **STATUS DI PAGAMENTO** per sei mesi consecutivi, dal mese di aprile al mese di settembre del 2005. Lo stato di pagamento viene indicato con valori compresi tra -1 e 9, dove -1 indica che il pagamento è stato effettuato regolarmente, mentre valori maggiori di 0 indicano un ritardo nei pagamenti.

Infine, vengono registrati i valori delle **FATTURE** emesse al cliente per i sei mesi in questione, espressi in NT dollar, e le **PAGAMENTI** effettuati dal cliente nel mese precedente per ciascuna delle fatture emesse.

L’ultimo attributo presente nel dataset è la colonna **default.payment.next.month**, che indica se il cliente ha effettuato il pagamento del mese successivo (1) o meno (0).

In sintesi, il dataset contiene informazioni dettagliate su diversi clienti di una banca, incluse informazioni relative al loro credito, alla loro situazione economica, all’età, allo stato civile e allo stato di pagamento dei loro debiti.

```
import matplotlib
import pandas as pd

# Adatto l'output stampato a schermo alla larghezza attuale della finestra
from IPython.display import HTML, display

display(HTML("<style>.container { width:100% !important; }</style>"))
pd.set_option("display.width", 1000)

# Cambio la palette dei colori standard per adattarli alla palette del sito

# definire i colori specificati dall'utente
colors = ["#0077b5", "#7cb82f", "#dd5143",
          "#00aeb3", "#8d6cab", "#edb220", "#262626"]

# cambiare la palette di colori di default
matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=colors)
```

## ► 2.3. Raccolta dei dati

```
import pandas as pd

# Carica il file CSV in un DataFrame(https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
df = pd.read_csv("UCI_Credit_Card.csv", sep=",", encoding="utf-8")

# Traduzione in italiano e accorciamento dei nomi delle colonne
#df.columns = [  'ID Titol','Saldo Res','Freq Sald','Acquisti','Acq Sing','Acq Rate','Antic Cont','Freq Acqu','Freq Sing','Freq Rate','Freq Antic','Trx Antic','Trx Acqu','Lim Credit','Pagamenti','Min Pagam','Prc Pagam','Durata Ser']


# Aggiunta del commento
df.metadata = {
"ID": "ID di ogni cliente",
"LIMIT_BAL": "Importo di credito concesso in dollari NT (include il credito individuale e familiare / supplementare)",
"SEX": "Genere (1=maschio, 2=femmina)",
"EDUCATION": "(1=scuola di specializzazione, 2=università, 3=scuola superiore, 4=altro, 5=sconosciuto, 6=sconosciuto)",
"MARRIAGE": "Stato civile (1=sposato, 2=single, 3=altro)",
"AGE": "Età in anni",
"PAY_0": "Stato di pagamento a settembre 2005 (-1=pagamento regolare, 1=ritardo di pagamento di un mese, 2=ritardo di pagamento di due mesi, ... 8=ritardo di pagamento di otto mesi, 9=ritardo di pagamento di nove mesi o più)",
"PAY_2": "Stato di pagamento ad agosto 2005 (scala come sopra)",
"PAY_3": "Stato di pagamento a luglio 2005 (scala come sopra)",
"PAY_4": "Stato di pagamento a giugno 2005 (scala come sopra)",
"PAY_5": "Stato di pagamento a maggio 2005 (scala come sopra)",
"PAY_6": "Stato di pagamento ad aprile 2005 (scala come sopra)",
"BILL_AMT1": "Importo della dichiarazione di fatturazione a settembre 2005 (dollari NT)",
"BILL_AMT2": "Importo della dichiarazione di fatturazione ad agosto 2005 (dollari NT)",
"BILL_AMT3": "Importo della dichiarazione di fatturazione a luglio 2005 (dollari NT)",
"BILL_AMT4": "Importo della dichiarazione di fatturazione a giugno 2005 (dollari NT)",
"BILL_AMT5": "Importo della dichiarazione di fatturazione a maggio 2005 (dollari NT)",
"BILL_AMT6": "Importo della dichiarazione di fatturazione ad aprile 2005 (dollari NT)",
"PAY_AMT1": "Importo del pagamento precedente a settembre 2005 (dollari NT)",
"PAY_AMT2": "Importo del pagamento precedente ad agosto 2005 (dollari NT)",
"PAY_AMT3": "Importo del pagamento precedente a luglio 2005 (dollari NT)",
"PAY_AMT4": "Importo del pagamento precedente a giugno 2005 (dollari NT)",
"PAY_AMT5": "Importo del pagamento precedente a maggio 2005 (dollari NT)",
"PAY_AMT6": "Importo del pagamento precedente ad aprile 2005 (dollari NT)",
"default.payment.next.month": "Pagamento predefinito (1=sì, 0=no)"
}


# Stampa i commenti descrittivi
#print(df.metadata)
# Stampa le prime 5 righe del DataFrame
print(df.head())
```

```
   ID  LIMIT_BAL  SEX  EDUCATION  MARRIAGE  AGE  PAY_0  PAY_2  PAY_3  PAY_4  ...  BILL_AMT4  BILL_AMT5  BILL_AMT6  PAY_AMT1  PAY_AMT2  PAY_AMT3  PAY_AMT4  PAY_AMT5  PAY_AMT6  default.payment.next.month
0   1    20000.0    2          2         1   24      2      2     -1     -1  ...        0.0        0.0        0.0       0.0     689.0       0.0       0.0       0.0       0.0                           1
1   2   120000.0    2          2         2   26     -1      2      0      0  ...     3272.0     3455.0     3261.0       0.0    1000.0    1000.0    1000.0       0.0    2000.0                           1
2   3    90000.0    2          2         2   34      0      0      0      0  ...    14331.0    14948.0    15549.0    1518.0    1500.0    1000.0    1000.0    1000.0    5000.0                           0
3   4    50000.0    2          2         1   37      0      0      0      0  ...    28314.0    28959.0    29547.0    2000.0    2019.0    1200.0    1100.0    1069.0    1000.0                           0
4   5    50000.0    1          2         1   57     -1      0     -1      0  ...    20940.0    19146.0    19131.0    2000.0   36681.0   10000.0    9000.0     689.0     679.0                           0

[5 rows x 25 columns]


C:\Users\wolvi\AppData\Local\Temp\ipykernel_9100\3669284972.py:11: UserWarning: Pandas doesn't allow columns to be created via a new attribute name - see https://pandas.pydata.org/pandas-docs/stable/indexing.html#attribute-access
  df.metadata = {
```

## ► 2.4. Verifica della qualità dei dati

```
import pandas as pd


# Analisi delle informazioni sul dataset
print("\033[1m" + "Numero di righe e colonne:     ".upper()+ "\033[0m", df.shape)
print("════════════════════════════════════════════════════════════════════════")
# print("Nomi delle colonne:", df.columns)
print("\033[1m" + "Tipi di dati delle colonne:\n\n".upper()+ "\033[0m", df.dtypes.to_frame().T)
print("════════════════════════════════════════════════════════════════════════")
print("\033[1m" + "Valori mancanti nel dataset: \n\n".upper()+ "\033[0m", df.isnull().sum().to_frame().T)
print("════════════════════════════════════════════════════════════════════════")
print("\033[1m" + "Numero di valori a zero: \n\n".upper()+ "\033[0m", (df == 0).sum().to_frame().T)
print("════════════════════════════════════════════════════════════════════════")
# Gestione dei valori mancanti
# rimozione delle righe con valori mancanti
# df = df.dropna()
# o sostituzione dei valori mancanti con un valore specifico
# df = df.fillna(0)
print("\n")


df.head()
```

```
[1mNUMERO DI RIGHE E COLONNE:     [0m (30000, 25)
════════════════════════════════════════════════════════════════════════
[1mTIPI DI DATI DELLE COLONNE:

[0m       ID LIMIT_BAL    SEX EDUCATION MARRIAGE    AGE  PAY_0  PAY_2  PAY_3  PAY_4  ... BILL_AMT4 BILL_AMT5 BILL_AMT6 PAY_AMT1 PAY_AMT2 PAY_AMT3 PAY_AMT4 PAY_AMT5 PAY_AMT6 default.payment.next.month
0  int64   float64  int64     int64    int64  int64  int64  int64  int64  int64  ...   float64   float64   float64  float64  float64  float64  float64  float64  float64                      int64

[1 rows x 25 columns]
════════════════════════════════════════════════════════════════════════
[1mVALORI MANCANTI NEL DATASET: 

[0m    ID  LIMIT_BAL  SEX  EDUCATION  MARRIAGE  AGE  PAY_0  PAY_2  PAY_3  PAY_4  ...  BILL_AMT4  BILL_AMT5  BILL_AMT6  PAY_AMT1  PAY_AMT2  PAY_AMT3  PAY_AMT4  PAY_AMT5  PAY_AMT6  default.payment.next.month
0   0          0    0          0         0    0      0      0      0      0  ...          0          0          0         0         0         0         0         0         0                           0

[1 rows x 25 columns]
════════════════════════════════════════════════════════════════════════
[1mNUMERO DI VALORI A ZERO: 

[0m    ID  LIMIT_BAL  SEX  EDUCATION  MARRIAGE  AGE  PAY_0  PAY_2  PAY_3  PAY_4  ...  BILL_AMT4  BILL_AMT5  BILL_AMT6  PAY_AMT1  PAY_AMT2  PAY_AMT3  PAY_AMT4  PAY_AMT5  PAY_AMT6  default.payment.next.month
0   0          0    0         14        54    0  14737  15730  15764  16455  ...       3195       3506       4020      5249      5396      5968      6408      6703      7173                       23364

[1 rows x 25 columns]
════════════════════════════════════════════════════════════════════════
```

|  | ID | LIMIT\_BAL | SEX | EDUCATION | MARRIAGE | AGE | PAY\_0 | PAY\_2 | PAY\_3 | PAY\_4 | ... | BILL\_AMT4 | BILL\_AMT5 | BILL\_AMT6 | PAY\_AMT1 | PAY\_AMT2 | PAY\_AMT3 | PAY\_AMT4 | PAY\_AMT5 | PAY\_AMT6 | default.payment.next.month |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 20000.0 | 2 | 2 | 1 | 24 | 2 | 2 | \-1 | \-1 | ... | 0.0 | 0.0 | 0.0 | 0.0 | 689.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 |
| 1 | 2 | 120000.0 | 2 | 2 | 2 | 26 | \-1 | 2 | 0 | 0 | ... | 3272.0 | 3455.0 | 3261.0 | 0.0 | 1000.0 | 1000.0 | 1000.0 | 0.0 | 2000.0 | 1 |
| 2 | 3 | 90000.0 | 2 | 2 | 2 | 34 | 0 | 0 | 0 | 0 | ... | 14331.0 | 14948.0 | 15549.0 | 1518.0 | 1500.0 | 1000.0 | 1000.0 | 1000.0 | 5000.0 | 0 |
| 3 | 4 | 50000.0 | 2 | 2 | 1 | 37 | 0 | 0 | 0 | 0 | ... | 28314.0 | 28959.0 | 29547.0 | 2000.0 | 2019.0 | 1200.0 | 1100.0 | 1069.0 | 1000.0 | 0 |
| 4 | 5 | 50000.0 | 1 | 2 | 1 | 57 | \-1 | 0 | \-1 | 0 | ... | 20940.0 | 19146.0 | 19131.0 | 2000.0 | 36681.0 | 10000.0 | 9000.0 | 689.0 | 679.0 | 0 |

5 rows × 25 columns

## ► 2.5. Archiviazione dei dati

```
# Salvataggio del dataset pulito
df.to_csv("04-dataset_pulito.csv", index=False)
```

## ► 2.6. Documentazione dei dati

La Documentazione dei dati è data, in questo caso dal salvataggio su github del notebook di lavoro, ordinato per sottosezioni, con ulteriori commenti all’interno del codice.

## 3\. Pulizia dei dati

## ► 3.1. Analisi iniziale dei dati

## ► 3.2. Trasformazione dei dati

```
import pandas as pd
import numpy as np

# sostituisci i valori mancanti con la mediana
df = df.fillna(df.median(numeric_only=True))
```

## ► 3.3. Normalizzazione dei dati

## ► 3.4. Creazione di nuove variabili

## ► 3.5. Documentazione dei dati

La Documentazione dei dati è data, in questo caso dal salvataggio su github del notebook di lavoro, ordinato per sottosezioni, con ulteriori commenti all’interno del codice.

## 4\. Esplorazione dei dati

## ► 4.1 Analisi multivariata

```
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

# Rimuovi  la prima colonna

df = df.iloc[:,1:]

# Trasforma i titoli delle colonne in minuscolo
df.columns = map(str.lower, df.columns)


# Analisi della distribuzione dei dati
print("Statistiche descrittive:\n\n", df.describe().round(0))
print("════════════════════════════════════════════════════════════════════════")
print("Valori unici per ogni colonna:\n\n", df.nunique().to_frame().T)
print("════════════════════════════════════════════════════════════════════════")
# Correzione dei tipi di dati
# df["colonna_specifica"] = df["colonna_specifica"].astype(int)

# visualizzare la distribuzione dei valori per ogni colonna
df.hist(bins=50, figsize=(20, 15))
plt.suptitle("Distribuzione dei valori per ogni colonna \n", fontsize=24)
plt.tight_layout()
plt.show()





# Calcola la correlazione lineare tra le colonne
corr_matrix = df.corr(numeric_only=True)

# Crea una figura con due sottografici
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Crea una heatmap della matrice di correlazione lineare
cmap = plt.cm.get_cmap("Spectral", 256)
im1 = ax1.imshow(corr_matrix, cmap=cmap,clim=(-1, 1))
ax1.set_xticks(np.arange(corr_matrix.shape[1]))
ax1.set_yticks(np.arange(corr_matrix.shape[0]))
ax1.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
ax1.set_yticklabels(corr_matrix.index)
ax1.set_title("Grafico di Correlazione Lineare fra le variabili \n")
plt.colorbar(im1, ax=ax1)

# Calcola la correlazione non lineare tra le colonne
corr_matrix_non_lineare = df.corr(method="spearman", numeric_only=True)

# Crea una heatmap della matrice di correlazione non lineare
im2 = ax2.imshow(corr_matrix_non_lineare, cmap=cmap,clim=(-1, 1))
ax2.set_xticks(np.arange(corr_matrix_non_lineare.shape[1]))
ax2.set_yticks(np.arange(corr_matrix_non_lineare.shape[0]))
ax2.set_xticklabels(corr_matrix_non_lineare.columns, rotation=45, ha="right")
ax2.set_yticklabels(corr_matrix_non_lineare.index)
ax2.set_title("Grafico di Correlazione non Lineare fra le variabili \n")
plt.colorbar(im2, ax=ax2)

plt.suptitle("Grafici di Correlazione fra le variabili", fontsize=16)
plt.tight_layout()
plt.show()
```

```
Statistiche descrittive:

            sex  education  marriage      age    pay_0    pay_2    pay_3    pay_4    pay_5    pay_6  ...  bill_amt4  bill_amt5  bill_amt6  pay_amt1   pay_amt2  pay_amt3  pay_amt4  pay_amt5  pay_amt6  default.payment.next.month
count  30000.0    30000.0   30000.0  30000.0  30000.0  30000.0  30000.0  30000.0  30000.0  30000.0  ...    30000.0    30000.0    30000.0   30000.0    30000.0   30000.0   30000.0   30000.0   30000.0                     30000.0
mean       2.0        2.0       2.0     35.0     -0.0     -0.0     -0.0     -0.0     -0.0     -0.0  ...    43263.0    40311.0    38872.0    5664.0     5921.0    5226.0    4826.0    4799.0    5216.0                         0.0
std        0.0        1.0       1.0      9.0      1.0      1.0      1.0      1.0      1.0      1.0  ...    64333.0    60797.0    59554.0   16563.0    23041.0   17607.0   15666.0   15278.0   17777.0                         0.0
min        1.0        0.0       0.0     21.0     -2.0     -2.0     -2.0     -2.0     -2.0     -2.0  ...  -170000.0   -81334.0  -339603.0       0.0        0.0       0.0       0.0       0.0       0.0                         0.0
25%        1.0        1.0       1.0     28.0     -1.0     -1.0     -1.0     -1.0     -1.0     -1.0  ...     2327.0     1763.0     1256.0    1000.0      833.0     390.0     296.0     252.0     118.0                         0.0
50%        2.0        2.0       2.0     34.0      0.0      0.0      0.0      0.0      0.0      0.0  ...    19052.0    18104.0    17071.0    2100.0     2009.0    1800.0    1500.0    1500.0    1500.0                         0.0
75%        2.0        2.0       2.0     41.0      0.0      0.0      0.0      0.0      0.0      0.0  ...    54506.0    50190.0    49198.0    5006.0     5000.0    4505.0    4013.0    4032.0    4000.0                         0.0
max        2.0        6.0       3.0     79.0      8.0      8.0      8.0      8.0      8.0      8.0  ...   891586.0   927171.0   961664.0  873552.0  1684259.0  896040.0  621000.0  426529.0  528666.0                         1.0

[8 rows x 23 columns]
════════════════════════════════════════════════════════════════════════
Valori unici per ogni colonna:

    sex  education  marriage  age  pay_0  pay_2  pay_3  pay_4  pay_5  pay_6  ...  bill_amt4  bill_amt5  bill_amt6  pay_amt1  pay_amt2  pay_amt3  pay_amt4  pay_amt5  pay_amt6  default.payment.next.month
0    2          7         4   56     11     11     11     11     10     10  ...      21548      21010      20604      7943      7899      7518      6937      6897      6939                           2

[1 rows x 23 columns]
════════════════════════════════════════════════════════════════════════
```

[![png](https://automataia.github.io/AutoGnosis/4-progetti/4.1-supervised-learning/4.1.3-predizione-correntisti-morosi/notebook-di-lavoro/output_17_1.png)](https://automataia.github.io/AutoGnosis/4-progetti/4.1-supervised-learning/4.1.3-predizione-correntisti-morosi/notebook-di-lavoro/output_17_1.png)

[![png](https://automataia.github.io/AutoGnosis/4-progetti/4.1-supervised-learning/4.1.3-predizione-correntisti-morosi/notebook-di-lavoro/output_17_2.png)](https://automataia.github.io/AutoGnosis/4-progetti/4.1-supervised-learning/4.1.3-predizione-correntisti-morosi/notebook-di-lavoro/output_17_2.png)

## ► 4.7 Interpretazione e comunicazione dei risultati

-   I risultati mostrano che la clientela è molto giovane, con media di 35 anni.
-   I dati personali(sex,marriage,age) sono scorrelati dall’inadempienza dei pagamenti data dall’ultima colonna.
-   I pagamenti(pay\_\*) sono simili fra loro, molto probabilmente sono dovuti a pagamenti periodici.
-   L’importo della dichiarazione di fatturazione(bill\_amt\*, pay\_amt\*) e simile nei mesi ed è correlato ai pagamenti. Vuol dire che i clienti fanno una vita abitudinaria

## 5\. Modellizzazione

## ► 5.1 Selezione del modello

## ► 5.2 Preparazione dei dati

## ► 5.3 Allenamento del modello

-   USO DEL MODELLO **ANN** ══════════════════════════════════════════════════════════

```
import pandas as pd
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

# Carica il dataset
data = pd.read_csv("04-dataset_pulito.csv")

# Rimuovi la colonna 'ID' dal dataset
data = data.drop('ID', axis=1)
y_train = data['default.payment.next.month']

# identificazione delle colonne categoriali e delle colonne intere
cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
int_cols = data.select_dtypes(include=['int', 'float']).columns.tolist()

# OrdinalEncoder per le variabili categoriali
ordinal_encoder = OrdinalEncoder()
data[cat_cols] = ordinal_encoder.fit_transform(data[cat_cols])

# normalizzazione delle variabili numeriche
scaler = MinMaxScaler(feature_range=(-1, 1))
data[int_cols] = scaler.fit_transform(data[int_cols])

# Crea una matrice X contenente le features
x_train = data.drop('default.payment.next.month', axis=1)

# Crea un array y contenente la variabile di output
#y_train = data['default.payment.next.month']
```

```
from keras import layers
from keras import models
from keras import optimizers
from keras import losses
from keras import regularizers
from keras import metrics# add validation dataset
validation_split = 15 #percentuale
validation_split = int(x_train.shape[0]*validation_split/100)

x_validation=x_train[:validation_split]
x_partial_train=x_train[validation_split:]
y_validation=y_train[:validation_split]
y_partial_train=y_train[validation_split:]

model=models.Sequential()
model.add(layers.Dense(3,kernel_regularizer=regularizers.l2(0.003),activation='sigmoid'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer=optimizers.Adam(learning_rate=1e-3, amsgrad=True),loss='binary_crossentropy',metrics=['acc'])
model.fit(x_partial_train,y_partial_train,epochs=50, batch_size=512,validation_data=(x_validation,y_validation), 
          workers=4, use_multiprocessing=True, verbose=0)
print("score on test: " + str(model.evaluate(x_validation,y_validation)[1]))
print("score on train: "+ str(model.evaluate(x_train,y_train)[1]))
# evaluate the model

model.evaluate(x_validation, y_validation, verbose=0)
```

```
141/141 [==============================] - 0s 2ms/step - loss: 0.5213 - acc: 0.7800
score on test: 0.7799999713897705
938/938 [==============================] - 2s 2ms/step - loss: 0.5191 - acc: 0.7788
score on train: 0.7788000106811523





[0.5213088393211365, 0.7799999713897705]
```