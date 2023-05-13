---
title: sentiment analysis using transformers
date: 2020-01-12
tags: ["markdown"]
image : "post/img/SA.jpg"
Description  : "Analisi del sentimento di un corpo di messaggi di testo tramite transformers..."
---

## Costruire un classificatore di sentiment

#### Introduzione ai Transformers

L’analisi del sentiment è un tipo di problema dell’elaborazione del linguaggio naturale che consiste nel determinare se un testo ha un sentiment positivo o negativo. Questo compito non è così facile come si potrebbe pensare, perché il linguaggio è per definizione molto intricato e complesso. Ciò è ancora più accentuato in questo tipo di insiemi di dati in cui si vuole valutare il sentiment sulla base di una recensione a testo libero, in quanto l’utente può scriverci qualsiasi cosa decida. Storicamente, questo problema veniva risolto facendo una sorta di conteggio istruito delle parole positive rispetto a quelle negative per quantificare il sentiment complessivo del testo.

Come si può immaginare, questo può diventare complesso molto velocemente. Pensiamo a un esempio come la seguente recensione: “Il film era molto bello”. Speriamo che il nostro modello sia in grado di classificarla come positiva, poiché si tratta di una recensione positiva. Ma ora pensiamo a un altro esempio come: “Il film non era molto bello”. Questa volta, la parola “non” nega la “positività” della parola “molto buono”, e quindi non è facile per un modello capire questa complessità se si limita a contare il sentiment positivo o negativo.

Il tipo successivo di modelli che abbiamo provato è stato quello delle Reti Neurali Ricorrenti. Questi modelli cercano di comprendere la relazione **sequenziale** tra le parole di una frase, costruendo fondamentalmente una rete che considera i risultati dei token o delle parole precedenti per prevedere quella successiva. In questo modo si ottengono reti più potenti, poiché il linguaggio è per definizione sequenziale. Questo concetto di sequenzialità è ciò che generalmente chiamiamo **contesto**. Utilizzando queste reti (in particolare le RNN e le LSTM), i modelli linguistici sono diventati molto più sofisticati, in quanto sono stati in grado di fare previsioni basate sulle parole precedenti apparse nella frase. Ma questo aveva un problema: il contesto è di natura **bidirezionale**. Ciò significa che potremmo riferirci a un token che viene **dopo** il token, non solo prima. E questi modelli tendono a fallire nei casi in cui il contesto è più legato alle parole che seguono la parola che stiamo passando al modello.

Così, qualche anno fa, è arrivato un nuovo giocatore che ha rivoluzionato il mondo del Deep Learning, dapprima nel campo della PNL, ma ora si è diversificato e li si può vedere ovunque, e si chiamano **Transformers**. Utilizzando un meccanismo chiamato auto-attenzione e bidirezionalità, i primi trasformatori erano in grado di introdurre un **contesto** alle frasi in entrambe le direzioni, comprendendo contemporaneamente **quali token fossero rilevanti** per il significato della frase. Utilizzando enormi insiemi di dati, questi modelli basati sui trasformatori hanno conquistato il mondo dell’NLP. Oggi utilizzeremo un tipo particolare di modello pre-addestrato per l’analisi del sentimento per cercare di dedurre il sentimento di una determinata recensione cinematografica.

### Introduzione alla libreria di trasformatori HuggingFace

La libreria **transformers** di HugginFace 🤗 ci consente di accedere gratuitamente a questi modelli di transformer pre-addestrati! Questi modelli possono quindi essere messi a punto per eseguire un’attività specifica che vogliamo svolgere con essi. Ancora meglio, HuggingFace offre già una serie di modelli di trasformatore che sono stati messi a punto per eseguire diverse attività. Se vuoi saperne di più sui diversi modelli perfezionati offerti da huggingface, visita [questo link](https://huggingface.co/docs/transformers/main_classes/pipelines). Per questo esercizio utilizzeremo il modello di analisi del sentimento offerto nella classe **pipeline**.

Per prima cosa, dobbiamo installare la libreria transformers nel kernel (o nel tuo computer locale, se stai eseguendo questo processo localmente. Quindi importeremo semplicemente la classe “pipeline” dalla libreria transformers, insieme ad alcune librerie extra che saranno utile per leggere i dati e calcolare l’accuratezza del modello.

```
! pip install transformers
```

```
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from transformers import pipeline
```

### Leggi i dati

Per utilizzare il modello, leggeremo i dati dalla cartella movie.csv nel kernel, che contiene molte recensioni di film in una colonna “text” e un tag che indica se il sentiment del testo per quella voce è positivo (1) o negativo (0), indicato dal nome della colonna “label”. La parte di inferenza del processo che utilizza i modelli pre-addestrati può richiedere molto tempo, quindi limiteremo il processo a un campione che contiene 2000 recensioni positive e 2000 recensioni negative, poiché vogliamo semplicemente testare il modello pre-addestrato e vedere quale accuratezza otteniamo.

Per visualizzare i risultati, salveremo la prima riga come esempio per testare il modello e verificare che l’output sia quello desiderato. Questo è ciò che faremo di seguito nella variabile “test\_drive”.

```
df = pd.read_csv('/kaggle/input/imdb-movie-ratings-sentiment-analysis/movie.csv')

test_drive = df.iloc[0]
test_drive
```

### Testando il modello pipeline da HF

Ora facciamo un’inferenza su un testo con il modello, usando l’oggetto **pipeline** di HuggingFace. L’oggetto pipeline ci permette di creare un’istanza del modello pre-addestrato di cui abbiamo parlato prima e di effettuare rapide chiamate di inferenza per ottenere una predizione. Il modello che caricheremo è un modello PyTorch, quindi dobbiamo usare la sua API per fare una previsione. L’API è semplicemente: model(“cosa si vuole predire”)

Un aspetto da tenere in considerazione è che il modello carica di default il modello BERT uncased, un trasformatore bidirezionale sviluppato da Google. Il modello consente di inserire testi che possono essere lunghi fino a **512 tokens**, quindi dobbiamo fare attenzione. I testi in questo set di dati sono spesso più lunghi, quindi dobbiamo essere pronti a risolvere questo problema.

Se siete interessati, il modello ha come default **Distil-BERT**, che è una versione più leggera del modello completo. Questo modello è addestrato in inglese e non fa distinzione tra caratteri maiuscoli e minuscoli. È un buon punto di partenza ed è anche il modello predefinito. Se siete interessati, posso anche pubblicare un altro quaderno per provare diversi modelli pre-addestrati e verificare come si comportano gli uni rispetto agli altri per questo compito.

Il processo di inferenza del modello BERT richiede un po’ di tempo, quindi lo eseguiremo su un sottoinsieme del nostro DataFrame originale, solo per ridurre i costi di calcolo. Se volete testare il modello sull’intero dataset di 40.000 voci, potete farlo, ma ci vorrà un po’ di tempo. Basta cambiare inference\_df per puntare all’intero dataset (df) eseguendo il seguente comando (si tenga presente che l’accuratezza sarà quasi identica):

```
smaller_df = pd.concat([
    df[df['label'] == 1].sample(2000, random_state=101),
    df[df['label'] == 0].sample(2000, random_state=101)
])

inference_df = smaller_df.copy()
```

### Costruire il modello per Sentiment Analysis

L’unica cosa di cui abbiamo bisogno ora per usare il modello è istanziare la classe pipeline e salvarla in una variabile chiamata “model”. **L’output dell’oggetto pipeline è il modello pre-addestrato** con i suoi pesi, quindi è sufficiente passare un testo attraverso il modello per ottenere una previsione.

Ricordiamo che **dobbiamo ancora ridurre il numero di token (parole)** al numero massimo consentito dal modello BERT pre-addestrato. Lo faremo semplicemente troncando i risultati a 512 token al massimo. In altre parole, se la stringa contiene più di 512 parole, è sufficiente mantenere le prime 512.

Dopodiché, l’unica cosa che resta da fare è calcolare l’accuratezza del modello così com’è. Lo faremo utilizzando la funzione accuracy\_score della libreria sklearn.metrics. Procediamo di seguito:

```
model = pipeline('sentiment-analysis')
```

```
model(test_drive['text'])
```

Pre-elaboriamo il DataFrame in modo che non contenga frasi più grandi di 512 parole (token).

```
inference_df['text'] = inference_df['text'].map(lambda x: x if len(x.split(' ')) <= 280 else ' '.join(x.split(' ')[:280]))
```

```
y_pred = model(inference_df['text'].to_list())
y_pred_values = [1 if dictionary['label'] == 'POSITIVE' else 0 for dictionary in y_pred]
y_pred_values
```

```
y_true = inference_df['label']

accuracy_score(y_true, y_pred_values)
```

```
print(classification_report(y_true, y_pred_values))
print(confusion_matrix(y_true, y_pred_values))
```

### Conclusioni

Alla fine, otteniamo un modello che prevede correttamente l'89% delle recensioni, senza alcun addestramento! I modelli di HuggingFace sono ottimi strumenti di base e possono funzionare molto bene anche se si decide di non metterli a punto. Se avete un compito di sentiment analysis e non avete il tempo o le conoscenze tecniche per ri-addestrare questo modello sui vostri dati, potete provare questo approccio e vedere voi stessi che i risultati possono essere davvero buoni!


###### _**Questo notebook preso da [Kaggle](https://www.kaggle.com/code/rererel/89-accuracy-with-4-lines-of-code-using-hf/edit) è stato tradotto in italiano. Volendo sottolineare le potenzialità dei TRANSFORMER e dei modelli PRE-ADDESTRATI, che tramite poche righe di codice riesce a fare risultati notevoli.**_