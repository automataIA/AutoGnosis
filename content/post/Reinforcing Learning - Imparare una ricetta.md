---
title: Reinforced Learning - Imparare una ricetta
date: 2020-01-27
tags: ["markdown"]
image : "post/img/RL.jpg"
Description  : "Far imparare una ricetta alla AI tramite Reinforced Learning..."
---

**Avvio presentazione fullscreen**

<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vSZIuakv_Zm1uhLa7_x5IGQOUkaauWVTF79JlGjQTJ4fgwI94tsrULwtJUwvBIlLmZ4xHMtx84LlBgE/embed?start=true&loop=false&delayms=5000" frameborder="0" width="820" height="498" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>

### Introduzione al problema

Lo script Python che ho scritto utilizza l’algoritmo di apprendimento rinforzato “Q-learning” per risolvere un problema di esplorazione di uno spazio di stati.

L’agente deve esplorare l’ambiente di gioco, prendere decisioni in ogni stato e massimizzare la sua ricompensa complessiva, seguendo una politica di scelta delle azioni che si basa sul valore Q degli stati e delle azioni possibili.

L’output dello script mostra la sequenza di azioni scelte dall’agente durante la sua esplorazione dell’ambiente di gioco, fino a raggiungere lo stato finale desiderato.

Lo script può essere utilizzato come base per risolvere problemi di esplorazione di spazi di stati più complessi, aggiungendo regole e restrizioni all’ambiente di gioco, o modificando l’algoritmo di apprendimento utilizzato.

### Descrizione del problema tramite un esempio

Immagina di dover imparare a cucinare un piatto specifico per la prima volta. Per farlo, devi seguire una ricetta che ti indica gli ingredienti e i passaggi necessari. Nel nostro caso, l’ambiente di gioco è come la tua cucina, la matrice di ricompensa è la soddisfazione che provi quando il piatto viene bene, mentre la matrice di transizione è il passaggio da un ingrediente o da una fase della ricetta all’altra.

Nel processo di apprendimento rinforzato, come nella cucina, impariamo dalle esperienze precedenti per migliorare il nostro processo decisionale in futuro. In questo esempio, l’algoritmo Q-learning impara la politica migliore per raggiungere l’obiettivo finale, che corrisponde alla cottura del piatto perfetto. Invece di seguire la ricetta esattamente come scritta, possiamo adattare la nostra strategia in base alle situazioni impreviste che possono presentarsi durante la preparazione.

Il ciclo di apprendimento corrisponde al processo di preparazione del piatto, dove impariamo a conoscere gli ingredienti, le quantità e le tecniche necessarie. Durante questo processo, aggiorniamo la nostra “matrice Q” personale, in cui registriamo le conoscenze acquisite e le utilizziamo per migliorare la qualità del piatto.

Infine, quando abbiamo imparato abbastanza e siamo pronti a testare la nostra politica appresa, possiamo preparare il piatto con sicurezza e facilità, sapendo esattamente cosa fare in ogni fase della preparazione. In sostanza, l’apprendimento rinforzato è come imparare a cucinare un nuovo piatto, utilizzando le esperienze passate per migliorare il nostro processo decisionale e ottenere un risultato finale soddisfacente.

```
import numpy as np
import random

# definire l'ambiente
num_states = 6
num_actions = 2
reward_matrix = np.array([[0, 0], [0, 0], [0, 0], [1, 1], [0, 0], [0, 0]])
transition_matrix = np.array([    [0.5, 0.5, 0, 0, 0, 0],
    [0.5, 0, 0.5, 0, 0, 0],
    [0, 0.5, 0, 0.5, 0, 0],
    [0, 0, 0.5, 0, 0.5, 0],
    [0, 0, 0, 0.5, 0, 0.5],
    [0, 0, 0, 0, 0.5, 0.5]])

# definire iperparametri
discount_factor = 0.9
learning_rate = 0.1
num_episodes = 1000

# inizializzare la matrice Q
q_matrix = np.zeros((num_states, num_actions))
```

### Inizializzazione del problema

Si definisco le grandezze di input scelte e le variabili da inizializzare utilizzate successivamente nella fase di addestramento. I parametri sono definiti come segue:

1.  `discount_factor`: anche noto come gamma, indica il fattore di sconto utilizzato per pesare la ricompensa a breve termine rispetto alla ricompensa a lungo termine. Un valore elevato di discount\_factor indica che l’agente darà maggiore importanza alle ricompense future rispetto alle ricompense immediate.
    
2.  `learning_rate`: anche noto come alpha, indica il tasso di apprendimento utilizzato dall’algoritmo Q-learning per aggiornare il valore Q di uno stato e di un’azione. Un valore elevato di learning\_rate può rendere l’apprendimento più rapido ma meno stabile, mentre un valore basso può rendere l’apprendimento più lento ma più stabile.
    
3.  `num_episodes`: indica il numero di episodi che l’agente dovrà completare durante il training. Un episodio è una sequenza di azioni che inizia dallo stato iniziale e termina quando l’agente raggiunge uno stato terminale. Un maggior numero di episodi può migliorare la stabilità e la qualità dell’apprendimento, ma richiederà più tempo di esecuzione.
    

```
# ciclo di apprendimento
for episode in range(num_episodes):
    state = random.randint(0, num_states-1)
    while state != 3:
        # scegliere un'azione con probabilità epsilon-greedy
        epsilon = 0.1
        if random.random() < epsilon:
            action = random.randint(0, num_actions-1)
        else:
            action = np.argmax(q_matrix[state])
        # eseguire l'azione e ottenere una ricompensa
        next_state = np.random.choice(num_states, p=transition_matrix[state])
        reward = reward_matrix[state][action]
        # aggiornare la matrice Q
        q_matrix[state][action] = q_matrix[state][action] + learning_rate * \
            (reward + discount_factor * np.max(q_matrix[next_state]) - q_matrix[state][action])
        state = next_state

# testare la politica appresa
state = 0
while state != 3:
    action = np.argmax(q_matrix[state])
    print("Stato:", state, "Azione:", action)
    state = np.random.choice(num_states, p=transition_matrix[state])
print("Stato finale:", state)
```

```
Stato: 0 Azione: 0
Stato: 1 Azione: 0
Stato: 0 Azione: 0
Stato: 0 Azione: 0
Stato: 1 Azione: 0
Stato: 2 Azione: 0
Stato finale: 3
```

### Spiegazione tramite esempio dei risultati ottenuti

1.  Stato 0: è il punto di partenza dell’agente, ovvero il momento in cui si trova davanti alla lista degli ingredienti necessari per la ricetta e deve decidere quale azione intraprendere.
    
2.  Stato 1: rappresenta il momento in cui l’agente ha deciso di prendere una certa quantità di un ingrediente specifico, ma ha scoperto che non è sufficiente per la ricetta.
    
3.  Stato 0: dopo il tentativo fallito di trovare la quantità giusta dell’ingrediente desiderato, l’agente torna allo stato 0 per cercare una soluzione alternativa.
    
4.  Stato 0: ancora una volta, l’agente non riesce a trovare la quantità giusta dell’ingrediente desiderato e torna allo stato 0.
    
5.  Stato 1: dopo aver nuovamente cercato un’altra quantità dell’ingrediente, l’agente scopre che la quantità giusta è ora disponibile e la prende.
    
6.  Stato 2: rappresenta il momento in cui l’agente si è spostato verso gli altri ingredienti necessari per la ricetta e ha iniziato a raccoglierli.
    
7.  Stato 3: rappresenta il momento in cui l’agente ha raccolto tutti gli ingredienti necessari e si sta preparando per la fase successiva della ricetta.
    
8.  Stato finale: rappresenta il momento in cui l’agente ha completato con successo la sua missione, ovvero raccogliere tutti gli ingredienti necessari per la ricetta.