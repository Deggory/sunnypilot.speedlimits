# Revisione UX del portale Speed limits

Data: 26 ottobre 2026. Autore: Codex. Destinatari: Cristian e Claude.

**Stato: proposta da valutare, non autorizzazione a implementare.** Cristian ha chiesto una valutazione precisa e il salvataggio di questo documento. In questa attività non sono stati modificati il sito, le impostazioni o i dati; non sono stati avviati, fermati o aggiornati processi. Il training visto nelle schermate era già in corso.

## 1. Decisione proposta

Organizzare il sito intorno a quattro sezioni: **Overview, Data, Training, Models**. Ogni operazione ha una sede principale; le altre pagine possono mostrarne un riepilogo e un collegamento.

L'aspetto attuale contiene già elementi utili: tipografia semplice, colori coerenti, pulsanti riconoscibili, temi chiaro e scuro, aggiornamento dei lavori in tempo reale. La priorità è sistemare la gerarchia: cosa vedo subito, cosa posso fare qui, dove trovo il resto.

Per «Apple style» propongo: poche decisioni visibili alla volta, testo breve, spaziatura regolare, stato comprensibile e dettagli accessibili quando servono. Il lavoro disponibile deve essere visibile all'apertura della pagina.

Sono possibili due interventi: conservare le sei schede e riordinare soltanto i contenuti, oppure raggrupparle nelle quattro sezioni proposte. Raccomando il secondo: rende esplicito che la revisione appartiene ai dati e che i risultati appartengono ai modelli. Il costo è introdurre un livello di navigazione secondario, che va mantenuto visibile e coerente.

### Struttura proposta

```text
Overview

Data
  Drives       scaricare e preparare i viaggi
  Review       revisionare i fotogrammi
  Dataset      importare le revisioni e controllare i dati per il training

Training
  Cutter       trova i cartelli
  Classifier   legge e classifica i cartelli
  Decider      stabilisce la strada a cui si applicano
  Advanced     training storico detector + reader

Models
  Versions     vedere e scegliere le versioni usate dal progetto
  Results      valutare le versioni e vedere gli errori
  Publish      preparare e pubblicare un pacchetto

Accessi globali: Activity e Settings
```

Le etichette dell'interfaccia restano in inglese, come nel sito attuale. Le spiegazioni di questo documento sono in italiano.

## 2. Cosa ho verificato

Ho osservato il sito servito sulla porta locale del progetto: Overview, Data, Train, Evaluate, Models, elenco Review e Settings; inoltre le viste Classifier, Decider e Pipeline, i menu comma e Server. Ho controllato le pagine principali a 1440 px e a 390 px. Non ho modificato annotazioni né eseguito una revisione di cartelli.

Ho confrontato le schermate con i sorgenti Vue/TypeScript, i template e la documentazione. Le richieste di modifica dal browser di osservazione erano bloccate.

**Il clone è condiviso e il sito è cambiato durante l'osservazione.** In particolare Models aveva inizialmente tre lunghe colonne di versioni; alla rilettura il sorgente era già stato riorganizzato con riepiloghi delle versioni in uso e una tabella per rete. Conservare questo miglioramento e completarlo secondo i criteri seguenti. Non ricreare la pagina precedente per poi sostituirla.

| Problema osservato | Conseguenza | Correzione proposta |
| --- | --- | --- |
| Importazione in Train, dettaglio del dataset in fondo a Data | Per capire e aggiornare lo stesso oggetto si cambia pagina | Riunire tutto in Data → Dataset |
| Versioni precedenti e scelta del modello presenti sia in Train sia in Models | Due luoghi gestiscono la stessa decisione | Rendere Models → Versions la sede della scelta |
| Download nel menu comma; Data rimanda a quel menu | Il primo passo del lavoro è nascosto in un indicatore di stato | Mettere il download in Data → Drives |
| Primo accesso a Review: contatori visibili, gruppi di viaggi tutti chiusi e nascosti | La pagina sembra priva di viaggi apribili | Mostrare subito il gruppo di lavoro pertinente |
| Overview ripete Resume review e mostra lo stesso lavoro in Running, Latest jobs e On the GPU | L'attenzione si distribuisce su contenuti ripetuti | Un'azione consigliata e un riepilogo compatto dei lavori |
| Su mobile il lavoro in corso arriva dopo tre grandi schede di riepilogo | Bisogna scorrere per controllare il training | Portare il lavoro in corso subito sotto il titolo |
| Train mostra numerosi parametri, log e versioni prima dei grafici | Lo stato utile è circondato da dettagli tecnici | Rendere compatto il progresso e spostare i dettagli nel pannello del lavoro |
| Evaluate descrive «80 speed signs» mentre alcune righe riportano 208 | Non è chiaro quale test si sta guardando | Ricavare descrizioni e denominatori dal report selezionato |
| Overview mostra 56% e Review 57% per 10.871 / 19.140 fotogrammi | Due schermate danno risposte diverse alla stessa domanda | Un solo criterio di arrotondamento |
| Settings e Review descrivono revisioni AI «to confirm» | Contraddizione con le regole correnti di AGENTS.md | Verificare e riallineare il contratto funzionale, come specificato nella sezione 9 |

## 3. Mappa esatta degli spostamenti

Questa tabella stabilisce la destinazione principale di ogni funzione. «Collegamento» significa navigazione alla destinazione, non un secondo controllo che esegue la stessa operazione.

| Elemento attuale | Destinazione proposta | Comportamento nella posizione precedente |
| --- | --- | --- |
| Download new drives nel menu comma | Data → Drives, prima azione della pagina | Il menu di stato rimanda a Drives |
| Indirizzo comma, SSH e configurazione iniziale | Settings → Connections → comma | In Drives rimangono stato e collegamento alla configurazione |
| Find the signs / Found by | Data → Drives, configurazione della preparazione | Conservare qui la scelta; mostrare i confronti completi nei risultati dei proponenti |
| Prepare new drives | Data → Drives, accanto al download | Un solo pulsante principale di preparazione |
| Propose again globale | Data → Drives → Advanced | Collegamento contestuale da Review, senza cambiare lo scopo dell'azione globale |
| Elenco dei viaggi da revisionare | Data → Review | Overview mantiene soltanto Resume review e il riepilogo |
| Import reviews / Add in Train | Data → Dataset, in cima alla pagina | In Training: numero di revisioni da importare e link Open dataset |
| In the dataset in fondo a Data | Data → Dataset | Drives mostra soltanto un link al dataset |
| Earlier versions e Use this one in Train | Models → Versions | In Training: link View versions, filtrato sulla rete corrente |
| Evaluate: Chain, Cutter, Classifier, Decider | Models → Results | Conservare l'accesso diretto ai risultati esistenti |
| Evaluate: Proposers | Data → Drives → Compare proposal models | Collegamento dalla scelta del proponente |
| Evaluate: AI reviewers | Data → Review → AI performance | Settings rimane la sede della configurazione dei provider |
| Publish in fondo a Models | Models → Publish, vista dedicata | Versions mostra lo stato del pacchetto e un link |
| Elenco Latest jobs | Activity → Recent | Overview mostra al massimo tre lavori terminati |
| GPU, CPU, RAM, disco, processi | Activity → Server details | Barra globale: indicatore compatto di attività o problema |
| Update the site nel menu GPU | Settings → Server | Nessuna azione di aggiornamento dentro gli indicatori di carico |

Conservare gli indirizzi esistenti e i collegamenti diretti: `/`, `/prepare`, `/routes`, `/train`, `/evaluate`, `/models`, `/account` e gli indirizzi delle singole route. `/routes` deve selezionare Data → Review; `/evaluate` deve selezionare Models → Results. Per il nuovo Dataset propongo `/dataset`. Per Publish propongo `/models#publish`, che deve aprire la vista dedicata. Eventuali nuovi indirizzi devono mantenere funzionanti quelli precedenti.

## 4. Overview: capire cosa fare e cosa sta succedendo

Ordine verticale obbligatorio:

1. Titolo **Overview**.
2. **Running**, soltanto se esistono lavori in corso: una riga compatta per lavoro.
3. **Next step**: una frase, un motivo concreto e un solo pulsante principale.
4. Tre riepiloghi compatti: viaggi da preparare, fotogrammi da revisionare, dataset disponibile.
5. **Recent activity**: massimo tre lavori terminati e il link View all activity.

Una riga Running contiene nome, avanzamento complessivo, fase attuale, tempo rimanente se disponibile e Open. Esempio di formato: `Training cutter · 26% · epoch 5 of 20 · about 19 min left`. I valori sono dinamici.

Aprire la riga porta al pannello del lavoro descritto nella sezione 8. I parametri, i log, la lista dei processi GPU e il pulsante Stop appartengono a quel pannello. Un lavoro in corso non compare una seconda volta fra i lavori recenti terminati.

I tre riepiloghi sono collegamenti alle rispettive destinazioni. Non ripetono Resume review: quel pulsante compare nella sola scheda Next step quando la revisione è l'azione suggerita.

La raccomandazione deve rispettare lo stato reale: un training già avviato va descritto come in corso. La presenza di un training può convivere con la proposta di continuare una revisione, se il backend consente entrambe le attività.

## 5. Data: tre viste con compiti distinti

### 5.1 Drives

Ordine della pagina: titolo; stato del comma e azione di download; preparazione dei viaggi già scaricati; elenco viaggi; Advanced chiuso.

Il blocco iniziale mostra **Download drives**, connessione del dispositivo e ultimo collegamento noto, quando disponibile. Il pulsante di download deve essere visibile anche con il dispositivo offline, disabilitato con una spiegazione leggibile. **Connection settings** apre Settings → Connections → comma.

Il blocco di preparazione mostra quanti viaggi attendono, **Proposal model**, proponente selezionato e **Prepare drives**. Prima dell'avvio, la scelta del proponente deve rendere visibile che si tratta dell'impostazione condivisa del server. Non modificare implicitamente questa preferenza durante la sola apertura della pagina.

Sostituire il paragrafo con tutte le misure con una sintesi: nome, stato Available/Unavailable, metrica principale con denominatore e link **Compare proposal models**. Il confronto mantiene qualità, falsi positivi, velocità e campione utilizzato. La configurazione o installazione di Qwen appartiene a Settings; in Drives compare la disponibilità e il link per configurarlo.

Quando non ci sono nuovi viaggi, mostrare `No drives waiting to be prepared` e mantenere il download ben visibile. Il ricalcolo delle proposte non deve diventare automaticamente l'azione principale solo perché la preparazione non è disponibile.

La tabella dei viaggi mostra nome leggibile, data, stato e azione pertinente. I codici tecnici della route stanno nei dettagli. Advanced contiene la rigenerazione globale delle proposte con il numero di viaggi coinvolti, l'ambito effettivo dell'operazione e la conservazione dei fotogrammi confermati. Non inventare una rigenerazione per singola route se il backend non la supporta.

### 5.2 Review

Mostrare in cima una riga di avanzamento complessivo e **Resume review**. Subito sotto, i filtri **In progress**, **To start**, **Done** con i conteggi; poi l'elenco dei viaggi del filtro selezionato.

Al primo accesso selezionare In progress se contiene viaggi, altrimenti To start, altrimenti Done. Deve sempre esserci un filtro selezionato. Cliccare di nuovo il filtro attivo non deve chiudere e far sparire tutto l'elenco. Una preferenza memorizzata va ripristinata soltanto se il gruppo esiste ancora e contiene elementi.

Ogni riga mostra nome del viaggio, data, fotogrammi confermati sul totale e **Resume** o **Open**. **Review with AI** resta un'azione secondaria esplicita. Le revisioni umane e AI possono avere conteggi distinti, con etichette chiare; il colore è un aiuto aggiuntivo.

**AI performance** apre i confronti dei revisori configurati; **AI settings** apre la configurazione del provider. La prova senza salvataggio e la revisione con salvataggio devono mantenere nomi e conseguenze distinti.

Questa proposta non ridisegna l'editor di annotazione del singolo fotogramma: il suo comportamento richiede una valutazione dedicata.

### 5.3 Dataset

Ordine della pagina:

1. Riepilogo di fotogrammi, oggetti annotati e viaggi già importati.
2. Revisioni nuove o modificate da applicare, pulsante **Import reviews** e stato dell'import.
3. Ultimo import: nuovi, aggiornati, rimossi e invariati, secondo i dati effettivamente disponibili.
4. Distribuzione per famiglia e provenienza umana/AI.
5. Suddivisione Train/Validation/Test/Buffer e relativi avvisi.
6. **By drive**, tabella inizialmente chiusa.

Sostituire **Add** con **Import reviews**: descrive cosa accade. Quando non ci sono modifiche, mostrare `Dataset is up to date`. Dopo un errore, mantenere visibili motivo e collegamento al lavoro.

Specificare sempre l'unità dei numeri: fotogrammi, oggetti o viaggi. Gli oggetti includono anche gli esempi Not a sign: non chiamare indistintamente tutti gli oggetti «signs». La barra dello split deve dichiarare l'unità conteggiata. Gli avvisi sulla qualità della separazione dei dati devono restare visibili anche con i dettagli chiusi.

L'import rimane un'operazione esplicita. Aprire Dataset, cambiare scheda o scegliere una rete non deve avviare importazioni o training.

## 6. Training: configurare, avviare, seguire

Ordine della pagina: titolo; riepilogo compatto del dataset; selettore Cutter/Classifier/Decider; spiegazione della rete; configurazione oppure lavoro in corso; grafici; collegamento al risultato.

Per ogni rete mantenere il nome tecnico e affiancare una spiegazione stabile: **Cutter — Finds signs**, **Classifier — Reads signs**, **Decider — Chooses the road**. Usare gli stessi nomi in risultati, versioni, lavori e messaggi.

Il riepilogo dataset mostra quantità già importate e, se necessario, `N reviews not imported` con **Open dataset**. Non ospita un secondo pulsante di importazione. Le condizioni che impediscono l'avvio devono avere una spiegazione visibile, con il collegamento alla funzione che le risolve.

Nel Cutter restano visibili dimensione dell'input e famiglie di cartelli: sono scelte che cambiano il training. Gli altri parametri già supportati possono comparire sotto **Advanced settings**. I parametri non configurabili diventano un riepilogo in sola lettura. Non aggiungere controlli che il backend non sa applicare.

Il pulsante principale nomina la rete: **Train cutter**, **Train classifier**, **Train decider**. Una volta avviato il lavoro, la sua posizione viene occupata dal riepilogo del progresso; la pagina non cambia rete in risposta a ogni aggiornamento. Se l'utente ha scelto una rete, conservarla durante quella visita.

I grafici del training seguono subito il progresso. **Latest version**, **Earlier versions** e **Use this one** lasciano questa pagina. Al termine mostrare **View results** e **View versions**, entrambi riferiti al training appena terminato.

La voce attuale **Pipeline** allena detector e reader; non va presentata come equivalente all'insieme Cutter + Classifier + Decider. Collocarla in **Advanced → Legacy detector + reader**, con una frase che ne spieghi l'ambito reale e conservando le opzioni oggi supportate.

## 7. Models: scegliere, capire, pubblicare

### 7.1 Versions — vista iniziale di Models

Conservare il miglioramento già presente nei sorgenti: tre riepiloghi delle reti selezionate e una tabella delle versioni di una rete alla volta.

Ogni riepilogo mostra rete, versione scelta e una metrica comprensibile. La tabella mostra versione/data, stato, metrica principale e denominatore del test. Su mobile ogni riga diventa una scheda compatta; i dettagli secondari si aprono a richiesta.

Aprendo una versione si vedono le altre misure, la configurazione e **View results**. Qui rimane l'unico controllo **Use this version**. Il riepilogo della conferma deve nominare la versione e gli utilizzi che il progetto ne fa realmente.

Preferire il badge **Selected for this project** al generico **In use**. Qualora esista un fallback automatico, mostrarlo separatamente: versione selezionata e versione effettivamente usata devono essere distinguibili. La scelta sul server, la pubblicazione e l'installazione su comma sono stati diversi.

### 7.2 Results

In alto: rete o combinazione misurata, versione, campione di valutazione e riferimento di confronto. Sotto: massimo tre metriche principali, elenco compatto delle versioni confrontabili, **Inspect errors** e **All metrics**.

Il player dei fotogrammi si apre con Inspect errors, mantenendo versione e filtri. Mantiene le funzioni attuali: route, soglia, fotogrammi con errori, riproduzione e legenda. I risultati numerici restano disponibili anche per i modelli che non hanno un player.

Regola obbligatoria: confrontare soltanto misure con campione/split, metrica, soglia e componenti della catena compatibili. Lo stesso numero di esempi o la stessa risoluzione del Cutter non provano che due test siano equivalenti. Usare gli identificativi o le impronte già disponibili; quando mancano, scrivere **Comparison unavailable**.

Versioni valutate su test differenti rimangono consultabili, in un gruppo **Different test set**, senza classifica comune, colore da vincitore o differenza percentuale fra loro. Test e validation devono essere identificati accanto alla misura. Nessun denominatore, come «80 signs», va scritto fisso nel testo della pagina.

I confronti dei proponenti e dei revisori AI, spostati in Data, seguono le stesse regole. Il benchmark deve continuare a usare il campione previsto dalla sua funzione.

### 7.3 Publish

Vista separata e raggiungibile dalla navigazione di Models. Sequenza: versioni incluse; **Prepare package**; esito dei controlli; **Publish to Hugging Face**; collegamento al pacchetto pubblicato.

Mostrare la connessione Hugging Face e un link a Settings per modificarla. Se le versioni selezionate differiscono da quelle del pacchetto, indicare quale pacchetto si sta pubblicando e invitare a prepararne uno aggiornato; non far credere che il pacchetto contenga automaticamente le nuove scelte.

La preparazione è locale e la pubblicazione è un upload esplicito. Le etichette e i messaggi devono descrivere queste operazioni. L'installazione sul comma può essere annunciata soltanto quando esistono e sono verificati il runtime e l'integrazione installabile previsti dalle regole del progetto.

## 8. Activity e Settings

**Activity** è accessibile da tutte le pagine. Sul desktop apre un pannello laterale; sul telefono una vista a tutta altezza. Contiene lavori in corso, lavori recenti e Server details.

Un unico pannello del lavoro, riutilizzato da Overview e dalle pagine operative, mostra: nome; stato; progresso complessivo; fase attuale; tempo trascorso e stima rimanente; grafici disponibili; parametri chiusi sotto Configuration; log sotto Log; Stop e sue conseguenze. I riepiloghi in altre pagine leggono gli stessi dati.

Chiamare esplicitamente **Overall progress** e **Current step** le due percentuali quando sono entrambe presenti. Senza una misura attendibile mostrare una fase in corso senza inventare una percentuale. La riga tecnica del log non deve sostituire la descrizione comprensibile della fase.

**Settings** contiene quattro gruppi: Account & appearance; Connections, con comma e Hugging Face; AI providers; Server, con manutenzione e Update the site. Conservare i permessi e le conferme oggi previsti. L'accesso in sola lettura deve mostrare chiaramente i limiti delle azioni.

## 9. Incoerenze da risolvere prima delle rifiniture

**Revisioni AI.** Le regole correnti di [AGENTS.md](../AGENTS.md) stabiliscono che una conferma AI salvata è definitiva e può essere importata con provenienza AI, senza una seconda approvazione umana. Ho invece osservato messaggi «waits for you», «AI to confirm» e conteggi collegati nel sito e nei sorgenti.

Claude deve verificare insieme interfaccia, API, conteggi e importazione rispetto a questa regola. Non basta sostituire il testo se il comportamento continua a imporre un'approvazione diversa. Non basta nemmeno rietichettare dei suggerimenti come conferme. Distinguere suggerimento/prova senza salvataggio da conferma effettivamente salvata; mantenere la provenienza AI. La risoluzione funzionale e gli eventuali interventi sui dati richiedono un incarico separato: questo documento non li autorizza.

**Percentuali.** Usare un unico arrotondamento per l'avanzamento delle revisioni: intero più vicino. Con 10.871 / 19.140 il valore visualizzato è 57% in tutte le viste. Le percentuali dei singoli passi di un lavoro possono differire da quella complessiva, con etichette distinte.

**Importazione.** Correggere le spiegazioni incompatibili con il comportamento attuale: alcune descrizioni parlano di aggiornamento delle revisioni, altre di dataset che aggiunge soltanto nuovi fotogrammi. La UI deve riportare quello che l'import supporta davvero, senza promettere aggiornamenti non verificati.

## 10. Specifiche visive e comportamento responsive

Questi valori sono la proposta iniziale da applicare in modo uniforme, non una descrizione del CSS attuale.

| Elemento | Specifica |
| --- | --- |
| Contenitore desktop | Massimo 1180 px; margine interno 24 px; 16 px su telefono |
| Spaziatura | Scala 4, 8, 12, 16, 24, 32, 48 px |
| Titolo pagina | Visibile anche su desktop; 28–32 px |
| Testo normale | 16 px; interlinea circa 1,5 |
| Testo secondario | Almeno 14 px; leggibile in entrambi i temi |
| Superfici | Sfondo, pannello, controllo: tre livelli coerenti; bordo sottile dove necessario |
| Angoli | Pannelli 16 px; controlli 10–12 px; pillole per stati brevi |
| Azione principale | Una per vista operativa nello stato corrente; blu |
| Stato positivo | Verde per completamento o disponibilità verificati |
| Attenzione/errore | Ambra/rosso con testo esplicativo; il carico GPU normale resta neutro |
| Aree cliccabili | Almeno 44 × 44 px per i controlli usati su telefono |
| Animazioni | 150–200 ms per transizioni; rispettare reduced motion |

Conservare Dark, Light e Automatic e rendere Automatic la proposta di default per i nuovi browser; mantenere le preferenze già scelte. Verificare contrasto del testo normale di almeno 4,5:1 e dei controlli di almeno 3:1.

Sopra 767 px usare la navigazione principale in alto, con Activity e Settings a destra. Sotto 768 px mostrare una barra inferiore persistente con Overview, Data, Training e Models; tenere logo, Activity e Settings in alto. Riservare lo spazio della barra inferiore e della safe area, così non coprono i contenuti.

La navigazione secondaria resta sotto il titolo: Drives/Review/Dataset oppure Versions/Results/Publish. I titoli e le spiegazioni delle reti restano leggibili su telefono; i selettori possono andare su più righe. Evitare scorrimento orizzontale dell'intera pagina. Focus da tastiera visibile, etichette accessibili e stato selezionato esplicito sono parte della verifica.

Gli aggiornamenti live cambiano valori e stato senza spostare focus, selezione, ordine delle righe o posizione di lettura. Caricamento, errore di lettura e assenza di dati devono avere messaggi distinti. Per un errore di lettura mostrare Retry; per assenza di dati indicare il primo passo utile.

## 11. Ordine di realizzazione proposto

| Priorità | Lavoro | Esito verificabile |
| --- | --- | --- |
| P0 | Verificare incoerenze AI, denominatori, arrotondamenti e testi dell'import | Ogni stato e numero ha lo stesso significato nelle varie pagine |
| P1 | Navigazione a quattro sezioni, sottoviste e collegamenti precedenti | Tutte le funzioni hanno una destinazione definita e raggiungibile |
| P1 | Dataset dedicato e import spostato; download visibile in Drives | Preparazione, revisione e importazione hanno sedi riconoscibili |
| P1 | Review con elenco visibile al primo accesso | Si può aprire subito un viaggio disponibile |
| P2 | Training concentrato su avvio/progresso/grafici; Versions come sede della scelta | Scompaiono i controlli duplicati fra Training e Models |
| P2 | Results con confronti compatibili; Publish separato | Valutazione, scelta e pubblicazione sono distinguibili |
| P3 | Activity comune, Settings riordinato, rifiniture visive e mobile | Stessa gerarchia e stessi controlli su tutte le pagine |

Realizzare e verificare una vista per volta, mantenendo il funzionamento degli indirizzi esistenti. Eventuali comportamenti nuovi che richiedono supporto backend vanno identificati come tali prima di presentarli nell'interfaccia.

## 12. Criteri di accettazione per Claude

- Aprendo Overview a 390 × 844, con un lavoro in corso, il suo riepilogo compare prima delle schede generali.
- Da qualunque pagina si raggiunge Activity con un solo comando.
- In Data → Drives il download è visibile senza aprire il menu comma.
- Con viaggi disponibili, il primo accesso a Data → Review mostra almeno un viaggio apribile senza dover scoprire che i contatori sono pulsanti.
- L'importazione è eseguibile in Data → Dataset; Training rimanda a quella vista.
- La scelta Use this version esiste in Models → Versions; Training rimanda alle versioni della rete selezionata.
- I risultati di test differenti sono identificati e non ricevono una classifica comune.
- Titolo, descrizione e frazioni di una misura derivano dal medesimo report.
- Suggerimenti AI senza salvataggio e conferme AI salvate hanno nomi e conteggi coerenti con le regole del progetto, dopo la verifica funzionale prevista al punto P0.
- Selezione di una versione, preparazione del pacchetto e pubblicazione rimangono operazioni esplicite e distinte.
- Un controllo disabilitato ha il motivo leggibile senza richiedere hover.
- Su telefono nessun controllo essenziale è coperto dalle barre persistenti e la pagina non scorre orizzontalmente.
- Tastiera, temi chiaro/scuro, reduced motion, caricamento, errore, dati assenti, lavoro in corso, fallito e completato sono verificati sulle viste interessate.
- Tutti gli indirizzi precedenti e i link ai singoli viaggi continuano a funzionare.

## 13. Punti del codice da cui partire

Questi sono riferimenti per l'eventuale implementazione; in questa revisione non sono stati modificati.

| Area | Sorgenti principali |
| --- | --- |
| Navigazione e indice dei viaggi | [pages.py](../src/nnslr_tools/pages.py), [site.ts](../web/src/pages/site.ts) |
| Contenitore delle viste | [ControlApp.vue](../web/src/control/ControlApp.vue), [tabs.ts](../web/src/control/tabs.ts) |
| Overview | [OverviewTab.vue](../web/src/control/OverviewTab.vue), [FlowTiles.vue](../web/src/control/FlowTiles.vue), [NextStepCard.vue](../web/src/control/NextStepCard.vue) |
| Drives e Dataset | [PrepareTab.vue](../web/src/control/PrepareTab.vue), [DatasetSection.vue](../web/src/control/DatasetSection.vue) |
| Training | [TrainTab.vue](../web/src/control/TrainTab.vue), [TrainStart.vue](../web/src/control/TrainStart.vue), [TrainingCharts.vue](../web/src/control/TrainingCharts.vue) |
| Risultati e versioni | [EvaluateTab.vue](../web/src/control/EvaluateTab.vue), [ModelsTab.vue](../web/src/control/ModelsTab.vue), [ModelItem.vue](../web/src/control/ModelItem.vue), [models.ts](../web/src/control/models.ts) |
| Pubblicazione | [PublishSection.vue](../web/src/control/PublishSection.vue) |
| Lavori e progresso | [JobProgress.vue](../web/src/control/JobProgress.vue), [LogSheet.vue](../web/src/control/LogSheet.vue), [TopMachine.vue](../web/src/components/TopMachine.vue) |
| Configurazione | [TopComma.vue](../web/src/components/TopComma.vue), [AccountApp.vue](../web/src/account/AccountApp.vue), [AiProviders.vue](../web/src/account/AiProviders.vue), [AiRouteSheet.vue](../web/src/components/AiRouteSheet.vue) |
| Aspetto | [shared.css](../src/nnslr_tools/web/shared.css), [site.css](../src/nnslr_tools/web/site.css), [control.html](../src/nnslr_tools/web/control.html) |

Le modifiche al frontend, se richieste successivamente, seguono il flusso di build e verifica locale documentato in [GUIDA.md](GUIDA.md). Questo documento non autorizza build, installazioni, riavvii, importazioni, training, pubblicazioni, commit o push.
