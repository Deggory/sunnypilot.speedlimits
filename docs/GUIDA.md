# Guida rapida: dai video al modello

Come si usa `sunnypilot.speedlimits` ogni giorno sul server (la LXC con la V100), con i tre helper e il pannello web. Per installare l'ambiente da zero vedi [INSTALL.md](INSTALL.md); per i comandi `nnslr` uno per uno vedi [upstream-training-guide.md](upstream-training-guide.md).

`nnslr` (Neural Network Speed Limit Recognition) è il nome del progetto e della sua CLI (`.venv/bin/nnslr`): riconosce i cartelli di limite di velocità con una rete neurale. Il nome viene dal progetto da cui è stato recuperato, `openpilot-nnslr-tools`.

Tutti i comandi si lanciano dalla cartella del repository, sul server:

```sh
cd ~/sunnypilot.speedlimits
```

Ogni operazione qui sotto è uno script del repository: niente passaggi a mano da ricordare. Se una cosa si fa spesso e non è ancora scriptata, va aggiunta come helper e documentata qui.

## 1. Il giro completo

| Passo | Comando | Cosa fai tu |
| --- | --- | --- |
| 0 | `helpers/fetch_comma_segments.sh` (sul sito: la pillola **comma** in alto a destra, pallino verde se il comma è raggiungibile e accetta la chiave del server, **Download new drives**; lì si cambia anche il suo indirizzo, salvato in `data/web/comma.json` e passato come `--host`. Per un comma nuovo il pannello guida i passi: SSH acceso e utente GitHub sul comma, chiave pubblica del server aggiunta a quell'utente GitHub con **Copy**, indirizzo, **Test now**; vedi docs/INSTALL.md, "The comma, from zero". Un comma a cui il server è già entrato una volta (l'orario dell'ultimo accesso è in `data/web/comma-seen.json`) e che adesso è solo spento mostra "Last connected …" invece dei passi) | Scarica tutti i giri ancora sul comma, anche quelli vecchi non in dashcam mode, nella cartella condivisa con `openpilot.distill` (`/root/comma-videos`); toglie i primi due e gli ultimi due segmenti e quelli con la retromarcia. Distill non li usa: allena solo su quelli che scarica lui (dashcam). |
| 1 | `helpers/1_prepare.sh` | Niente: collega i video, estrae i frame (1 al secondo) e propone i cartelli per ogni route nuova. |
| 2 | `helpers/2_review.sh` | Apri il link che stampa e confermi o correggi i cartelli. È l'unico passo a mano. |
| 3 | `helpers/3_train.sh` | Niente: importa le revisioni, controlla il dataset, fissa lo split, crea `.venv-gpu` la prima volta e allena. |
| 4 | `scripts/publish_model.sh data/runs/<run>/bundle --no-upload` | Controlla il bundle; poi senza `--no-upload` lo pubblica su Hugging Face. |

In qualsiasi momento `helpers/status.sh` mostra ogni route (segmenti, frame, frame revisionati, oggetti nel dataset), cosa sta girando, GPU e disco, e il **prossimo comando** da lanciare. Non cambia niente.

Gli helper si possono rilanciare quando si vuole: il lavoro già fatto viene saltato. `--help` su ognuno spiega le opzioni.

### Opzioni utili

- `helpers/1_prepare.sh --jobs=4`: prepara 4 route alla volta (più core, più veloce).
- `helpers/1_prepare.sh --background` e `helpers/3_train.sh --background`: continuano anche se chiudi la ssh; il log è in `data/logs/`.
- `helpers/3_train.sh --dataset-only`: solo import e controllo del dataset, senza GPU.
- `helpers/3_train.sh --continue`: nuovo training che parte dai pesi "stable" pubblicati su Hugging Face.
- `helpers/3_train.sh --resume=<run>`: riprende un training interrotto (stessa cartella `data/runs/<run>/`).

### Tre regole da sapere

- **Servono almeno 3 route revisionate** per allenare. `nnslr build-splits` divide il dataset in tre parti indipendenti, per route intere: validation e test prendono circa il 15% ciascuna, tutto il resto è train, quindi ogni route revisionata viene usata. Il test prende per prime le route con più fotogrammi confermati da una persona (`--test-route <route>` ne fissa una nel test). Con meno di 3 route si ferma. Le revisioni importate restano nel dataset, non si perde niente.
- **Il dataset segue l'ultima revisione di ogni fotogramma.** Un fotogramma già importato e poi rivisto (riquadri cambiati, o revisione riclassificata come AI) sostituisce le sue righe vecchie; uno già importato e poi ritirato viene tolto. Le righe tolte restano archiviate in `data/annotations/replaced/`. `nnslr import-annotations --from-db --dry-run` mostra cosa entrerebbe, cosa verrebbe sostituito e cosa tolto, senza scrivere niente.
- **Una route già pre-annotata non viene più rielaborata.** Una nuova pre-annotazione lascerebbe le revisioni sul report vecchio e l'import le salterebbe. I segmenti di quella route arrivati dopo restano fuori; `helpers/status.sh` li segnala.

## 2. Il pannello web

Il sito è in inglese e ha l'aspetto di `distill-web` e `perception-web` (scuro; l'icona **Account** in alto a destra apre il menu con l'aspetto Dark / Light / Automatic, la password e l'uscita). Quattro sezioni nella barra in alto (sul telefono in una barra in basso, come un'app): **Overview**; **Data**, con le pagine **Drives** (`/prepare`), **Review** (`/routes`), **Roads** (`/roads`) e **Training dataset** (`/dataset`); **Training** (`/train`); **Models**, con **Versions** (`/models`), **Results** (`/evaluate`) e **Publish** (`/publish`). Le pagine di una sezione sono sotto il suo titolo; una pagina sola che cambia scheda senza ricaricare:

- **Overview** (`/`, la pagina iniziale): in alto i lavori in corso, una riga ciascuno (percentuale, passo, tempo rimasto; un tocco apre il log con Stop), o l'ultimo lavoro se è fallito; il prossimo passo con il suo pulsante; tre riepiloghi che portano alle pagine (Drives da preparare, Review con la percentuale confermata, Training dataset con le revisioni da importare); gli ultimi tre lavori finiti (**Show all** per gli altri). La macchina (GPU, CPU, RAM, disco), i processi sulla GPU e **Update the site** sono nel pannello della pillola del server in alto. L'elenco delle route `/routes` (Data › Review) si apre su un riepilogo: fotogrammi confermati, da te e dagli assistenti AI, quelli dell'AI da confermare, e una tessera per gruppo — In progress, To start, Done — che mostra la sua lista, uno alla volta e sempre uno, ricordato nel browser, altrimenti il primo che ha giri.
- **Data** (`/prepare`, prima “Prepare”): **Download from the comma** e **Prepare new drives**, il loro avanzamento, la tabella delle route. **Find the signs → Found by** sceglie chi propone i cartelli (Vietnamese YOLO, Our networks, Qwen3-VL) e ne mostra la misura di Evaluate › Proposers (cartelli di velocità trovati e letti giusti, secondi per fotogramma); la scelta resta sul server per tutti i browser (`data/web/proposer.json`) e, finché nessuno sceglie, è il proponente misurato meglio. Mentre una preparazione del sito è in corso mostra il suo proponente e non si cambia. Con Qwen3-VL non installato, o con meno di circa 18 GB liberi sulla GPU, **Prepare new drives** e **Propose again** restano spenti con il motivo. Senza route nuove la pagina rimanda a **Download new drives** del comma nella barra in alto.
- **Results** (`/evaluate`): **Drive**, la prima vista, rivede un giro come in macchina: il video vero della telecamera della strada del comma (20 fotogrammi al secondo; il server lo converte in MP4 la prima volta che Drive mostra il giro, circa 4 minuti per 45 minuti di giro, con `nnslr drive-video`, e intanto mostra i fotogrammi, uno al secondo, con quanto manca), sul fotogramma i cartelli di velocità letti dalle reti (verde giusto, arancione un altro numero, rosso dove non c'è un cartello, tratteggiato quelli persi) e nell'angolo il limite che l'auto mostrerebbe ora accanto a quello dei cartelli rivisti; sotto, la striscia del giro, verde dove il limite è giusto e rosso dove è sbagliato (un tocco ci va), e a lato la percentuale del giro con il limite giusto. Il limite è l'ultimo cartello letto, tenuto fino al prossimo o a una fine limite: un solo cartello letto male resta finché non se ne legge un altro, come farebbe l'auto; la strada a cui si applica (il decider) non c'è ancora, ogni cartello letto conta. Si sceglie il giro, la versione del classificatore (con il cutter con cui è stata misurata) e **Test drives** (strade che le reti non hanno mai visto in training: come andrebbero su una strada nuova) o **Validation drives** (usate durante il training per scegliere quando fermarsi, quindi un po' più facili). **Scores** ha i numeri di ogni versione per rete (Proposers, Chain, Cutter, Classifier, Decider, AI reviewers): due versioni si confrontano solo se misurate allo stesso modo sullo stesso test.
- **Models** (`/models`): in alto le tre reti in uso (cutter, classifier, decider) con i loro numeri; sotto, le versioni di una di esse (lo switch, o un tocco sulla sua scheda) in una tabella da confrontare (cartelli di velocità, tutti i cartelli, per il classificatore la catena con il cutter; sotto ogni numero su quanti cartelli è misurato); una riga apre la sua scheda con **Use this one**. **Publish** (`/publish`) prepara e pubblica il pacchetto delle tre reti in uso.
- **Roads** (`/roads`): il tipo di strada di ogni fotogramma dei giri preparati (città, fuori città, semiautostrada, autostrada, parcheggio), letto dalla mappa OpenStreetMap locale e dal GPS del comma, senza rete. Il pulsante **Find the road types** lancia `helpers/road_types.sh`: GPS dei giri che non l'hanno ancora, strade della mappa vicino ai giri (circa 4 minuti, solo quando c'è un giro o una mappa nuova), tipo di strada di ogni fotogramma in `data/derived/road_types/`. Parte al clic, senza domanda, e non serve aver rivisto niente; dice quanti giri preparati non hanno ancora i tipi. Sotto, ogni giro è una striscia sottile colorata per tipo di strada, con le percentuali; la legenda ha sei voci: Town, Rural, Expressway, Motorway, Parking (viola) e Unknown (grigio). **Parking** è un parcheggio (un parcheggio o una sua corsia sulla mappa): lì non c'è un limite generale, per sunnypilot il limite è sempre 0; è solo un tipo di luogo che la rete impara. Sono proposte della mappa; i cartelli e le loro revisioni non cambiano. Un tocco su un giro lo apre (`/roads?drive=<route>`, **‹ Roads** o Indietro tornano all'elenco): le sue foto una alla volta, con il tipo di strada (Town, Rural, Expressway, Motorway, Parking o Unknown), il perché in parole semplici ("The map says 50 here", "A motorway ramp on the map: check it", "No GPS here (garage, tunnel or the start of the drive)", "Set by you"; niente quando la mappa non ha un motivo da dire), **Not sure** dove la mappa sbaglia spesso e **Reviewed** dove l'ha deciso una persona; sotto, la striscia del giro (una linea arancione sotto i tratti da controllare, una scura sotto quelli rivisti, una blu sopra il tratto che cambia un tocco; un tocco o un trascinamento ci va), Play, le frecce e la barra spaziatrice. Per correggere: un tocco su Town / Rural / Expressway / Motorway / Parking / Can't tell dà quel tipo al tratto mostrato, cioè i fotogrammi vicini dello stesso tipo, ugualmente sicuri o incerti e non ancora rivisti (su una rampa incerta solo i fotogrammi incerti, mai l'autostrada sicura intorno né un tratto già corretto), oppure, con **Mark start**, dal punto segnato a qui. Si salva subito; un tratto di più di 60 foto (un minuto) chiede prima conferma ("Set 485 pictures (8 min) to Town?"); **Undo last change** toglie l'ultima correzione. **Next to check** va al prossimo tratto sconosciuto o incerto non ancora rivisto e dice quanti tratti restano ("31 stretches"). Sul telefono foto, tipo, pulsanti e Next to check stanno in una schermata, sopra la barra in basso, e dopo Next to check la pagina torna sulla foto; velocità, Mark start, Undo e legenda sono sotto. Con il telefono in orizzontale i pulsanti sono accanto alla foto. Le correzioni stanno a parte, in `data/annotations/road_types/<route>.jsonl` (una riga per correzione, vince l'ultima), e restano dopo **Find again**: i tipi della mappa vengono riscritti, le correzioni ci vanno sopra.
- **Training dataset** (`/dataset`): le revisioni nuove o cambiate da importare con **Import reviews** (o *Dataset is up to date*), l'import in corso, e **In the dataset** (fotogrammi e oggetti, oggetti per tipo, revisioni di persone e AI, l'ultimo import, lo split, la tabella per route). Aprire la pagina non importa niente.
- **Training** (`/train`): la riga dei dati (oggetti, fotogrammi, route, le revisioni non importate) porta a Training dataset; **Train a model** con la scelta del punto di partenza (**Pre-trained on ImageNet**, consigliata; **Random weights**; **The stable model**, quando su Hugging Face ce n'è uno; un training precedente finito su questo server) e **Start training**; **Train the classifier** (lancia `scripts/run_classifier.sh` con un nome generato, `AAAA-MM-GG_HHMM-classifier`: box falsi del cutter in uso, classificatore con "not a sign", valutazione da solo e della catena; serve un cutter in uso, scelto nella scheda Models); i **grafici dell'allenamento** (loss del detector, loss e accuratezza del reader, epoca per epoca); in alto la scelta della rete (Cutter, Classifier, Decider, Pipeline); la scheda di un training in corso mostra con che parametri è stato lanciato (dimensione, segnali, epoche, batch…, letti dall'inizio del suo log); l'ultima versione della rete scelta in una riga con il suo numero, che porta alle sue versioni nella scheda Models (`/models#role-classifier`): la scelta di una versione si fa solo lì.

I vecchi indirizzi (`/pannello`, `/stato`, `/panoramica`, `/prepara`, `/allena`) portano alle nuove schede. Le conferme (avvio di un training, Stop, Update the code, uscita da tutti i browser) sono finestre del sito, non del browser.

Ogni lavoro in corso ha una scheda dal vivo: percentuale (solo quando si sa davvero: dai passi fatti, dalle epoche e dai batch, dalle immagini controllate, dalle route pronte; altrimenti una barra che scorre), i passi spuntati con quello attuale e la sua barra, il dettaglio ("detector · epoch 5 of 20"), il tempo trascorso e, con la percentuale, il tempo che manca (circa), l'ultima riga del log, piccoli indicatori di GPU, memoria GPU, CPU e RAM, **Stop** e **Details** (il log). Il progresso lo calcola il server dal log del lavoro (`nnslr_tools/job_progress.py`) e da `data/logs/progress.json`, dove i comandi lunghi (controllo del dataset, split, training) scrivono a che punto sono (`nnslr_tools/progress.py`). **Stop** funziona anche dopo un riavvio del sito: il numero del processo è in `data/logs/panel/<lavoro>.pid`. Un training lanciato da terminale si vede con passi e grafici (dal suo `data/logs/train-<run>.log`) ma si ferma dal terminale. Un'azione spenta ha sotto il motivo. La finestra di **Stop** dice cosa resta per ogni azione: download e installazione di Qwen3-VL riprendono da dove erano, il training della pipeline ha **Resume**, una misura e **Propose again** ripartono da capo. Un lavoro fallito dice in una frase cosa è andato storto quando il log mostra un guasto noto (GPU senza memoria, disco pieno, comma che non risponde, ...) e ha **Try again** se è un download, una preparazione, l'installazione di Qwen3-VL, un import o una misura di un proponente; codice di uscita e percorsi restano in **Details**, che dice anche quanto è durato e ha **Stop** mentre il lavoro gira. La misura di un proponente si chiama col suo nome ("Measure Qwen3-VL") e conta i fotogrammi fatti; **Propose again** ha un solo tempo che manca, quello del server. L'elenco dei lavori recenti mostra gli ultimi 12. **Update the code** lancia `helpers/update.sh --no-link` (sezione 3), cioè `git pull` e riavvio del sito; parte solo se non ci sono preparazioni, training o revisioni AI in corso.

**Download from the comma** lancia `helpers/fetch_comma_segments.sh` senza argomenti (passo 0): tutti i giri ancora sul comma, via ssh con la chiave di questa macchina (`Host comma` in `~/.ssh/config`), senza bisogno di `ssh -A`. Per leggere i log del comma (durata del giro, retromarcia) usa `COMMA_FETCH_PYTHON` se è impostata, altrimenti il Python di `../openpilot.distill/.venv`, che ha il `LogReader` di openpilot; senza nessuno dei due il pulsante resta spento con il motivo. Non parte insieme a una preparazione: tutti e due scrivono i collegamenti in `data/raw/routes`. Dopo il download il prossimo passo è **Prepare new drives**.

Il pannello lancia solo le sue azioni fisse, mai un comando qualsiasi, e una sola per tipo alla volta (download, preparazione, import/training, revisione AI), contando anche quelle partite da terminale, compreso il download di `openpilot.distill`. È protetto come il resto del sito: token sulla LAN, e i pulsanti accettano solo richieste dalla pagina stessa. I log delle azioni sono in `data/logs/panel/`; l'ultima riga di ognuno è il codice di uscita.

`NNSLR_PANEL_ACTIONS=0` (nell'ambiente del server, per il servizio in `/etc/speedlimits-web.env`) lascia il pannello in sola lettura, per esempio se il sito lo usano altre persone per revisionare.

## 3. Il sito sempre acceso: il servizio `speedlimits-web`

Per avere sito e pannello sempre accesi, anche dopo un riavvio del server, si installa il servizio systemd una volta sola (da root):

```sh
cd ~/sunnypilot.speedlimits
helpers/2_review.sh --stop                        # se avevi acceso la revisione a mano (porta 8765)
scripts/install_review_service.sh --dry-run       # mostra cosa farebbe, non cambia niente
scripts/install_review_service.sh                 # installa, avvia e stampa il link
```

Il servizio ascolta su `0.0.0.0:8765`, riparte da solo se cade e a ogni boot. Rilanciare l'installazione non fa danni: riscrive il servizio solo se `deploy/speedlimits-web.service` è cambiato e non tocca il token.

### Entrare dal browser: nome e password

Sulla LAN il sito chiede di accedere con **nome e password** (pagine, pannello e immagini). La **prima visita** crea l'amministratore: la pagina chiede il **codice di configurazione**, che è il token di accesso del server, poi il nome e la password (almeno 12 caratteri). Da lì in poi il token non apre più il sito nel browser: serve solo come chiave dell'API `/api/v1`, di MCP e degli agenti che lo mandano nell'header `X-Review-Token`.

- La pagina **Account** (in alto) mostra chi è connesso, fa uscire (anche da tutti i browser) e cambia la password; dopo il cambio gli altri browser devono accedere di nuovo.
- Una sessione dura 7 giorni senza uso, 30 in tutto. 5 tentativi sbagliati da un indirizzo lo bloccano per 15 minuti.
- I file dell'accesso sono nella cartella dati, `web/auth/` (`admin.json`, `sessions.json`, permessi 600; la password è salvata solo come hash scrypt).
- **Password dimenticata**: sul server, nella cartella del clone, `helpers/reset_password.sh`. Ferma il sito, sposta `web/auth/admin.json` della cartella dati in `admin.json.<ora>.old` e riaccende il sito (il servizio, o quello acceso da `helpers/2_review.sh`): la visita successiva chiede di nuovo il codice di configurazione, poi un nome e una password nuovi. Revisioni e impostazioni restano; tutti i browser devono accedere di nuovo. La pagina di accesso lo ricorda sotto **Forgot the password?**.
- I vecchi link con `?token=...` non aprono più il sito: portano alla pagina di accesso.
- Avviato a mano solo su `127.0.0.1` (senza `--host 0.0.0.0`) il sito resta aperto a questo computer senza accesso, come prima.

### Il token (codice di configurazione e chiave dell'API)

Il posto dove cercarlo è il file **`.env` del repository** (`~/sunnypilot.speedlimits/.env`), riga `NNSLR_REVIEW_TOKEN=...`:

```sh
grep NNSLR_REVIEW_TOKEN ~/sunnypilot.speedlimits/.env
```

- `.env` è in `.gitignore`: non finisce mai in git né su GitHub. Ha permessi 600 (lo legge solo root).
- Lo crea la prima volta chi serve per primo: `helpers/2_review.sh` o `scripts/install_review_service.sh`. L'installazione del servizio usa il token già presente in `.env`; se non c'è, ne crea uno e lo scrive in `.env`.
- Il servizio systemd lo legge da una sua copia, `/etc/speedlimits-web.env` (permessi 600); ogni installazione rimette in `.env` lo stesso valore.
- L'indirizzo del sito (`http://<IP del server>:8765/`) lo stampano l'installazione, `helpers/2_review.sh` e `helpers/update.sh`, senza il token.

Il token non va condiviso: chi lo ha può creare l'amministratore finché non esiste e usa l'API. Per cambiarlo: scrivi il nuovo valore (almeno 16 caratteri tra lettere, cifre, `-` e `_`) in `/etc/speedlimits-web.env`, poi `scripts/install_review_service.sh` (lo copia in `.env`) e `systemctl restart speedlimits-web`.

### Aggiornare il codice

```sh
cd ~/sunnypilot.speedlimits
helpers/update.sh
```

`helpers/update.sh` fa `git pull --ff-only` (si ferma se ci sono modifiche locali), aggiorna `.venv` se è cambiato `pyproject.toml`, rilancia `scripts/install_review_service.sh` (riscrive il servizio solo se è cambiato) e riavvia il servizio; senza servizio riavvia il sito acceso da `helpers/2_review.sh`. Alla fine stampa il link (non con `--no-link`, che è come lo lancia il pulsante **Update the site** nel pannello del server, la pillola GPU/CPU in alto a destra). Il pulsante è spento, con il motivo, mentre gira un lavoro che l'aggiornamento non può interrompere; dopo la conferma il sito segue il lavoro e ricarica la pagina solo se è riuscito, altrimenti scrive "Not updated." e perché (nel pannello e in un avviso in basso).

Il servizio ha `KillMode=process`: un riavvio ferma solo il sito, non una preparazione o un training partiti dal pannello. Dopo il riavvio il pannello li ritrova come "in corso".

### Controllare e fermare

```sh
systemctl status speedlimits-web                     # acceso? da quanto?
journalctl -u speedlimits-web -n 50                  # ultime righe del server (anche "Pannello: avviata ...")
systemctl stop speedlimits-web                       # spegne (riparte al prossimo boot)
scripts/install_review_service.sh --uninstall     # toglie il servizio (il token resta)
```

### API REST e MCP (revisione senza browser)

Lo stesso servizio offre anche la **REST API** (`/api/v1/...`) e, con le dipendenze opzionali MCP, il server **MCP** su `/mcp`, per revisionare con `curl`, uno script o un'AI (Claude Code, Codex) che gira sul proprio computer. Tutto è in [API_MCP.md](API_MCP.md). In breve:

- token **nell'header** `Authorization: Bearer <NNSLR_REVIEW_TOKEN>`, mai nell'indirizzo; serve anche su 127.0.0.1;
- stesse regole e stesso database della pagina; il pannello (Prepara, Importa, Allena, Aggiorna) non è raggiungibile da API e MCP;
- installare una volta le dipendenze MCP e riavviare il sito: `helpers/setup_mcp.sh --restart` (sezione "Installazione locale per MCP" qui sotto);
- MCP anche senza HTTP, via ssh: `helpers/mcp.sh` (con `--read-only` per una prova senza salvataggio; `NNSLR_MCP_READ_ONLY=1` nel servizio fa lo stesso su `/mcp`);
- le revisioni di un'AI sono salvate come `ai/<nome>`, tipo `ai`, e non cambiano le conferme di una persona. Sono definitive come quelle di una persona: `helpers/3_train.sh` le importa nel dataset senza un'altra approvazione, con la provenienza AI (`ai/<nome>`, `label_basis` `ai_review`, canale). Confermare non importa e non allena: import e training si lanciano a parte.

#### Installazione locale per MCP

Alcune cose non stanno in git perché sono installazioni o impostazioni della macchina. Le fa tutte `helpers/setup_mcp.sh`, che si può rilanciare quando si vuole (salta quello che c'è già):

```sh
cd ~/sunnypilot.speedlimits
helpers/setup_mcp.sh                         # solo l'SDK MCP in .venv (extra [mcp])
helpers/setup_mcp.sh --claude                # + Claude Code in questa cartella, per le sessioni di revisione
helpers/setup_mcp.sh --claude --effort=low   # come sopra, con un effort diverso (default medium)
helpers/setup_mcp.sh --restart               # + copia del database se va migrato, e riavvio di speedlimits-web
```

| Cosa | Dove finisce | Perché non è in git |
| --- | --- | --- |
| SDK MCP (`mcp` 2.x e dipendenze) | `.venv` | pacchetti installati |
| server MCP `nnslr-review` (stdio, `helpers/mcp.sh`) per Claude Code | `~/.claude.json`, per la cartella del repository | configurazione di Claude Code della macchina |
| strumenti `mcp__nnslr-review` permessi senza chiedere, `effortLevel` delle sessioni | `.claude/settings.local.json`, elencato in `.git/info/exclude` | impostazione locale: le sessioni di revisione lanciate qui (anche quelle remote dell'ambiente bridge) non si fermano a chiedere e non usano l'effort alto |
| copia del database prima della migrazione | `data/annotations/review.sqlite.bak-v1-<ora>` | dati |

Per togliere i permessi alle sessioni basta cancellare `.claude/settings.local.json`; il server MCP si toglie con `claude mcp remove -s local nnslr-review`.

Revisioni fatte da un'AI dalla pagina (con un nome come `Codex` o `ChatGPT`) risultano umane: `nnslr review-reclassify-ai --reviewer Codex --reviewer ChatGPT --dry-run` conta quelle da cambiare, senza `--dry-run` le segna come AI (nuova revisione nella cronologia, stessi riquadri). Fare prima una copia del database.

Il database delle revisioni si aggiorna da solo alla versione 2 al primo avvio del codice nuovo (aggiunge tipo di revisore e canale; le revisioni già fatte restano "umane, dalla pagina"). Dopo, un `nnslr` vecchio non lo apre più.

### Il revisore AI (Settings)

Una revisione si può fare a mano o con un'AI che vede le immagini, attraverso un qualsiasi server compatibile con l'API OpenAI (`/v1/models` e `/v1/chat/completions`): strata, il proprio `llama-server` di llama.cpp su un'altra macchina con un modello visivo e il suo file `--mmproj` (per esempio Qwen3-VL-32B-Instruct), o un altro servizio.

- **Collegare un provider**: menu **Account → Settings**, riquadro **AI reviewer**: nome, indirizzo che finisce in `/v1`, chiave API (se il server la vuole) e modello (la lista viene dal server; se non risponde, si scrive a mano). **Test** manda un fotogramma rivisto da una persona, senza le proposte, e dice se il modello risponde in JSON e legge il cartello di velocità; se il server dichiara il modello solo testo, lo dice senza chiedergli niente. Sotto, **Who reviews** è un elenco con il segno di spunta: si sceglie un provider che ha passato il Test, oppure **Don't use AI**. Un provider mai provato, o che non ha passato l'ultimo Test, non si può scegliere (né da terminale), e se il revisore scelto non lo passa più **Ask AI** e **Review with AI** restano spenti, con il motivo, finché non lo ripassa. Cambiare indirizzo o modello cancella il Test.
- **Da terminale**, per esempio strata:

  ```sh
  .venv/bin/nnslr llm-provider add --name strata --base-url https://strata.farm.14bodhi.com/v1 --key-file /root/.config/strata-api-key
  .venv/bin/nnslr llm-provider test strata
  .venv/bin/nnslr llm-provider use strata
  ```

  `nnslr llm-provider list` elenca i provider (mai le chiavi); poi `models ID`, `remove ID`, `use none`. Senza `--model` prende il primo modello che il server elenca.
- **Dove sta la chiave**: in `data/web/private/llm-providers.json` (permessi 600, cartella 700, fuori da git); `NNSLR_LLM_PROVIDERS=/percorso/completo.json` sposta il file. La chiave non esce mai dal server: nessuna pagina, risposta, log o resoconto la contiene, e il browser sa solo che c'è. Lasciare vuoto il campo della chiave tiene quella salvata; **Remove key** la toglie.
- **Sulla pagina di revisione**: **Ask AI** (tasto **I**) mostra il suggerimento dell'AI per il fotogramma, con Keep, Correct, Add, Delete o Not judged per ogni riquadro. Non salva niente: **Apply** lo trasforma in modifiche da controllare, e il tuo **Confirm** le salva come tua revisione, come sempre; **Dismiss** lo scarta. Al massimo due richieste all'AI alla volta.
- **Una route intera**: nell'elenco delle route, **Review with AI** sulle route non finite (finché in Settings non c'è un revisore AI scelto, al loro posto c'è un solo link **Set up AI review**, e nella pagina di revisione una riga con lo stesso link). **Try without saving** prova 20 fotogrammi senza salvare: i suggerimenti finiscono in `data/ai_reviews/<route>/` e la pagina di revisione della route li mostra, nel riquadro **AI reviewer** di ogni fotogramma provato (con **Apply** e **Dismiss**, come **Ask AI**) e nel filtro **AI trial**; **Review and save** salva le revisioni dell'AI, solo con un provider che ha passato il Test; restano da confermare (nella pagina **AI · to confirm**, nell'elenco «N by AI to confirm») e vanno nel training solo dopo la conferma di una persona: l'import le lascia fuori, e toglie quelle importate prima di questa regola. Parte come lavoro del pannello (`helpers/ai_review.sh`, si segue nella Overview), mai insieme a una preparazione o a un aggiornamento. I fotogrammi confermati da una persona non vengono mai toccati.
- **Le revisioni salvate dall'AI sono definitive**: nessuno le riapprova, e il prossimo import (`helpers/3_train.sh`, **Import reviews**) le mette nel training come revisioni AI: `reviewer` `ai/<provider>/<modello>`, tipo `ai`, canale `web`, `label_basis` `ai_review`. Salvare non importa e non allena.
- **Misurare un provider**: Evaluate → **AI reviewers**, **Measure** (`helpers/measure_ai_reviewer.sh`): fino a 60 fotogrammi rivisti da persone, sempre gli stessi per tutti i provider; cartelli di velocità trovati e letti giusti, accordo sui fotogrammi senza cartelli, riquadri falsi e secondi per fotogramma. Non salva niente; il risultato va in `data/evaluations/ai_reviewers/`.
- **Chi può**: solo un browser che ha fatto l'accesso, o il sito su 127.0.0.1, cambia i provider e usa l'AI. Il token (API, MCP, agenti) può solo leggere le impostazioni, lo stato dell'AI e le misure ([API_MCP.md](API_MCP.md), sezione 8).

## 4. Dove stanno le cose

| Cosa | Dove |
| --- | --- |
| Video originali (condivisi con distill) | `/root/comma-videos/<route>--<segmento>/` |
| Link ai video, per route | `data/raw/routes/<route>/<segmento>/` |
| Frame estratti | `data/derived/frames/<route>/` |
| Video dei giri per Results › Drive (`nnslr drive-video`) | `data/derived/video/<route>.mp4` e `.json` |
| Cartelli proposti (report) | `data/derived/preannotations/<route>/` |
| Revisioni confermate | `data/annotations/review.sqlite` |
| Dataset importato | `data/annotations/objects.jsonl` |
| Split train/validation/test | `data/splits/` (`latest.json` è quello in uso) |
| Training, valutazione, bundle | `data/runs/<run>/` |
| Log degli helper | `data/logs/` (`prepare/<route>.log`, `train-<run>.log`, `review.log`) |
| Log delle azioni del pannello | `data/logs/panel/` |
| Provider AI di Settings e le loro chiavi | `data/web/private/llm-providers.json` (permessi 600; `NNSLR_LLM_PROVIDERS` lo sposta) |
| Suggerimenti delle revisioni AI di una route | `data/ai_reviews/<route>/` |
| Misure dei revisori AI | `data/evaluations/ai_reviewers/` |
| Token di accesso al sito (il posto dove cercarlo) | `.env` del repository, riga `NNSLR_REVIEW_TOKEN=` |
| Copia del token letta dal servizio | `/etc/speedlimits-web.env` |
| Definizione del servizio | `deploy/speedlimits-web.service` → installato in `/etc/systemd/system/speedlimits-web.service` |

`data/` e `.env` non sono in git.

### Il codice delle pagine del sito

| Cosa | Dove |
| --- | --- |
| Sorgenti delle pagine: componenti Vue 3 (`.vue`) e TypeScript; un ingresso per pagina, i componenti e le funzioni comuni | `web/src/pages/<pagina>.ts`; `web/src/review/` (revisione), `web/src/control/` (Panoramica, Prepara, Allena, grafici), `web/src/account/`, `web/src/login/`; comuni in `web/src/components/` (menu Account, avvisi, barre), `web/src/composables/`, `web/src/common/` |
| Template HTML, stili (`shared.css`: colori e cornice comuni con distill-web) | `src/nnslr_tools/web/` |
| Script compilati (committati: il server e `helpers/update.sh` non hanno bisogno di Node) | `src/nnslr_tools/web/js/<pagina>.js` |
| Chi mette insieme pagina, stili e script, con il nonce (la pagina Vue si disegna in `<div id="app">`) | `src/nnslr_tools/pages.py` |

Per cambiare una pagina: modifica `web/src` (o il template), lancia `helpers/build_web.sh`, poi committa sorgenti e `js/` insieme. I componenti non hanno `<style>`: gli stili restano nei template e in `shared.css`. Vue è compilato prima (niente `eval`, che la CSP del sito vieta); un test fallisce se `js/` non corrisponde ai sorgenti. HTML, CSS e `js/` sono riletti a ogni richiesta: basta ricaricare la pagina. `pages.py` e il resto del Python valgono dopo il riavvio del servizio.

## 6. Script del repository

| Script | Cosa fa | Effetti |
| --- | --- | --- |
| `helpers/status.sh` | stato di tutto e prossimo comando | solo lettura |
| `helpers/1_prepare.sh` | video → frame → cartelli proposti | scrive in `data/`, CPU |
| `helpers/2_review.sh` | accende/spegne il sito di revisione, stampa il link | apre la porta 8765 sulla LAN, crea il token in `.env` |
| `helpers/3_train.sh` | revisioni → dataset → modello | scrive il dataset, installa `.venv-gpu` la prima volta, GPU |
| `helpers/update.sh` | aggiorna il codice e riavvia il sito | `git pull`, `.venv`, riavvio del servizio |
| `helpers/reset_password.sh` | password dimenticata: l'amministratore riparte da capo (la prossima visita chiede il codice di configurazione) | ferma e riaccende il sito, sposta `web/auth/admin.json` |
| `helpers/setup_mcp.sh` | installa l'SDK MCP; `--claude` prepara Claude Code in questa cartella; `--restart` copia il database se va migrato e riavvia il sito | pacchetti in `.venv`, `~/.claude.json`, `.claude/settings.local.json`, riavvio del servizio |
| `helpers/mcp.sh` | server MCP (stdio) per Claude Code, Codex e altri client, anche via ssh ([API_MCP.md](API_MCP.md)) | legge le route; salva revisioni AI nel database (non con `--read-only`) |
| `helpers/fetch_comma_segments.sh` | scarica i giri nuovi dal comma | rete, cartella condivisa dei video |
| `helpers/road_types.sh [--again]` | il tipo di strada di ogni fotogramma dei giri preparati, dalla mappa e dal GPS: a. `nnslr gps --all --missing`, b. `scripts/osm_roads.py` se c'è un giro o una mappa nuova (`--again`: sempre), c. `nnslr road-types --all`; è il pulsante **Find the road types** di Data › Roads | scrive in `data/derived/`, CPU per qualche minuto quando legge la mappa, senza rete |
| `helpers/build_web.sh` | controlla i tipi delle pagine (Vue 3 e TypeScript in `web/src`, con `vue-tsc`) e le compila con Vite in `src/nnslr_tools/web/js`; `--check` dice solo se i file compilati sono aggiornati | la prima volta installa Vue, Vite e TypeScript in `web/node_modules` (Node da PyPI con `uvx`, niente nel sistema) |
| `scripts/install_review_service.sh` | installa/aggiorna/toglie il servizio `speedlimits-web` | systemd, `/etc/speedlimits-web.env`, `.env` |
| `scripts/setup_v100_training_env.sh` | crea `.venv-gpu` (lo lancia anche `helpers/3_train.sh`) | installa pacchetti |
| `scripts/setup_vlm.sh [--no-test]` | installa Qwen3-VL-8B-Instruct (vede le immagini, Apache-2.0, ~17.5 GB) in `data/models/vlm/` e `transformers`, `accelerate`, `qwen-vl-utils` in `.venv-gpu` (nessun pacchetto già presente cambia versione), poi lo prova sulla GPU su un fotogramma; per ora nessuno lo usa ([INSTALL.md](INSTALL.md)) | installa pacchetti, scarica 17.5 GB, GPU |
| `uv run --no-project --with osmium python scripts/osm_places.py` | estrae dalla mappa OpenStreetMap locale (`data/raw/osm/`) i nomi di città, paesi e quartieri svizzeri in `data/derived/osm/places.jsonl`: la lista Review chiama ogni giro con il giorno, l'ora e da dove a dove ("Herznach → Zürich") | scrive dati, senza rete |
| `helpers/measure_proposer.sh [vietnam\|ours\|qwen]` | misura un proponente sui fotogrammi di test rivisti, come lo userebbe Prepare (`nnslr evaluate-proposer`): cartelli di velocità trovati e letti giusti, tutti i cartelli trovati, proposte false per fotogramma, secondi per fotogramma; è il pulsante **Measure** della vista Proposers di Evaluate | scrive solo `data/evaluations/proposers/`; `ours` e `qwen` usano la GPU |
| `helpers/ai_review.sh <route> [--provider=ID] [--save] [--limit=N]` | revisiona con l'AI di Settings i fotogrammi non confermati di una route (`nnslr ai-review`); senza `--save` è una prova e salva solo i suggerimenti in `data/ai_reviews/<route>/`; è il pulsante **Review with AI** dell'elenco delle route | rete (il server del provider); con `--save` salva nel database revisioni AI da confermare (l'import le prende solo dopo la conferma di una persona) |
| `helpers/measure_ai_reviewer.sh [--provider=ID] [--limit=N]` | misura un provider AI sui fotogrammi rivisti da persone (`nnslr measure-ai-reviewer`); è il pulsante **Measure** di Evaluate → AI reviewers | rete, scrive solo `data/evaluations/ai_reviewers/` |
| `helpers/1_prepare.sh --again` | ripropone i cartelli dei fotogrammi non ancora confermati di ogni route, con il proponente scelto; i fotogrammi confermati restano come sono (copiati sul nuovo report prima che diventi quello attuale, `nnslr preannotate --keep-reviews`), una route tutta confermata resta fuori, e ogni route è in sola lettura finché le sue nuove proposte non sono pronte (`NNSLR_PROPOSER=vietnam`, `ours` o `qwen`, la scelta **Find the signs → Found by** della scheda Prepare) | scrive le proposte in `data/derived/preannotations/`; `ours` e `qwen` usano la GPU |
| `scripts/run_v100_baseline.sh` | training completo (lo lancia `helpers/3_train.sh`) | GPU |
| `scripts/run_experiment_b.sh <nome>` | esperimento B ([NEURAL_NETWORK.md](NEURAL_NETWORK.md), passo 2): allena una dopo l'altra le varianti `crop`, `crop_geom` e `full` di `nnslr experiment-b` sui box revisionati e stampa il confronto (lettura, other_road, macro-F1, astensioni) su test e sulle righe umane; prima `nnslr experiment-b --variant full --dry-run` (solo CPU) | GPU, scrive in `data/runs/<nome>/experiment-b/` |
| `scripts/run_detector_sizes.sh <nome> [misure...]` | la misura d'ingresso del detector ([NEURAL_NETWORK.md](NEURAL_NETWORK.md), passo 3): allena un detector per misura (di default `320x320 640x360 960x544`, `nnslr train --task detector --input-size`) con le stesse impostazioni (il box predefinito più piccolo è 0.02 dell'immagine, `NNSLR_DETECTOR_ANCHOR_MIN_RATIO`; il 10% dei fotogrammi senza cartelli per epoca; l'epoca migliore è quella con la recall di validazione più alta alla soglia 0.1; i box sono i cartelli di velocità e gli altri cartelli stradali, `NNSLR_CUTTER_FAMILIES=all`, o solo quelli di velocità con `speed`), lo valuta su test e validation e stampa il confronto (recall totale, dei soli cartelli di velocità e per larghezza del cartello alle soglie 0.05, 0.1, 0.25, 0.3 e 0.5, box per fotogramma passati al classificatore, box falsi per fotogramma, precisione, ms per fotogramma, s per epoca, memoria GPU, epoca migliore), salvato in `comparison.txt`; rilanciato con lo stesso nome finisce le misure mancanti; prima `nnslr train --task detector --input-size 640x360 --dry-run` (solo CPU) | GPU, scrive in `data/runs/<nome>/detector-sizes/` |
| `scripts/run_classifier.sh <nome>` | il classificatore ([NEURAL_NETWORK.md](NEURAL_NETWORK.md), passo 2; è anche il pulsante **Train the classifier** della scheda Train): raccoglie i box falsi del cutter in uso (`nnslr mine-negatives`: punteggio ≥ 0.1, nessuna sovrapposizione con i box revisionati, al massimo 3 per fotogramma e 3 volte i cartelli di ogni partizione, prima dai fotogrammi rivisti da una persona), allena il classificatore con `not_a_sign` e i tipi degli altri cartelli (`nnslr train --task reader --classes v2 --negatives`, da ImageNet, ritagli più scuri e box spostati), lo valuta da solo su test e validation, valuta la catena cutter → classificatore (`nnslr evaluate-chain`: cartelli trovati e letti giusti, box falsi per fotogramma che sopravvivono, alle soglie 0.05, 0.1 e 0.25), fa la stessa catena con il vecchio reader e stampa il confronto, salvato in `comparison.txt`; rilanciato con lo stesso nome finisce i passi mancanti; prima `nnslr mine-negatives --dry-run` e `nnslr train --task reader --classes v2 --dry-run` (solo CPU) | GPU, scrive in `data/runs/<nome>/` |
| `scripts/publish_model.sh` | pubblica un bundle su Hugging Face | upload pubblico |

## 5. Problemi comuni

| Messaggio | Cosa fare |
| --- | --- |
| `servono almeno 3 route (o luoghi) indipendenti revisionate` | Prepara e revisiona altre route, poi rilancia `helpers/3_train.sh`. |
| `la GPU 0 e' occupata` / pulsante "la GPU è occupata" | Sta girando un altro training (spesso `openpilot.distill`): aspetta che finisca. Da terminale `--force` parte lo stesso, più lenti tutti e due. |
| `la porta 8765 e' gia' occupata` | C'è già un sito acceso (il servizio o un altro `nnslr review`): `helpers/2_review.sh` ristampa il link, `helpers/2_review.sh --stop` spegne quello acceso a mano. |
| `Sign-in required: reload the page and sign in` | La sessione è finita (7 giorni senza uso, uscita, password cambiata): ricarica e accedi con nome e password (sezione 3). |
| API o MCP: `401 unauthorized`, `501 mcp_unavailable`, `409 ...` | Vedi [API_MCP.md](API_MCP.md), sezione 7. |
| `Sign in on the site to use the AI reviewer.` (403 `session_required`) | Il revisore AI si usa solo da un browser che ha fatto l'accesso, o dal sito su 127.0.0.1: il token può solo leggere. |
| `Choose an AI reviewer in Settings.` (409 `no_ai_reviewer`) | Account → Settings → AI reviewer: aggiungi un provider, premi **Test** e sceglilo in **Who reviews**, oppure `nnslr llm-provider test ID` e `nnslr llm-provider use ID`. |
| `... did not pass its last Test ...` / `Test ... in Settings first ...` (409 `ai_not_tested`) | Il provider non ha passato il Test (o non è mai stato provato): sistemalo e premi **Test** in Settings (`nnslr llm-provider test ID`). Solo dopo può revisionare. |
| 429 `ai_busy` | Due richieste all'AI sono già in corso: riprova tra poco. |
| 502 `ai_unreachable`, `ai_http_error`, 504 `ai_timeout` | Il server del provider non risponde, rifiuta (chiave sbagliata, modello che non c'è: il messaggio dice `HTTP <codice>`) o non risponde entro 2 minuti: controlla con **Test**. |
| Test: `... takes text only: it cannot see the frames.` | Il modello non vede le immagini: scegline uno visivo (con llama.cpp serve anche il suo file `--mmproj`). |
| `su ... non ci sono ancora pesi stable` | `--continue` vuole un modello già pubblicato come stable: allena prima da zero. |
| Una route resta "videos to link" (`helpers/status.sh`) | `link-videos` ha trovato un conflitto: vedi `data/logs/prepare/link-videos.json`. I segmenti con solo il log e senza `fcamera.hevc` (il fetch scarica solo il log dei primi e ultimi due) non sono un errore: non vengono collegati e sono elencati in `log_only`. |
| Estrazione o pre-annotazione fallita | Il motivo è in `data/logs/prepare/<route>.log`. |
