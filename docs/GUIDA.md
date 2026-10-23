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
| 0 | `helpers/fetch_comma_segments.sh` | Scarica tutti i giri ancora sul comma, anche quelli vecchi non in dashcam mode, nella cartella condivisa con `openpilot.distill` (`/root/comma-videos`); toglie i primi due e gli ultimi due segmenti e quelli con la retromarcia. Distill non li usa: allena solo su quelli che scarica lui (dashcam). |
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

### Due regole da sapere

- **Servono almeno 5 route revisionate** per allenare. `nnslr build-splits` divide il dataset in cinque parti indipendenti (train, validation, test, route-held-out, hard-negative): con meno route si ferma. Le revisioni importate restano nel dataset, non si perde niente.
- **Una route già pre-annotata non viene più rielaborata.** Una nuova pre-annotazione lascerebbe le revisioni sul report vecchio e l'import le salterebbe. I segmenti di quella route arrivati dopo restano fuori; `helpers/status.sh` li segnala.

## 2. Il pannello web

Il sito di revisione ha due pagine, raggiungibili dalla barra in alto:

- **Revisione** (`/`): le route con il loro report, da aprire e revisionare.
- **Pannello** (`/pannello`): lo stato di tutto e i pulsanti per lanciare i passi 0, 1 e 3 senza terminale.

Il pannello mostra:

- il **prossimo passo**, con il pulsante giusto;
- i tre passi con i numeri (route nuove, frame revisionati, route revisionate su 5) e i pulsanti **Scarica dal comma**, **Prepara le route nuove**, **Importa le revisioni**, **Allena da zero**, **Allena dai pesi stable**. Un pulsante spento ha sotto il motivo (per esempio "servono 5 route revisionate, ora 1" o "la GPU è occupata");
- GPU, disco, CPU e cosa sta girando, anche lanciato da terminale e anche il training di `openpilot.distill`;
- durante una preparazione, la **percentuale** e le route pronte sul totale del giro, più la route su cui sta lavorando (segmenti estratti, frame con i cartelli proposti). Ogni route vale metà per l'estrazione e metà per i cartelli. Il totale lo scrive `helpers/1_prepare.sh` in `data/logs/prepare/batch.json` quando parte: una preparazione partita prima dell'aggiornamento mostra solo le route in lavorazione;
- la tabella delle route con lo stato;
- le azioni lanciate dal pannello, con il loro stato (in corso, finita, fallita, interrotta), il pulsante **Log** (il log scorre dal vivo in basso nella pagina) e **Ferma**.

In **Azioni dal pannello** c'è anche **Aggiorna il codice**: lancia `helpers/update.sh --no-link` (sezione 3), cioè `git pull` e riavvio del sito, senza bisogno della ssh. Parte solo se non ci sono preparazioni o training in corso, e finché gira non ne parte nessuno: il pull cambierebbe gli script sotto i piedi a quelli in corso. Durante il riavvio la pagina resta qualche secondo "non raggiungibile", poi riprende da sola.

**Scarica dal comma** lancia `helpers/fetch_comma_segments.sh` senza argomenti (passo 0): tutti i giri ancora sul comma, via ssh con la chiave di questa macchina (`Host comma` in `~/.ssh/config`), senza bisogno di `ssh -A`. Per leggere i log del comma (durata del giro, retromarcia) usa `COMMA_FETCH_PYTHON` se è impostata, altrimenti il Python di `../openpilot.distill/.venv`, che ha il `LogReader` di openpilot; senza nessuno dei due il pulsante resta spento con il motivo. Non parte insieme a una preparazione: tutti e due scrivono i collegamenti in `data/raw/routes`. Dopo il download il prossimo passo è **Prepara le route nuove**.

Il pannello lancia solo queste sei azioni fisse, mai un comando qualsiasi, e una sola per tipo alla volta (download, preparazione, import/training), contando anche quelle partite da terminale, compreso il download di `openpilot.distill`. È protetto come il resto del sito: token sulla LAN, e i pulsanti accettano solo richieste dalla pagina stessa. I log delle azioni sono in `data/logs/panel/`; l'ultima riga di ognuno è il codice di uscita.

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
- **Password dimenticata**: `systemctl stop speedlimits-web`, sposta via `data/web/auth/admin.json` (cartella dati del servizio), `systemctl start speedlimits-web`: la visita successiva chiede di nuovo il codice di configurazione.
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

`helpers/update.sh` fa `git pull --ff-only` (si ferma se ci sono modifiche locali), aggiorna `.venv` se è cambiato `pyproject.toml`, rilancia `scripts/install_review_service.sh` (riscrive il servizio solo se è cambiato) e riavvia il servizio; senza servizio riavvia il sito acceso da `helpers/2_review.sh`. Alla fine stampa il link (non con `--no-link`, che è come lo lancia il pulsante **Aggiorna il codice** del pannello).

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
- le revisioni di un'AI sono salvate come `ai/<nome>`, tipo `ai`: non cambiano le conferme di una persona e `helpers/3_train.sh` non le importa nel dataset (restano nel database, elencate come `skipped_ai_reviews`).

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

## 4. Dove stanno le cose

| Cosa | Dove |
| --- | --- |
| Video originali (condivisi con distill) | `/root/comma-videos/<route>--<segmento>/` |
| Link ai video, per route | `data/raw/routes/<route>/<segmento>/` |
| Frame estratti | `data/derived/frames/<route>/` |
| Cartelli proposti (report) | `data/derived/preannotations/<route>/` |
| Revisioni confermate | `data/annotations/review.sqlite` |
| Dataset importato | `data/annotations/objects.jsonl` |
| Split train/validation/test | `data/splits/` (`latest.json` è quello in uso) |
| Training, valutazione, bundle | `data/runs/<run>/` |
| Log degli helper | `data/logs/` (`prepare/<route>.log`, `train-<run>.log`, `review.log`) |
| Log delle azioni del pannello | `data/logs/panel/` |
| Token di accesso al sito (il posto dove cercarlo) | `.env` del repository, riga `NNSLR_REVIEW_TOKEN=` |
| Copia del token letta dal servizio | `/etc/speedlimits-web.env` |
| Definizione del servizio | `deploy/speedlimits-web.service` → installato in `/etc/systemd/system/speedlimits-web.service` |

`data/` e `.env` non sono in git.

## 6. Script del repository

| Script | Cosa fa | Effetti |
| --- | --- | --- |
| `helpers/status.sh` | stato di tutto e prossimo comando | solo lettura |
| `helpers/1_prepare.sh` | video → frame → cartelli proposti | scrive in `data/`, CPU |
| `helpers/2_review.sh` | accende/spegne il sito di revisione, stampa il link | apre la porta 8765 sulla LAN, crea il token in `.env` |
| `helpers/3_train.sh` | revisioni → dataset → modello | scrive il dataset, installa `.venv-gpu` la prima volta, GPU |
| `helpers/update.sh` | aggiorna il codice e riavvia il sito | `git pull`, `.venv`, riavvio del servizio |
| `helpers/setup_mcp.sh` | installa l'SDK MCP; `--claude` prepara Claude Code in questa cartella; `--restart` copia il database se va migrato e riavvia il sito | pacchetti in `.venv`, `~/.claude.json`, `.claude/settings.local.json`, riavvio del servizio |
| `helpers/mcp.sh` | server MCP (stdio) per Claude Code, Codex e altri client, anche via ssh ([API_MCP.md](API_MCP.md)) | legge le route; salva revisioni AI nel database (non con `--read-only`) |
| `helpers/fetch_comma_segments.sh` | scarica i giri nuovi dal comma | rete, cartella condivisa dei video |
| `scripts/install_review_service.sh` | installa/aggiorna/toglie il servizio `speedlimits-web` | systemd, `/etc/speedlimits-web.env`, `.env` |
| `scripts/setup_v100_training_env.sh` | crea `.venv-gpu` (lo lancia anche `helpers/3_train.sh`) | installa pacchetti |
| `scripts/run_v100_baseline.sh` | training completo (lo lancia `helpers/3_train.sh`) | GPU |
| `scripts/publish_model.sh` | pubblica un bundle su Hugging Face | upload pubblico |

## 5. Problemi comuni

| Messaggio | Cosa fare |
| --- | --- |
| `servono almeno 5 route (o luoghi) indipendenti revisionate` | Prepara e revisiona altre route, poi rilancia `helpers/3_train.sh`. |
| `la GPU 0 e' occupata` / pulsante "la GPU è occupata" | Sta girando un altro training (spesso `openpilot.distill`): aspetta che finisca. Da terminale `--force` parte lo stesso, più lenti tutti e due. |
| `la porta 8765 e' gia' occupata` | C'è già un sito acceso (il servizio o un altro `nnslr review`): `helpers/2_review.sh` ristampa il link, `helpers/2_review.sh --stop` spegne quello acceso a mano. |
| `Accesso richiesto: ricarica la pagina e accedi` | La sessione è finita (7 giorni senza uso, uscita, password cambiata): ricarica e accedi con nome e password (sezione 3). |
| API o MCP: `401 unauthorized`, `501 mcp_unavailable`, `409 ...` | Vedi [API_MCP.md](API_MCP.md), sezione 7. |
| `su ... non ci sono ancora pesi stable` | `--continue` vuole un modello già pubblicato come stable: allena prima da zero. |
| Una route resta "da collegare" | `link-videos` ha trovato un conflitto: vedi `data/logs/prepare/link-videos.json`. I segmenti con solo il log e senza `fcamera.hevc` (il fetch scarica solo il log dei primi e ultimi due) non sono un errore: non vengono collegati e sono elencati in `log_only`. |
| Estrazione o pre-annotazione fallita | Il motivo è in `data/logs/prepare/<route>.log`. |
