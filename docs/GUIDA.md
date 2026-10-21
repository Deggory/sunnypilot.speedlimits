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
| 0 | `helpers/fetch_comma_segments.sh` | Scarica i giri nuovi dal comma nella cartella condivisa con `openpilot.distill` (`/root/comma-videos`). Se distill li ha già scaricati, salta questo passo. |
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
- **Pannello** (`/pannello`): lo stato di tutto e i pulsanti per lanciare i passi 1 e 3 senza terminale.

Il pannello mostra:

- il **prossimo passo**, con il pulsante giusto;
- i tre passi con i numeri (route nuove, frame revisionati, route revisionate su 5) e i pulsanti **Prepara le route nuove**, **Importa le revisioni**, **Allena da zero**, **Allena dai pesi stable**. Un pulsante spento ha sotto il motivo (per esempio "servono 5 route revisionate, ora 1" o "la GPU è occupata");
- GPU, disco, CPU e cosa sta girando, anche lanciato da terminale e anche il training di `openpilot.distill`;
- la tabella delle route con lo stato;
- le azioni lanciate dal pannello, con il loro stato (in corso, finita, fallita, interrotta), il pulsante **Log** (il log scorre dal vivo in basso nella pagina) e **Ferma**.

Il pannello lancia solo quelle quattro azioni fisse, mai un comando qualsiasi, e una sola per tipo alla volta (preparazione da una parte, import/training dall'altra), contando anche quelle partite da terminale. È protetto come il resto del sito: token sulla LAN, e i pulsanti accettano solo richieste dalla pagina stessa. I log delle azioni sono in `data/logs/panel/`; l'ultima riga di ognuno è il codice di uscita.

`NNSLR_PANEL_ACTIONS=0` (nell'ambiente del server, per il servizio in `/etc/nnslr-review.env`) lascia il pannello in sola lettura, per esempio se il sito lo usano altre persone per revisionare.

## 3. Il sito sempre acceso: il servizio `nnslr-review`

Per avere sito e pannello sempre accesi, anche dopo un riavvio del server, si installa il servizio systemd una volta sola (da root):

```sh
cd ~/sunnypilot.speedlimits
helpers/2_review.sh --stop                        # se avevi acceso la revisione a mano (porta 8765)
scripts/install_review_service.sh --dry-run       # mostra cosa farebbe, non cambia niente
scripts/install_review_service.sh                 # installa, avvia e stampa il link
```

Il servizio ascolta su `0.0.0.0:8765`, riparte da solo se cade e a ogni boot. Rilanciare l'installazione non fa danni: riscrive il servizio solo se `deploy/nnslr-review.service` è cambiato e non tocca il token.

### Il token per entrare dal browser

Sulla LAN ogni richiesta vuole il token di accesso. Il posto dove cercarlo è il file **`.env` del repository** (`~/sunnypilot.speedlimits/.env`), riga `NNSLR_REVIEW_TOKEN=...`:

```sh
grep NNSLR_REVIEW_TOKEN ~/sunnypilot.speedlimits/.env
```

- `.env` è in `.gitignore`: non finisce mai in git né su GitHub. Ha permessi 600 (lo legge solo root).
- Lo crea la prima volta chi serve per primo: `helpers/2_review.sh` o `scripts/install_review_service.sh`. L'installazione del servizio usa il token già presente in `.env`; se non c'è, ne crea uno e lo scrive in `.env`.
- Il servizio systemd lo legge da una sua copia, `/etc/nnslr-review.env` (permessi 600); ogni installazione rimette in `.env` lo stesso valore.
- Il **link già pronto** (`http://<IP del server>:8765/?token=...`) lo stampano l'installazione, `helpers/2_review.sh` e `helpers/update.sh`.
- Dopo la **prima apertura** del link il browser tiene il token in un cookie: da lì in poi basta `http://<IP del server>:8765/` (e `/pannello`).

Il link contiene il token: non condividerlo. Per cambiarlo: scrivi il nuovo valore (almeno 16 caratteri tra lettere, cifre, `-` e `_`) in `/etc/nnslr-review.env`, poi `scripts/install_review_service.sh` (lo copia in `.env`) e `systemctl restart nnslr-review`.

### Aggiornare il codice

```sh
cd ~/sunnypilot.speedlimits
helpers/update.sh
```

`helpers/update.sh` fa `git pull --ff-only` (si ferma se ci sono modifiche locali), aggiorna `.venv` se è cambiato `pyproject.toml`, rilancia `scripts/install_review_service.sh` (riscrive il servizio solo se è cambiato) e riavvia il servizio; senza servizio riavvia il sito acceso da `helpers/2_review.sh`. Alla fine stampa il link.

Il servizio ha `KillMode=process`: un riavvio ferma solo il sito, non una preparazione o un training partiti dal pannello. Dopo il riavvio il pannello li ritrova come "in corso".

### Controllare e fermare

```sh
systemctl status nnslr-review                     # acceso? da quanto?
journalctl -u nnslr-review -n 50                  # ultime righe del server (anche "Pannello: avviata ...")
systemctl stop nnslr-review                       # spegne (riparte al prossimo boot)
scripts/install_review_service.sh --uninstall     # toglie il servizio (il token resta)
```

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
| Copia del token letta dal servizio | `/etc/nnslr-review.env` |
| Definizione del servizio | `deploy/nnslr-review.service` → installato in `/etc/systemd/system/nnslr-review.service` |

`data/` e `.env` non sono in git.

## 6. Script del repository

| Script | Cosa fa | Effetti |
| --- | --- | --- |
| `helpers/status.sh` | stato di tutto e prossimo comando | solo lettura |
| `helpers/1_prepare.sh` | video → frame → cartelli proposti | scrive in `data/`, CPU |
| `helpers/2_review.sh` | accende/spegne il sito di revisione, stampa il link | apre la porta 8765 sulla LAN, crea il token in `.env` |
| `helpers/3_train.sh` | revisioni → dataset → modello | scrive il dataset, installa `.venv-gpu` la prima volta, GPU |
| `helpers/update.sh` | aggiorna il codice e riavvia il sito | `git pull`, `.venv`, riavvio del servizio |
| `helpers/fetch_comma_segments.sh` | scarica i giri nuovi dal comma | rete, cartella condivisa dei video |
| `scripts/install_review_service.sh` | installa/aggiorna/toglie il servizio `nnslr-review` | systemd, `/etc/nnslr-review.env`, `.env` |
| `scripts/setup_v100_training_env.sh` | crea `.venv-gpu` (lo lancia anche `helpers/3_train.sh`) | installa pacchetti |
| `scripts/run_v100_baseline.sh` | training completo (lo lancia `helpers/3_train.sh`) | GPU |
| `scripts/publish_model.sh` | pubblica un bundle su Hugging Face | upload pubblico |

## 5. Problemi comuni

| Messaggio | Cosa fare |
| --- | --- |
| `servono almeno 5 route (o luoghi) indipendenti revisionate` | Prepara e revisiona altre route, poi rilancia `helpers/3_train.sh`. |
| `la GPU 0 e' occupata` / pulsante "la GPU è occupata" | Sta girando un altro training (spesso `openpilot.distill`): aspetta che finisca. Da terminale `--force` parte lo stesso, più lenti tutti e due. |
| `la porta 8765 e' gia' occupata` | C'è già un sito acceso (il servizio o un altro `nnslr review`): `helpers/2_review.sh` ristampa il link, `helpers/2_review.sh --stop` spegne quello acceso a mano. |
| `Accesso negato: apri l'indirizzo con il token` | Apri il link con `?token=...` (sezione 3). |
| `su ... non ci sono ancora pesi stable` | `--continue` vuole un modello già pubblicato come stable: allena prima da zero. |
| Una route resta "da collegare" | Nella cartella condivisa manca `fcamera.hevc` per quei segmenti, oppure `link-videos` ha trovato un conflitto: vedi `data/logs/prepare/link-videos.json`. |
| Estrazione o pre-annotazione fallita | Il motivo è in `data/logs/prepare/<route>.log`. |
