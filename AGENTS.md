# Istruzioni per gli agenti

- Prima di agire, leggere e applicare anche `/Users/cristianku/Gitlab/codex-contexts/AGENTS.md`. Se non è leggibile, segnalarlo senza inventarne il contenuto.
- Per revisionare i cartelli tramite l'interfaccia web, leggere [review_instructions.md](review_instructions.md): descrive accesso con Playwright, controllo delle immagini, annotazioni, salvataggio e verifica indipendente sul server.
- Distinguere una prova del programma da una revisione reale. Salvare conferme nel database soltanto quando Cristian ha autorizzato la revisione con salvataggio. Non confermare immagini non esaminate.
- Lavorare su `main`, preservando le modifiche esistenti. Non creare branch o worktree senza richiesta esplicita. Commit e push richiedono autorizzazione.
- La revisione autorizza le annotazioni richieste, non modifiche al codice del servizio, installazioni, riavvii, importazioni nel dataset, addestramenti o deployment.
- Gli aggiornamenti e le installazioni sul comma li esegue Cristian: non modificare o riavviare il dispositivo senza richiesta esplicita per quella specifica operazione.
- Non accendere mai `serverone-ai`, nemmeno indirettamente. Non usare Bonsai o worker RX6800; non contattare il worker V100 finché Cristian non ne comunica nuovamente la disponibilità.
- Non inserire token, cookie o credenziali nei file versionati, nei resoconti o nei log condivisi.

## Helper scripts e punti di ingresso

- Prima di usare un comando, consultare [README — Helpers, commands and scripts](README.md#helpers-commands-and-scripts), [docs/INSTALL.md](docs/INSTALL.md) e, per dataset/training, [docs/upstream-training-guide.md](docs/upstream-training-guide.md). Verificare gli argomenti con `nnslr <subcommand> --help` quando pertinente.
- La procedura d'uso quotidiana (helper, pannello, servizio `speedlimits-web`, token, aggiornamenti) è in [docs/GUIDA.md](docs/GUIDA.md). Ogni operazione ricorrente deve essere uno script del repository documentato lì: non lasciare passaggi fatti a mano senza traccia.
- Gli helper semplici `helpers/1_prepare.sh` (scrive nel data root, usa molta CPU), `helpers/2_review.sh` (apre la porta 8765 sulla LAN, crea il token in `.env`), `helpers/3_train.sh` (importa nel dataset, installa `.venv-gpu`, usa la GPU) e `helpers/update.sh` (`git pull`, aggiorna `.venv`, riavvia il servizio) hanno effetti reali come gli script sotto: non lanciarli senza richiesta. Il token in `.env` non va mai copiato in file versionati, resoconti o log. `helpers/status.sh` è solo lettura. Lo stesso vale per i pulsanti della pagina `/pannello` del sito di revisione: una revisione autorizzata non autorizza a premerli.
- `nnslr` è la CLI di questo progetto. Gli script `scripts/setup_v100_training_env.sh`, `scripts/run_v100_baseline.sh`, `scripts/publish_model.sh` e `scripts/install_review_service.sh` hanno effetti diversi (installazione dipendenze, consumo GPU, upload pubblico, modifica systemd): non eseguirli solo perché si sta consultando la documentazione. Preferire `--dry-run`/`--no-upload` quando disponibili e appropriati.
- Non confondere i progetti: `openpilot.distill` contiene gli helper video del modello di guida; la skill `OP-update-distilled` in `cristianku/openpilot_scripts` aggiorna le branch Distill di Sunnypilot/openpilot e **non** distribuisce i modelli di speed limit di questo repository.
- Per ogni affermazione su una feature installabile su comma, distinguere la ricerca/offline dal runtime effettivo; verificare prima che esistano backend, integrazione Speed Limit Assist e branch installabile. Non annunciare come funzionante lo snapshot advisory recuperato.
