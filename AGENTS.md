# Istruzioni per gli agenti

- Prima di agire, leggere e applicare anche `/Users/cristianku/Gitlab/codex-contexts/AGENTS.md`. Se non è leggibile, segnalarlo senza inventarne il contenuto.
- Per revisionare i cartelli tramite l'interfaccia web, leggere [instructions.md](instructions.md): descrive accesso con Playwright, controllo delle immagini, annotazioni, salvataggio e verifica indipendente sul server.
- Distinguere una prova del programma da una revisione reale. Salvare conferme nel database soltanto quando Cristian ha autorizzato la revisione con salvataggio. Non confermare immagini non esaminate.
- Lavorare su `main`, preservando le modifiche esistenti. Non creare branch o worktree senza richiesta esplicita. Commit e push richiedono autorizzazione.
- La revisione autorizza le annotazioni richieste, non modifiche al codice del servizio, installazioni, riavvii, importazioni nel dataset, addestramenti o deployment.
- Gli aggiornamenti e le installazioni sul comma li esegue Cristian: non modificare o riavviare il dispositivo senza richiesta esplicita per quella specifica operazione.
- Non accendere mai `serverone-ai`, nemmeno indirettamente. Non usare Bonsai o worker RX6800; non contattare il worker V100 finché Cristian non ne comunica nuovamente la disponibilità.
- Non inserire token, cookie o credenziali nei file versionati, nei resoconti o nei log condivisi.
