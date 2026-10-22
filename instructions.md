# Revisione dei cartelli con Playwright

Procedura ricavata dalla revisione reale svolta il **21 ottobre 2026**. Serve al prossimo agente per revisionare immagini e salvare le decisioni, verificando anche il funzionamento del programma.

## 1. Definire l'operazione e rispettare il lavoro esistente

1. Leggere `AGENTS.md` e il contesto condiviso indicato lì.
2. Controllare directory, branch e modifiche con `pwd`, `git branch --show-current` e `git status --short`. Lavorare su `main` senza cambiare branch o cancellare modifiche altrui.
3. Distinguere **test senza salvataggio** da **revisione reale con salvataggio**. In questa sessione Cristian ha scelto esplicitamente la seconda opzione.
4. Usare un nome di revisore riconoscibile: qui è stato usato **Codex**, senza attribuire le decisioni a `cristianku`.
5. Non avviare o riconfigurare il server per questa attività. Se è irraggiungibile, segnalarlo. Non accendere host, aggiornare il comma, importare annotazioni nel dataset o avviare addestramenti.

## 2. Aprire un browser Playwright controllabile

In questa sessione non era disponibile uno strumento browser MCP. È stato usato Playwright Node già presente sul Mac, con Chromium grafico già installato.

Percorsi effettivamente usati, da **ricontrollare** perché la cache può cambiare:

```text
Modulo: /Users/cristianku/.npm/_npx/9833c18b2d85bc59/node_modules/playwright
Browser: /Users/cristianku/Library/Caches/ms-playwright/chromium-1243/chrome-mac-x64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing
```

Il primo modulo trovato richiedeva un browser `chromium-1248` assente. È stato scelto il browser già disponibile, senza installarne un altro. Il sandbox impediva l'avvio grafico su macOS; il comando è stato rieseguito tramite l'escalation prevista dallo strumento, per la sola apertura del browser richiesta dall'utente.

Per una sessione lunga occorre conservare `browser`, `context` e `page`. È stato scritto un piccolo runner Node temporaneo con `readline`, avviato con PTY (`tty: true`). Il runner eseguiva una riga alla volta in un contesto asincrono e restituiva un risultato JSON sintetico.

Schema minimo del runner; i percorsi si passano dall'ambiente dopo averli verificati:

```js
const { chromium } = require(process.env.NNSLR_PW_MODULE);
const readline = require('node:readline');

(async () => {
  globalThis.browser = await chromium.launch({
    headless: false,
    executablePath: process.env.NNSLR_PW_EXECUTABLE,
  });
  globalThis.context = await browser.newContext({
    viewport: { width: 1440, height: 1000 },
  });
  globalThis.page = await context.newPage();
  const lines = readline.createInterface({ input: process.stdin });
  for await (const line of lines) {
    try {
      const result = await eval(`(async () => { ${line} })()`);
      console.log(JSON.stringify({ result }));
    } catch (error) {
      console.log(JSON.stringify({ error: error.message }));
    }
  }
})();
```

Questo runner è solo uno strumento locale dell'agente: non va esposto come servizio. Nel PTY è stato disabilitato l'echo con `stty -echo`. **Inviare comandi brevi**: le righe lunghe sono state troncate dal terminale. Per operazioni articolate scrivere un file JavaScript temporaneo e inviare soltanto:

```js
return await eval(require('node:fs').readFileSync('/private/tmp/review-job.js', 'utf8'));
```

Il file deve contenere un'espressione asincrona, per esempio `(async () => { /* operazioni */ })()`. Non affidarsi a stdin di un processo avviato senza PTY: in questa sessione risultava chiuso.

## 3. Autenticarsi e aprire la route

Indirizzi usati nella sessione, non valori da assumere validi per tutte le revisioni:

```text
Server: http://10.10.10.213:8765/
Route:  http://10.10.10.213:8765/route/0000011e--0c8825628a/
```

1. Aprire nel contesto Playwright il link di accesso con token fornito da Cristian.
2. Aprire la route **nello stesso contesto**, mantenendo il cookie di autenticazione.
3. Verificare risposta HTTP, titolo e testo della pagina. Senza autenticazione era comparso `403 — Accesso negato`.
4. Non salvare il token nel repository, negli screenshot dell'indirizzo o negli output. Passarlo al browser a runtime; non copiare cookie nei documenti.

Raccogliere fin dall'apertura gli eventi `pageerror` e le risposte HTTP con stato almeno 400. Nei risultati stampare soltanto stato e `pathname`, evitando query string e credenziali.

## 4. Fotografare lo stato iniziale

Prima di salvare nuove annotazioni:

1. Registrare `route_id`, `run_id`, numero di fotogrammi e conferme già presenti.
2. Leggere tramite la sessione autenticata:

   ```text
   GET /api/route/<route_id>/reviews
   ```

3. Conservare una copia locale delle revisioni iniziali, inclusi `frame_key`, revisore, revisione e riquadri. Servirà per provare che il lavoro precedente è rimasto intatto.
4. Leggere il JSON nello script HTML `#payload`. Per la lavorazione bastano indice, `frame_key`, `preview_path`, proposte, seconda lettura e stato di conferma.
5. **Non stampare tutto il payload**: contiene metadati estesi per ogni fotogramma. Un tentativo in questa sessione ha prodotto un output enorme. Salvare i dati necessari in un file temporaneo e restituire soltanto conteggi o estratti mirati.

In questa sessione: 720 fotogrammi, 15 già confermati da `cristianku`, 95 sospetti iniziali, seconda lettura presente su 148 fotogrammi. Sono dati storici: rileggerli dal server prima di un'altra attività.

## 5. Comprendere interfaccia e formato corrente

Controllare la versione effettivamente servita, non soltanto il checkout locale. I riferimenti nel repository sono:

- `src/nnslr_tools/review.html`: navigazione, editor, filtri e salvataggio.
- `src/nnslr_tools/review.py`: autenticazione, immagini ed endpoint HTTP.
- `src/nnslr_tools/review_db.py`: validazione, revisioni concorrenti e storico.
- `tests/preannotation/test_review_browser.py` e `tests/review/`: test già esistenti, utili se viene richiesta una modifica del programma.

Nella pagina esaminata erano accessibili a `page.evaluate`:

```text
rows, S, currentRow(), workOf(row), openFrame(key), isConfirmed(key),
frameStatus(key), addManualBox(row, box, family, value), setField(key, value),
changed(row), render(), loadServer(), unsavedKeys()
```

Sono dettagli interni e possono cambiare. Verificarne l'esistenza e il comportamento prima di riutilizzarli. Il percorso normale rimane l'interfaccia con `+ Riquadro`, editor e conferma.

### Segnali stabili per gli agenti (dal 22 ottobre 2026)

La pagina espone il proprio stato con attributi che non cambiano il layout. Usarli al posto di colori, testi italiani e globali interne:

| Dove | Attributo | Significato |
| --- | --- | --- |
| `body` | `data-frame-key` | Fotogramma mostrato |
| `body` | `data-frame-status` | `todo`, `dirty`, `saving`, `error`, `done` |
| `body` | `data-server` | `loading` finché `GET /reviews` non risponde, poi `ready`; `error` o `stale` (report rigenerato); `offline` per il file aperto senza server |
| `body` | `data-unsaved` | Fotogrammi confermati non ancora salvati sul server (`0` = tutto salvato) |
| `body` | `data-last-save-key`, `data-last-save-result`, `data-last-save-revision`, `data-save-count` | Ultimo salvataggio: `saving`, poi `ok` o `error`; `unchanged` se la conferma non ha inviato nulla; `withdrawn` dopo un ritiro; `need-reviewer` se manca il revisore |
| `#stage` | `data-photo-key` | Chiave del fotogramma la cui foto è caricata (`error:<chiave>` se non si carica) |
| `#boxList .box-row` | `data-index`, `data-detection-id`, `data-duplicate-of` | Riga di un riquadro; `.box-main` lo seleziona, `.box-del` lo elimina |
| `#families`, `#familiesMore` | `[data-family]` | Tipo di cartello |
| `#quickValues` | `[data-value]` | Velocità rapide; `data-value="unknown"` = numero non leggibile |
| `#roads` | `[data-applicability]` | `own_road`, `other_road`, `unresolved` |
| `#hintList` | `[data-hint-index]` | Pulsante “Usa” di una voce della seconda lettura |
| `#confirm`, `#mConfirm` | `data-action="confirm"`, `data-save-state` | Usare `[data-action="confirm"]:visible`: sotto i 981 px è visibile solo quello della barra in basso |
| `#withdraw` | `data-armed` | “Annulla conferma” vuole due clic: il primo lo arma (`true`), il secondo ritira |

Anche `data-index` sui riquadri disegnati nella foto esiste: per la lista usare sempre `#boxList .box-row[...]`, meglio con `data-detection-id`, perché l'indice cambia dopo ogni eliminazione.

Campi e valori incontrati:

| Campo | Valori |
| --- | --- |
| `family` | `maximum_speed`, `cancellation`, `zone`, `variable_display`, `road_marking_candidate`, `other_sign`, `unreadable`, `not_a_sign` |
| `value_kph` | Numero leggibile oppure `null` |
| `applicability` | `own_road`, `other_road`, `unresolved` |
| `visibility` | `unknown`, `clear`, `partial`, `poor` |
| `occlusion` | `unknown`, `none`, `partial`, `heavy` |
| `supplementary_panel` | `unknown`, `none`, `present`, `unresolved` |
| `temporary`, `variable_display` | Valori booleani oppure stato sconosciuto previsto dall'interfaccia |

Leggere le opzioni dei controlli correnti prima di comporre annotazioni programmatiche. In questa sessione `visibility: distant` era un valore errato introdotto dall'agente: il server lo ha correttamente respinto con HTTP 400 `unknown_enum`. È stato sostituito con `poor` e la conferma è poi riuscita. Non è stato un difetto del programma.

I riquadri usano `bbox_xyxy = [x1, y1, x2, y2]` nelle coordinate **dell'immagine originale**, non della miniatura. Per questa route le immagini erano 1344 × 760.

## 6. Esaminare davvero tutte le immagini interessate

Per questa revisione sono state scaricate le anteprime mediante `context.request.get`, che riusa i cookie del browser. Le richieste erano in gruppi di **6**, verificando lo stato HTTP di ogni risposta. Non sono stati modificati i report o i filmati originali sul server.

Per orientarsi sono stati preparati fogli di contatto con Pillow:

- 12 fotogrammi per foglio, 2 colonne;
- ogni immagine ridotta a 672 × 380;
- etichetta con indice, `segmento/immagine`, stato iniziale e numero di proposte/aiuti;
- lettura dei fogli in ordine e controllo dei candidati mediante immagini originali e ritagli ingranditi.

Il foglio di contatto **non basta per leggere un numero piccolo**. Gli strumenti possono ridimensionare ulteriormente immagini molto alte. Aprire l'originale o un ritaglio con contesto quando servono dettagli; non confermare un'assenza se il ridimensionamento lascia un dubbio sostanziale.

Per gli ingrandimenti sono stati usati i riquadri proposti o della seconda lettura come punti di partenza, aggiungendo margine attorno. Controllare sempre che il riquadro racchiuda il cartello giusto e correggerlo se necessario. Non considerare le coordinate o le etichette automatiche una verità già verificata.

**Non automatizzare il giudizio:** un ciclo può scaricare immagini o applicare decisioni già prese, ma non deve confermare fotogrammi non esaminati né accettare in massa le proposte del modello.

## 7. Criteri di revisione applicati

- Assegnare una velocità soltanto quando il numero è leggibile nel fotogramma. Non copiarla dai fotogrammi vicini o dalla seconda lettura per colmare una mancanza.
- Se si riconosce un limite ma non il numero, usare il valore sconosciuto previsto dall'editor. Se non è determinabile neppure il contenuto del cartello, usare `unreadable` e una nota.
- Distinguere un limite attivo da una **fine limitazione**. Nel formato corrente `cancellation` non ha valore numerico: eventuale numero barrato leggibile può essere descritto nelle note.
- Distinguere `zone` da un normale limite massimo. Valutare la strada di applicazione: una zona 30 su un accesso laterale non diventa il limite della strada percorsa.
- Usare `unresolved` quando l'applicabilità non si può stabilire, per esempio guardando un ramo oltre una rotonda.
- Un pannello variabile vuoto non visualizza una velocità: valore sconosciuto e nota esplicita. Non inventare il messaggio di un display spento.
- Un segnale luminoso sovraesposto rimane illeggibile anche se le proposte suggeriscono un numero.
- Escludere dalle annotazioni di velocità loghi, pubblicità, precedenze, frecce, limiti di altezza e altri divieti non pertinenti. Le classi `other_sign` e `not_a_sign` sono disponibili se richieste dal protocollo di annotazione; non introdurre una diversa politica di dataset autonomamente.
- Rimuovere proposte duplicate sullo stesso cartello. Controllare anche i riquadri piccoli completamente contenuti in uno più grande: la sola IoU può non individuarli.
- Confermare senza riquadri i fotogrammi esaminati senza cartelli pertinenti: anche gli esempi negativi fanno parte della revisione.
- Preservare le conferme già presenti di altri revisori. Per correggerle serve un'autorizzazione che comprenda quella revisione.

Esempi effettivi della route: fine limite generale 50 proposta come 40; limiti 60 proposti come 40/80; loghi B&R proposti come limiti; limite 10 all'ingresso del parcheggio da distinguere dall'altezza massima 2,3 m accanto.

## 8. Salvare tramite il flusso del programma

Per ogni fotogramma già esaminato:

1. Aprirlo con la timeline/navigazione, oppure con `openFrame(frame_key)` dopo avere verificato il client corrente.
2. Attendere `#stage[data-photo-key="<frame_key>"]` (oppure `error:<frame_key>`). Fino ad allora i riquadri del nuovo fotogramma restano nascosti: non compaiono più sulla foto precedente. Poi controllare visivamente.
3. Preparare l'intera annotazione del fotogramma: conservare i riquadri validi, correggere etichette/coordinate, aggiungere cartelli mancanti e rimuovere falsi positivi e doppioni.
4. Se si usano gli helper interni, passare attraverso lo stato dell'editor (`workOf`, `addManualBox`, `setField`, `changed`, `render`), non modificare SQLite o i report di pre-annotazione. Non azzerare i riquadri prima di aver stabilito l'intera annotazione sostitutiva.
5. Premere `[data-action="confirm"]:visible` e attendere `body[data-last-save-key="<frame_key>"][data-last-save-result="ok"]` (oppure `error`), che corrisponde alla risposta **a quel frame**:

   ```text
   POST /api/route/<route_id>/frame/<segment_index>/<output_index>
   ```

   Su un fotogramma già confermato e non modificato il pulsante dice “Già confermato · avanti →”: non parte nessuna POST, `data-last-save-result` vale `unchanged` e la pagina passa al successivo.

6. Verificare HTTP 200, `review.confirmed === true`, revisore, revisione e riquadri restituiti. Il client gestisce `run_id` e `base_revision`.
7. Registrare localmente un audit sintetico con indice, `frame_key`, revisione e annotazioni salvate. Procedere in sequenza: l'interfaccia ha un solo fotogramma corrente.

La conferma avanza al fotogramma successivo secondo il filtro: dopo ogni conferma riaprire il prossimo per chiave e controllare `body[data-frame-key]`, invece di premere → o Invio. **Modificare un campo non equivale a salvarlo.** Un errore di rete o validazione lascia lavoro non salvato: correggere il motivo e verificare di nuovo. In caso di conflitto di revisione non forzare la sovrascrittura; rileggere lo stato del server e preservare il lavoro concorrente.

## 9. Verificare indipendentemente i salvataggi

1. Rileggere `GET /api/route/<route_id>/reviews` dopo l'ultimo salvataggio e conservare il risultato finale.
2. Confrontare le conferme preesistenti con lo snapshot iniziale: stessi revisori, revisioni e riquadri. In questa sessione tutte le 15 sono rimaste intatte.
3. Contare le revisioni **uniche per fotogramma**, non le richieste: una correzione successiva incrementa lo storico senza aggiungere un fotogramma.
4. Aprire un contesto Playwright nuovo con i soli cookie, senza il `localStorage` delle revisioni:

   ```js
   const verifyContext = await browser.newContext({
     storageState: { cookies: await context.cookies(), origins: [] },
   });
   const verifyPage = await verifyContext.newPage();
   ```

5. Aprire la route, attendere `body[data-server="ready"]` e verificare conteggi, `body[data-unsaved="0"]` e assenza di avvisi in `#notices`.
6. Controllare immagini e annotazioni di casi rappresentativi, aprire l'editor e lo zoom, attivare/disattivare la seconda lettura, provare avanti/indietro e il filtro `Da rivedere`. I contatori “Confermati con dubbi” (senza numero, con illeggibili, strada non indicata) sono filtri: usarli per ricontrollare i casi incerti.
7. Non chiamare riuscita una route soltanto perché compare `0 sospetti`: è un controllo automatico, non una prova autonoma della qualità delle etichette. Riferire anche i casi illeggibili o di applicabilità irrisolta.

## 10. Risultato storico e passaggio al prossimo agente

Esito verificato il 21 ottobre 2026 per `0000011e--0c8825628a`, run `20261021T083422Z-2468e469`:

| Verifica | Risultato |
| --- | --- |
| Confermati sul server e nel browser nuovo | 720 / 720 |
| Nuovi fotogrammi revisionati da Codex | 705 |
| Conferme preesistenti preservate | 15 / 15 |
| Fotogrammi Codex con riquadri / senza riquadri | 124 / 581 |
| Riquadri Codex dopo rimozione di un doppione | 150 |
| Famiglie dei riquadri Codex | 78 limiti massimi, 31 fine limitazione, 9 zone, 11 display, 21 illeggibili |
| Stato finale del programma | 0 sospetti, 0 modifiche non salvate, nessun avviso server |
| Errori JavaScript rilevati durante la revisione | 0 |
| Navigazione finale provata | `13/47` → `13/48` → `13/47` |

Sono state salvate anche etichette con valore sconosciuto: il completamento delle conferme non significa che ogni cartello fosse leggibile. La validazione ha respinto un input errato dell'agente, poi corretto; non va presentato come un bug del servizio. I controlli svolti sono uso reale del programma e verifica HTTP/UI, non una dichiarazione che tutta la suite automatica sia stata eseguita.

Artefatti locali della sessione, **temporanei e non garantiti per il futuro**:

```text
/private/tmp/cartelli-0000011e-review/
  metadata.json
  initial-reviews.json
  codex-saved.jsonl
  final-server-reviews.json
  verification.json
  verified-10-parking.png
  <indici>.jpg, sheet-*.jpg, crops-*.jpg

/private/tmp/route-review-playwright.cjs
/private/tmp/cartelli-sheets.py
/private/tmp/cartelli-crops.py
```

`initial-reviews.json` contieneva lo stato del client con le revisioni iniziali e le rispettive basi server; il file finale contiene la risposta API. Per una nuova sessione preferire anche uno snapshot iniziale diretto della risposta API. Il repository non contiene token o copie delle immagini: la fonte delle conferme resta il server.

Al termine riferire route, fotogrammi effettivamente controllati/salvati, conferme altrui preservate, incertezze e verifiche svolte. Fermarsi alla revisione: importazione nel dataset, modifiche del servizio, commit/push e deployment sono operazioni separate da eseguire soltanto entro l'autorizzazione ricevuta.
