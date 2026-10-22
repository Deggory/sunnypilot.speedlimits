# Istruzioni per l'agente che revisiona i cartelli

Queste istruzioni servono a un agente che apre il sito di revisione, guarda i fotogrammi delle route e conferma i cartelli di velocità. Valgono sia per un agente che lavora dentro Chrome con l'estensione di Claude o di ChatGPT, sia per un agente che pilota il browser con Playwright. Descrivono il sito nella versione del **22 ottobre 2026**.

L'indirizzo del sito lo indica Cristian, oppure il sito è già aperto in Chrome. Non cercarlo da solo e non scriverlo nei resoconti.

## 1. Regole prima di iniziare

1. Leggere anche `AGENTS.md`.
2. Capire quale delle due attività è stata chiesta:
   - **prova senza salvataggio**: si guarda e si modifica, ma non si preme **Conferma**;
   - **revisione con salvataggio**: si conferma. Solo se Cristian l'ha chiesto esplicitamente per quella route.
3. Scrivere nel campo **Revisore** (in alto a destra) un nome riconoscibile dell'agente, per esempio `Claude` o `ChatGPT`. Mai `cristianku` o altri nomi di persone.
4. Non modificare né ritirare conferme già fatte da altri revisori, salvo richiesta esplicita che le comprenda.
5. Nel **Pannello** non premere nessun pulsante (Prepara, Importa, Allena, Aggiorna il codice): hanno effetti reali sul server. Non riavviare il server, non toccare il comma, non importare nel dataset, non avviare training.
6. Il link di accesso contiene un token. Non copiarlo in file, resoconti, screenshot o log.
7. Non confermare mai un fotogramma senza averlo guardato davvero. Non accettare in blocco le proposte del modello.

## 2. Accesso

1. Aprire l'indirizzo ricevuto. Se compare **Accesso negato**, serve il link con il token: chiederlo a Cristian. Dopo la prima apertura il browser tiene un cookie e basta l'indirizzo senza token.
2. La pagina iniziale **Route** elenca le route con un report. Ogni card mostra "N di M confermati", i sospetti e un'azione: **Inizia →**, **Riprendi →** o **Ricontrolla →**. Aprire la route indicata da Cristian.
3. Lavorare con la finestra larga almeno 1000 px. Sotto i 981 px la pagina passa al layout da telefono: il pulsante Conferma diventa la barra fissa in basso.
4. Se in fondo al pannello compare **"Revisioni non lette dal server"** o **"Il report di questa route è stato rigenerato: ricarica la pagina"**, fermarsi: ricaricare la pagina e, se resta, avvisare Cristian.

## 3. Com'è fatta la pagina di revisione

**Barra in alto**
- Il menu della route.
- I filtri **Da rivedere / Sospetti / Con proposte / Tutti**:
  - **Da rivedere**: i fotogrammi non ancora confermati;
  - **Sospetti**: modello e seconda lettura non sono d'accordo;
  - **Con proposte**: il modello ha proposto almeno un riquadro.
- Il contatore "N di M confermati · K sospetti".
- Il campo **Revisore**.

**A sinistra**
- **La foto**, con i riquadri:
  - etichette gialle come `1 · 40`, con il numero del riquadro e il valore;
  - riquadri azzurri tratteggiati con etichette come `60?`: sono la seconda lettura, solo un aiuto.
- **La barra sotto la foto**:
  - le frecce **‹ ›** con la posizione nel filtro, per esempio `62 / 68`;
  - lo stato del fotogramma: Da rivedere, Da confermare, Da riconfermare, Salvataggio…, Confermato, Non salvato;
  - "Segmento X · immagine Y · N proposte";
  - **+ Riquadro**, **Elimina** e l'interruttore **Seconda lettura**.
- **La timeline**: una riga di tacche per segmento ("seg 2", "seg 3"…). Cliccando una tacca si apre quel fotogramma. I colori:
  - verde: confermato;
  - verde con la parte alta gialla: confermato con dubbi;
  - viola: sospetto;
  - giallo: con proposte o seconda lettura;
  - arancione: modificato, da confermare;
  - rosso: non salvato;
  - grigio: niente proposto.
- **Confermati con dubbi:** tre contatori cliccabili, **senza numero**, **con illeggibili** e **strada non indicata**. Ciascuno filtra i fotogrammi confermati che hanno quel dubbio; un secondo clic torna a "Da rivedere".

**A destra, il pannello**
- **Cartelli**: un'eventuale riga viola **Sospetto: …** con il motivo, poi un riquadro per riga:
  - numero e descrizione, per esempio "Limite massimo · 40 km/h";
  - sotto, l'origine: "modello · 92%", "modello 40 → 60" (cioè hai cambiato il valore proposto) oppure "aggiunto a mano";
  - **"stesso cartello di N"** in arancione: è un doppione di un altro riquadro;
  - **✕**: elimina quel riquadro.
- **Riquadro N**, quando un riquadro è selezionato:
  - lo zoom sul cartello, con gli angoli gialli per ridimensionare;
  - i cartelli rapidi **20 30 40 50 60 70 80 100 120** e **?** (numero non leggibile);
  - **Altro valore** per i numeri che non ci sono tra quelli rapidi;
  - **Tipo**: Limite massimo, Fine limitazione, Illeggibile, Segnale variabile, Zona; più in piccolo Numero a terra, Altro cartello, Non è un cartello;
  - **Strada**: La nostra, Un'altra, Da chiarire;
  - **Dettagli del cartello**, chiuso: visibilità, occlusione, pannello sotto, temporaneo, display variabile, note.
- **Seconda lettura**: c'è solo sui fotogrammi che ne hanno una. Ogni voce indica tipo, valore, sicurezza e lato, con il pulsante **Usa**, che diventa **Già usato** quando quel cartello c'è già.
- **Altro**, chiuso: sito fisico, incontro, esportazione JSONL, scorciatoie.
- **In fondo**, sempre nello stesso punto:
  - la riga di stato;
  - i link **Annulla modifiche** e **Annulla conferma**;
  - il pulsante **Conferma**, che dice cosa salva: "Conferma · 60 · 40", "Conferma · nessun cartello", oppure "Già confermato · avanti →" su un fotogramma già confermato e non modificato.

La foto e il pulsante Conferma non si spostano mai da soli: avvisi ed errori compaiono nel pannello, sopra la riga di stato.

## 4. Come guardare le immagini

- Per leggere un cartello selezionarne il riquadro e usare lo zoom nel pannello. La foto intera serve per il contesto: strada, corsie, rami laterali.
- Lo zoom e uno screenshot ridotto possono non bastare per un numero piccolo. Se resta un dubbio sostanziale, non inventare: usare **?** o **Illeggibile**.
- Le proposte del modello e la seconda lettura sono punti di partenza, non verità. Controllare sempre che il riquadro racchiuda il cartello giusto.
- Prima di confermare un fotogramma senza cartelli, controllare tutta la foto: bordi, cartelli lontani e piccoli, portali sopra la strada, numeri dipinti sull'asfalto.

## 5. Criteri di revisione

- Dare un valore solo quando il numero si legge in quel fotogramma. Non copiarlo dai fotogrammi vicini né dalla seconda lettura.
- Se si riconosce un limite ma non il numero: tipo **Limite massimo** e cartello **?**. Se non si capisce neppure che cartello è: **Illeggibile**, con una nota nei dettagli.
- Distinguere un limite attivo da una **Fine limitazione**. Nel formato attuale la fine limitazione non ha un numero: un numero barrato leggibile va nelle note.
- Distinguere **Zona** da un normale limite massimo.
- **Strada**:
  - **La nostra**: il limite vale per la strada che si sta percorrendo;
  - **Un'altra**: per esempio una zona 30 su un accesso laterale;
  - **Da chiarire**: quando non si può stabilire, per esempio un ramo oltre una rotonda.
- Un pannello variabile spento o vuoto non mostra una velocità: tipo **Segnale variabile**, cartello **?** e una nota. Un segnale luminoso sovraesposto resta illeggibile anche se la proposta suggerisce un numero.
- Non sono cartelli di velocità e vanno eliminati con **✕**: loghi, pubblicità, precedenze, frecce, limiti di altezza e altri divieti. **Altro cartello** e **Non è un cartello** si usano solo se Cristian lo chiede: tengono il riquadro come esempio.
- I doppioni sullo stesso cartello ("stesso cartello di N", per esempio 40 e 80 sullo stesso disco) si eliminano con **✕**. Si tiene un solo riquadro, con il valore giusto.
- Controllare anche i riquadri piccoli contenuti in uno più grande.
- Un fotogramma senza cartelli pertinenti si conferma comunque, con **Conferma · nessun cartello**: anche gli esempi negativi servono.

Esempi visti nelle route reali:
- una fine limite generale 50 proposta come 40;
- limiti 60 proposti come 40/80;
- loghi aziendali proposti come limiti;
- un limite 10 all'ingresso di un parcheggio, accanto a un divieto di altezza 2,3 m.

## 6. Procedura per ogni fotogramma

1. **Aprire il fotogramma**: con le frecce **‹ ›**, con una tacca della timeline o, per un agente che usa JavaScript, per chiave (sezione 9). Il filtro **Da rivedere** mostra solo ciò che resta da fare.
2. **Aspettare la foto**. I riquadri compaiono solo quando la foto nuova è caricata. Controllare che "Segmento X · immagine Y" sia il fotogramma voluto.
3. **Correggere i riquadri esistenti.** Cliccare la riga nella lista (o il riquadro sulla foto), poi:
   - scegliere il valore con un cartello rapido, con **?** o con **Altro valore**;
   - scegliere il **Tipo** e la **Strada**;
   - se il riquadro non racchiude bene il cartello, trascinare gli angoli gialli, meglio nello zoom;
   - eliminare con **✕** i falsi positivi e i doppioni.
4. **Aggiungere i cartelli mancanti**:
   - premere **+ Riquadro** (la foto mostra "Trascina sull'immagine per disegnare il riquadro");
   - trascinare sulla foto da un angolo all'opposto del cartello;
   - il riquadro nuovo nasce come **Illeggibile** e resta selezionato: assegnargli tipo, valore e strada.

   In alternativa, se la seconda lettura ha visto il cartello, premere **Usa** sulla sua voce e poi controllarlo.
5. **Confermare**:
   - verificare che il campo **Revisore** sia compilato;
   - premere **Conferma · …**.

   Il risultato:
   - **Riuscita**: la pagina passa da sola al fotogramma successivo del filtro. In fondo compare in verde "Salvato `<segmento/immagine>` alle HH:MM. Ecco il prossimo.", il contatore in alto sale e la tacca diventa verde.
   - **Fallita**: si resta sul fotogramma, lo stato diventa **Non salvato** e la riga di stato dice in rosso "NON salvato: motivo". Vedere la sezione 7.
   - **Già confermato · avanti →**: il fotogramma era già confermato e non è stato modificato. Non si salva niente, si passa al successivo.
6. **Annotarsi** l'esito: chiave del fotogramma, riquadri salvati e dubbi. Servono per il resoconto (sezione 8).

Non usare la tastiera per navigare o confermare. **Invio** conferma il fotogramma, anche subito dopo un clic su una freccia. Usare sempre i pulsanti.

## 7. Errori e casi particolari

- **"Scrivi nome o sigla del revisore, poi conferma."**: il campo Revisore è vuoto. Compilarlo e premere di nuovo **Conferma**.
- **"NON salvato: …"**: il motivo è scritto nella riga.
  - "Valore non ammesso…" o un valore fuori 5–250 km/h: correggere il riquadro e riconfermare.
  - "salvato da un'altra pagina nel frattempo": qualcun altro ha cambiato quel fotogramma. Ricaricare la pagina, guardare la versione attuale e non sovrascriverla senza autorizzazione.
  - "server non raggiungibile": aspettare e premere **Riprova a salvare** nel riquadro rosso. Se continua, fermarsi e avvisare Cristian.
- **Riquadro rosso "N fotogrammi confermati NON salvati sul server"**: ci sono conferme rimaste nel browser. Premere **Riprova a salvare**. Il lavoro è finito solo quando il riquadro sparisce.
- **Annulla modifiche** riporta il fotogramma all'ultima versione confermata, o alle proposte se non era mai stato confermato.
- **Annulla conferma** va premuto due volte: il primo clic lo trasforma in **Sì, annulla**, il secondo toglie la conferma. Va usato solo se richiesto.
- **Modificare non è salvare.** Un fotogramma modificato e non confermato resta arancione "Da confermare". Prima di finire non ne devono restare.
- Dopo l'ultimo fotogramma del filtro compare "Non restano fotogrammi da rivedere con questo filtro."

## 8. Controllo finale e resoconto

1. Ricaricare la pagina. Controllare il contatore "N di M confermati" e che non ci siano riquadri rossi né tacche rosse o arancioni.
2. Aprire il filtro **Da rivedere**: deve restare vuoto, o contenere solo quello che si è deciso di lasciare.
3. Guardare i contatori **Confermati con dubbi** e ricontrollare almeno alcuni di quei fotogrammi.
4. Per gli agenti che possono fare richieste HTTP: `GET /api/route/<route_id>/reviews`, con la stessa sessione del browser, restituisce `confirmed_count` e la lista `reviews` (`frame_key`, `confirmed`, `reviewer`, `revision`, `detections`). Confrontarla con lo stato iniziale: le conferme di altri revisori devono essere rimaste uguali.
5. Riferire a Cristian:
   - route;
   - fotogrammi confermati da te, con e senza cartelli;
   - quanti riquadri per tipo;
   - quanti con numero sconosciuto, illeggibili e con strada da chiarire;
   - errori incontrati e come sono stati risolti;
   - fotogrammi lasciati in sospeso e perché.

   Lo "0 sospetti" non dimostra che le etichette siano giuste: dirlo esplicitamente.

## 9. Per agenti che usano JavaScript o Playwright

La pagina espone il proprio stato con attributi che non cambiano il layout. Conviene usarli al posto di colori, testi e coordinate:

| Dove | Attributo | Significato |
| --- | --- | --- |
| `body` | `data-frame-key` | Fotogramma mostrato (`segmento/immagine`) |
| `body` | `data-frame-status` | `todo`, `dirty`, `saving`, `error`, `done` |
| `body` | `data-server` | `loading` finché le revisioni non arrivano dal server, poi `ready`; `error` o `stale` (report rigenerato); `offline` per la pagina aperta senza server |
| `body` | `data-unsaved` | Fotogrammi confermati non ancora salvati (`0` = tutto salvato) |
| `body` | `data-last-save-key`, `data-last-save-result`, `data-last-save-revision`, `data-save-count` | Ultimo salvataggio. Il risultato vale `saving`, poi `ok` o `error`; `unchanged` se la conferma non ha inviato niente; `withdrawn` dopo un ritiro; `need-reviewer` se manca il revisore |
| `#stage` | `data-photo-key` | Chiave del fotogramma la cui foto è caricata (`error:<chiave>` se la foto non si carica) |
| `#boxList .box-row` | `data-index`, `data-detection-id`, `data-duplicate-of` | Riga di un riquadro. `.box-main` lo seleziona, `.box-del` lo elimina |
| `#families`, `#familiesMore` | `[data-family]` | Tipo di cartello |
| `#quickValues` | `[data-value]` | Velocità rapide; `data-value="unknown"` = numero non leggibile |
| `#roads` | `[data-applicability]` | `own_road`, `other_road`, `unresolved` |
| `#hintList` | `[data-hint-index]` | Pulsante **Usa** di una voce della seconda lettura |
| `#timeline` | `button[data-key]` | Tacca di un fotogramma |
| `#confirm`, `#mConfirm` | `data-action="confirm"`, `data-save-state` | Usare `[data-action="confirm"]:visible`: sotto i 981 px è visibile solo quello della barra in basso |
| `#withdraw` | `data-armed` | Annulla conferma: `true` dopo il primo clic |

Note:
- Anche i riquadri disegnati sulla foto hanno `data-index`. Per la lista usare sempre `#boxList .box-row[...]`, meglio con `data-detection-id`, perché l'indice cambia dopo ogni eliminazione.
- I campi di **Dettagli del cartello** e di **Altro** sono dentro sezioni chiuse: aprirle (clic sul titolo) prima di usarli.

Schema per un fotogramma, con Playwright:

```js
const key = '3/28';
await page.locator('#timeline button[data-key="' + key + '"]').click();
await page.waitForSelector('body[data-frame-key="' + key + '"]');
await page.waitForSelector('#stage[data-photo-key="' + key + '"], #stage[data-photo-key="error:' + key + '"]');
// correzioni: per esempio il primo riquadro diventa 60, sulla nostra strada
await page.locator('#boxList .box-row').first().locator('.box-main').click();
await page.locator('#quickValues [data-value="60"]').click();
await page.locator('#roads [data-applicability="own_road"]').click();
// conferma e attesa del risultato di QUEL fotogramma
await page.locator('[data-action="confirm"]:visible').click();
await page.waitForSelector('body[data-last-save-key="' + key + '"]:not([data-last-save-result="saving"])');
const result = await page.evaluate(() => document.body.dataset.lastSaveResult);   // 'ok', 'error', 'unchanged', 'need-reviewer'
```

Più indicazioni:
- **Attese.** Dopo il caricamento della route aspettare `body[data-server="ready"]` prima di leggere conteggi o modificare. Prima del controllo finale aspettare `body[data-unsaved="0"]`.
- **Un riquadro nuovo.** Premere `#draw`, poi trascinare con il mouse sulla foto. Le coordinate dei riquadri (`bbox_xyxy = [x1, y1, x2, y2]`) sono quelle dell'immagine originale, non dello schermo. Per convertire un punto dell'immagine in coordinate dello schermo:

  ```js
  const [sx, sy] = await page.evaluate(([x, y]) => {
    const svg = document.getElementById('scene');
    const p = svg.createSVGPoint(); p.x = x; p.y = y;
    const q = p.matrixTransform(svg.getScreenCTM());
    return [q.x, q.y];
  }, [x, y]);
  ```
- **Navigazione.** Dopo ogni conferma riuscita la pagina passa da sola al fotogramma successivo. Aprire sempre il prossimo per chiave e controllare `body[data-frame-key]`, invece di premere ‹ › o Invio.
- **Funzioni interne.** Esistono anche funzioni come `openFrame(key)`, `rows`, `S`, `workOf(row)`, `addManualBox(row, box, family, value)` e `render()`, ma sono dettagli interni che possono cambiare. Se una modifica passa da `workOf(...)`, chiamare poi `render()`. Non modificare mai SQLite o i report sul server.
- **Valori ammessi dal server.** Se non sono rispettati, il server risponde con un errore 400:

  | Campo | Valori |
  | --- | --- |
  | `family` | `maximum_speed`, `cancellation`, `zone`, `variable_display`, `road_marking_candidate`, `other_sign`, `unreadable`, `not_a_sign` |
  | `value_kph` | intero tra 5 e 250, oppure `null` |
  | `applicability` | `own_road`, `other_road`, `unresolved` |
  | `visibility` | `unknown`, `clear`, `partial`, `poor` |
  | `occlusion` | `unknown`, `none`, `partial`, `heavy` |
  | `supplementary_panel` | `unknown`, `none`, `present`, `unresolved` |

  Al massimo 100 riquadri per fotogramma; il nome del revisore al massimo 100 caratteri.
- **Stato iniziale.** Il JSON dei fotogrammi è nello script `#payload` della pagina ed è grande: non stamparlo tutto, estrarre solo chiavi, `preview_path` e numero di proposte. Per una sessione lunga conservare lo stesso `context` (cookie) e registrare gli eventi `pageerror` e le risposte HTTP con stato ≥ 400, stampando solo stato e percorso, senza query string né token.
