# LLigMa Tutor — assistente didattico per istituti agrari

Un assistente italiano che risponde a domande di **agronomia**, **biologia** e
**cultura generale italiana**, pensato per studenti di un istituto tecnico agrario.
Gira su un portatile, risponde in millisecondi nella modalità di riferimento e
non inventa: se non sa, lo dice.

---

## 1. La decisione che tiene tutto in piedi

La richiesta era un modello piccolo, veloce, con le tecniche più recenti per
ridurre i parametri. La tentazione è addestrare da zero un LLigMa da 50-100M su
dispense scolastiche. Non funziona: sotto il miliardo di parametri un modello
addestrato da zero non impara una lingua, e comunque non memorizza in modo
affidabile un curriculum.

La scelta fatta è l'opposto, ed è quella che rende il sistema davvero piccolo:

> **I fatti non stanno nei pesi, stanno in un indice.**

Un modello da 0,3-0,6B non può contenere la rotazione colturale, la meiosi e la
Costituzione. Ma può scrivere bene una risposta a partire da cinque passaggi
corretti. Spostando la conoscenza fuori dal modello:

| | Modello da zero | LLigMa Tutor |
|---|---|---|
| Conoscenza | nei pesi (imprecisa) | nell'indice (verificabile) |
| Aggiornamento | riaddestrare | ricostruire l'indice (1 secondo) |
| Costo per risposta | GPU | CPU, ~0,5 ms di retrieval |
| Allucinazioni | possibili | strutturalmente limitate: la risposta cita una fonte |
| Dimensione minima utile | miliardi di parametri | ~300M, e funziona anche senza |

Il caso limite più utile da capire: **il sistema risponde già bene senza alcun
modello**. Il backend estrattivo di riferimento compone la risposta incollando le
frasi pertinenti delle fonti, con citazioni. Il modello distillato serve a
trasformare quel collage in prosa scorrevole, non a fornire i fatti. Se un giorno
il modello peggiora il testo invece di migliorarlo, il confronto è immediato,
perché il riferimento è lì (`--backend extractive`).

---

## 2. Architettura

```
domanda
   │
   ├─ normalizzazione + stemming leggero (it)
   ├─ espansione sinonimi (opzionale, solo query di retrieval)
   ├─ rilevamento materia (lessico, conservativo)
   │
   ▼
[ BM25 su passaggi ] ──► guardia di astensione (soglia + termini minimi)
   │                          │
   │                          └─► "non lo so" (nessuna generazione)
   ▼
[ prompt groundato ]  sistema + regole + contesto numerato + domanda
   │
   ▼
[ backend ]
   ├─ estrattivo   nessun modello, ~0,5 ms      ← riferimento
   ├─ locale       studente QLoRA 4-bit          ← produzione
   └─ API          docente (distillazione)       ← non gira a scuola
   │
   ▼
risposta + fonti citate + latenza
```

Componenti, tutti in `lligma/tutor/`:

| File | Responsabilità |
|---|---|
| `text.py` | normalizzazione, stemming italiano, frasi, passaggi |
| `expansion.py` | dizionario di sinonimi applicato alla query di retrieval |
| `index.py` | indice invertito BM25, persistenza JSON |
| `prompt.py` | contratto di risposta, prompt, rilevamento materia |
| `backends.py` | estrattivo / API / modello locale |
| `service.py` | orchestrazione, astensione, cache semantica |
| `server.py` | HTTP + SSE, solo libreria standard |

---

## 3. Tecniche usate, e perché

Ognuna è nel codice, non solo nel documento.

**Retrieval-augmented generation invece di parametri.** La conoscenza è
esternalizzata: un indice BM25 da 108 KB copre le 34 schede. È la singola scelta
che riduce di più i parametri necessari.

**BM25 invece di embedding densi.** Su un corpus monolingua di poche migliaia di
passaggi, BM25 è accurato quanto un encoder da 100 MB, con zero GPU e zero
dipendenze. "Piccolo e veloce" vale anche per il retriever. Se un giorno il
corpus supera ~50.000 passaggi, il posto dove aggiungere un ibrido denso è
`BM25Index.search`.

**Stemming italiano minimo ma mirato.** Non è Snowball: toglie un suffisso e la
vocale finale, così `pota`, `potare` e `potatura` cadono sulla stessa chiave —
uno studente scrive il verbo, la dispensa usa il nome. I suffissi sono divisi in
due classi. Quelli **derivativi** (`-azione`, `-atura`, `-mente`) si tolgono solo
se resta una radice di almeno tre lettere; quelli **flessionali** (`-are`,
`-ano`, `-ono`, `-ando`, `-avano`) ammettono anche due, perché `arare` e
`aratura` devono cercare la stessa scheda. La distinzione non è teorica: prima
`arare` diventava `arar` e `aratura` diventava `ara`, e la domanda «a cosa serve
arare il campo prima della semina?» recuperava la scheda sull'irrigazione
invece di quella sulle lavorazioni. I suffissi che accorciavano troppo
(`-zione`, `-ale`) sono stati tolti: facevano combaciare `funzione` con `fun` e
`integrale` con `integr`, cioè parole non correlate.

I termini ridotti sono salvati **dentro** l'indice, quindi un cambio di stemming
richiede la ricostruzione. `index/tutor-index.json` è ora in formato **v2**: un
indice v1 viene rifiutato con un messaggio esplicito invece di essere usato con
chiavi diverse dalle nuove. La costruzione è anche riproducibile: con
`SOURCE_DATE_EPOCH` fissato due build producono file byte per byte identici;
senza quella variabile cambia solo `built_at`, l'unico campo che dipende
dall'orologio. Misurato dopo il cambio, sui due set di valutazione
spediti: `eval.jsonl` 30/30 recall@3, 30/30 top@1, copertura 1,00, astensione
4/4; `eval-paraphrase.jsonl` 0/39 senza espansione e 39/39 con espansione. Sono
gli stessi numeri di prima. Il vocabolario dell'indice scende da 1.720 a 1.656
termini, perché forme della stessa famiglia ora si fondono.

**Distillazione docente → studente.** `scripts/distill_tutor.py` usa il modello
grande via API per generare coppie domanda-risposta **ancorate a un passaggio
reale**: il docente vede solo quel passaggio e le risposte che non vi trovano
appoggio vengono scartate dal filtro. Il dataset cresce senza inventare.

**QLoRA in 4 bit (NF4, double quant).** `scripts/finetune_tutor.py`: il modello
base resta quantizzato, si addestra un adapter da ~1% dei pesi. Un 0,6B diventa
~0,4 GB: sta su una GPU da 8 GB o su un Kaggle gratuito, e si rigenera ogni anno
quando cambiano le dispense — cosa che un training completo non permetterebbe.

**Mascheramento della loss sul solo turno dell'assistente.** Il modello non
impara a riscrivere la domanda o il prompt di sistema: capacità sprecata su token
che a scuola non deve generare.

**Cache semantica delle domande.** Le domande in una classe si ripetono. Due
domande con similarità Jaccard ≥ 0,9 sulle parole di contenuto condividono la
risposta: nella misura, il 50% delle richieste del secondo giro viene servito
dalla cache. La cache vale a parità di cronologia: se la conversazione è
diversa, il prompt è diverso e la risposta viene rigenerata. Anche le astensioni
vengono memorizzate, così una domanda fuori ambito ripetuta non riesegue il
retrieval. Risposte e fonti vengono copiate in ingresso e in uscita: un chiamante
che modifica l'oggetto restituito non può alterare ciò che riceve un altro
studente.

**Astensione come funzione di prima classe.** Soglia sul punteggio BM25 *e*
numero minimo di termini in comune. L'ultimo tentativo (una parola sola) è
ammesso solo se la domanda appartiene a una materia riconosciuta e il passaggio è
di quella materia: una domanda fuori ambito non deve aggrapparsi a una parola
qualsiasi. In una scuola un "non lo so" è un risultato, non un errore.

**Espansione controllata della query.** Un dizionario di sinonimi settoriali
(`stallatico` → `letame`, `avvicendamento` → `rotazione`, `corso d'acqua` →
`fiume`) vive in `configs/tutor.yaml` ed è **attivo per default**. Misurato con
`scripts/evaluate_tutor.py --compare`: sul set originale il recall@3 resta 30/30
con astensione 4/4, sul set di parafrasi si passa da 0/39 a 39/39 e il p95 da
0,9 a 1,0 ms. Da spento, una domanda in ambito riscritta con un sinonimo veniva
rifiutata (37 astensioni su 39): per un servizio scolastico non è un default
accettabile. La sostituzione avviene solo sulla stringa di retrieval e sulla
selezione delle frasi del backend estrattivo: la domanda mostrata, la chiave
della cache e il limite di 800 caratteri restano quelli originali. Il dizionario
non introduce dipendenze e non modifica l'indice; le regole più lunghe hanno la
precedenza, così una frase non viene spezzata da una regola più corta.

**Degradazione controllata.** Se il backend generativo fallisce (API giù, modello
non caricato), il servizio risponde con l'estrattivo e lo dichiara
(`degraded: true`) invece di mostrare un errore allo studente.

**Streaming SSE con le fonti per prime.** Il primo evento sono le fonti: lo
studente vede subito su cosa si basa la risposta, e può accorgersi che sono
sbagliate prima che il modello finisca di scrivere. L'evento finale dichiara se
la risposta è un'astensione: in quel caso l'interfaccia non mostra fonti.

**Tokenizzazione italiana a 32K.** `scripts/train_tokenizer.py` (già nel
repository) misura ~4,1 caratteri per token su italiano, contro ~3,3 del
tokenizer GPT-2: a parità di contenuto servono meno token, quindi meno contesto,
quindi meno calcolo per risposta.

**Nel modello LLigMa, per quando si addestra davvero:** MLA (compressione latente
di KV, DeepSeek-V2/V3), RoPE, RMSNorm con kernel fuso, SwiGLU, embedding
condivisi, `scaled_dot_product_attention` — già implementati in `lligma/model/`.

---

## 4. Avvio

```bash
cd "LLigMa 3.0 (Bare)"

# 1. indice (una volta sola, o quando cambiano le schede)
python scripts/build_tutor_index.py

# 2. servizio
python -m lligma.tutor.server            # http://127.0.0.1:8080
python -m lligma.tutor.server --backend extractive   # senza alcun modello
```

Nessuna dipendenza obbligatoria: serve solo `PyYAML`. Il server usa `http.server`
della libreria standard.

### Schede di conoscenza

`data/tutor/*.jsonl`, un documento per riga:

```json
{"id": "agro.suolo", "title": "Il suolo agrario", "subject": "agronomia",
 "source": "Dispensa interna", "text": "..."}
```

Per aggiungere materiale: crea il file, rilancia `build_tutor_index.py`. Le
materie sono `agronomia`, `biologia`, `cultura`. Il salvataggio dell'indice è
atomico: scrive un file temporaneo, lo sincronizza e sostituisce il vecchio solo
al termine. Un aggiornamento interrotto lascia quindi intatto l'indice
precedente, invece di lasciare il tutor senza una base valida.

---

## 5. Addestrare lo studente

```bash
# 0. chiave del docente in .env (mai nel repository)
cp .env.example .env

# 1. dataset di distillazione ancorato alle schede
python scripts/distill_tutor.py --per-chunk 2 --out data/tutor/distill.jsonl
python scripts/distill_tutor.py --list-models      # quali modelli esistono

# 2. fine-tuning QLoRA del modello piccolo
python scripts/finetune_tutor.py \
    --model Qwen/Qwen3-0.6B \
    --data data/tutor/distill.jsonl \
    --output-dir outputs/tutor-ft \
    --epochs 2 --batch-size 4 --gradient-accumulation-steps 4 --merge

# 3. servire il modello
#    configs/tutor.yaml → local.model_path: outputs/tutor-ft/merged
python -m lligma.tutor.server
```

Il servizio legge `.env` automaticamente (`lligma/tutor/env.py`): una variabile
già presente nell'ambiente ha la precedenza sul file, così una chiave iniettata da
Docker o da un servizio di sistema non viene sovrascritta da un `.env` vecchio.

Il modello base è configurabile: `Qwen/Qwen3-0.6B` è il default, ma lo stesso
script funziona con altri modelli piccoli (SmolLM2-360M-Instruct,
granite-4.0-nano, LFM2-350M). Verifica il nome esatto sul tuo mirror prima di
lanciare: il comando legge il modello da Hugging Face.

**Hardware.** Lo script richiede `torch`, `transformers`, `peft`, `bitsandbytes`.
Su questa macchina (Windows, Radeon RX 9060 XT) il percorso ROCm è documentato
come non utilizzabile: vedi la nota nel `README.md` principale. Il posto pratico
per addestrare è il notebook Kaggle già presente in `notebooks/`, che ha CUDA
funzionante e 12 ore per sessione.

### Revoca della chiave a fine addestramento

La chiave del docente serve **solo** alla distillazione. Dopo il fine-tuning il
modello studente risponde da sé e il backend `local` non legge alcuna chiave:
non c'è niente da riconfigurare, si smette semplicemente di usarla. La
procedura, dall'inizio alla fine.

**Dove sta la chiave.** In un solo posto: `LLigMa 3.0 (Bare)/.env`, accanto a
`.env.example` e fuori dal versionamento (`.gitignore`). Il percorso non dipende
dalla cartella da cui lanci il comando: `lligma/tutor/env.py` risolve la radice
con `Path(__file__).parents[2]`. Nel codice e nei YAML la chiave si legge da
`OPENAI_API_KEY` tramite `${OPENAI_API_KEY}`.

**La precedenza.** Una variabile già presente nell'ambiente vince sul file: una
chiave iniettata da Docker, da `systemd` o da PowerShell non viene sovrascritta
da un `.env` vecchio, e un `.env` con il campo vuoto non cancella la variabile
d'ambiente. Per questo, dopo la revoca, basta togliere la chiave dall'ambiente
oltre che dal file.

**La revoca.** Revoca la chiave dal provider e svuota `OPENAI_API_KEY` nel
`.env` (oppure elimina il file). Il modello addestrato è già su disco: la chiave
non serve più né per servire né per valutare.

**L'avvio dopo la revoca.** Senza `auto`, così non c'è nessun backend da
scegliere:

```bash
cd "LLigMa 3.0 (Bare)"
python -m lligma.tutor.server --backend extractive   # sempre: nessun modello richiesto
python -m lligma.tutor.server --backend local        # dopo il fine-tuning
```

Il servizio regge comunque una chiave non più valida: un 401 diventa
`BackendError` e il servizio riprova con il backend estrattivo, restituendo
`degraded: true` nella risposta. Il motivo per cui conviene togliere la chiave è
la latenza: `auto` sceglie il backend all'avvio in base alla *presenza* della
chiave, non alla sua validità, quindi con una chiave revocata ancora presente
ogni domanda paga prima un tentativo HTTP destinato a fallire.

---

## 6. Valutazione

`data/tutor/eval.jsonl` contiene 34 domande: 30 in ambito con parole chiave
attese, 4 deliberatamente fuori ambito. `data/tutor/eval-paraphrase.jsonl`
contiene 39 domande parafrasate su 27 schede dello stesso corpus, con la scheda
attesa già presente nel repository: usano sinonimi assenti dal vocabolario
dell'indice (`stallatico`, `pollina`, `monosuccessione`, `avvicendamento`,
`ammendante`, `Vate`, `corso d'acqua`, ma anche `granulometria`, `citosol`,
`fitoregolatore`, `CEE`, `PAC`, ecc.) e servono a misurare l'espansione, non a
sostituire il set originale.

```bash
python scripts/evaluate_tutor.py --scores
python scripts/evaluate_tutor.py --scores --json outputs/tutor-eval.json

# Il set di parafrasi con la configurazione distribuita (espansione già accesa)
python scripts/evaluate_tutor.py --eval data/tutor/eval-paraphrase.jsonl

# Il prima/dopo: l'espansione spenta si vede solo forzandola, e --compare lo fa

# Un'unica esecuzione: entrambi i set, espansione spenta e accesa
python scripts/evaluate_tutor.py --compare
```

`--compare` stampa recall@3, top@1, copertura, astensione e p95 per ciascuna
delle quattro combinazioni. Il fallimento del set parafrasi a espansione spenta
è atteso e non fa uscire il comando in errore; il comando fallisce se una
configurazione che deve passare (set originale in entrambe le configurazioni,
parafrasi espanse) non passa. Una misura senza `--compare` usa invece la
configurazione distribuita, quindi con l'espansione accesa.

Misure reali con backend estrattivo. La configurazione distribuita è quella con
l'espansione **accesa**; le righe «spenta» sono il confronto, non il default:

| Set | Espansione | recall@3 | top@1 | copertura | astensione | p95 |
|---|---|---|---|---|---|---|
| `eval.jsonl` | spenta | **30/30** | **30/30** | **1,00** | **4/4** | 1,0–1,7 ms |
| `eval.jsonl` | accesa (default) | **30/30** | **30/30** | **1,00** | **4/4** | 0,9–1,6 ms |
| `eval-paraphrase.jsonl` | spenta | **0/39** | **0/39** | 0,03 | n/d | 0,0–0,2 ms |
| `eval-paraphrase.jsonl` | accesa (default) | **39/39** | **39/39** | **1,00** | n/d | 1,0–1,8 ms |

Senza espansione nessuna delle 39 parafrasi raggiunge la scheda attesa: 37 si
astengono e 2 rispondono su una scheda sbagliata. La copertura residua di 0,03
viene da una parola chiave presente in una di quelle due risposte errate, non
da un recupero corretto. Con il dizionario attivo tutte le parafrasi hanno la
scheda attesa in prima posizione e come prima fonte, e la copertura delle parole
chiave attese sale a 1,00. Le 14 domande precedenti restano 0/14 senza espansione
e 14/14 con espansione; le 25 nuove passano da 0/25 a 25/25, sempre con la scheda
attesa in prima posizione. La latenza resta nello stesso ordine di grandezza:
p95 1,0–1,8 ms con l'espansione, 0,9–1,7 ms sul set originale. Il default
distribuito è l'espansione accesa: sul set originale non sposta nulla (30/30 e
astensione 4/4 anche a dizionario spento, quindi nessuna regola serve a far
passare `eval.jsonl`), mentre sul set di parafrasi è la differenza fra 0/39 e
39/39. Il costo misurato è nell'ordine del decimo di millisecondo sul p95.

La modalità senza chiave è verificata anche end-to-end: `tests/test_tutor_server_e2e.py`
avvia il server reale con `--backend extractive`, rimuove `OPENAI_API_KEY`,
sceglie una porta libera, interroga le due domande storiche e le quattro domande
fuori ambito, e verifica anche lo stream SSE di `/api/stream` (eventi
incrementali, evento finale `done`, stesse fonti e citazioni di `/api/ask`).
Un terzo test chiude la socket a metà stream e verifica che il server resti
sano, che serva ancora stream e ask completi e che il log non contenga
traceback. Il ramo di scrittura fallita è inoltre coperto in modo deterministico
da un test mirato sul writer SSE con un sink chiuso. Il test E2E usa un delay
solo-test tra gli eventi e verifica il contatore additivo `stream_aborts` in
`/api/health`. Un quarto test tiene sovrapposti cento client abortiti con RST e
un client completo, e verifica il delta esatto di `stream_aborts` e il payload
SSE identico byte per byte. Il server di produzione contiene solo il lock reale:
barriera, trace sull'opcode `STORE_ATTR` e delay SSE vivono esclusivamente in
`tests/tutor_server_instrumented.py`, l'entrypoint test-only che il test avvia
al posto del modulo di produzione. Questi controlli non modificano gli eventi
SSE. Il comando ripetibile è `python tests/mutation_stream_aborts.py` e copre
due invarianti distinti. Prima di applicare una mutazione crea una transazione
nascosta `.lligma-mutation-<id>` con i byte e gli hash originali e avvia
`tests/mutation_watchdog.py` fuori dall'albero del processo: su Windows usa
`Win32_Process.Create` via WMI, quindi un `taskkill /T` non lo raggiunge; su
POSIX usa una nuova sessione. I `finally` ripristinano e verificano i file; se
il processo principale viene ucciso senza eseguirli, il watchdog ripristina dai
backup, verifica gli hash e rimuove la transazione. Se WMI non è disponibile il
comando fallisce prima di mutare, invece di usare un watchdog figlio. All'avvio
`recover_pending()` recupera eventuali transazioni orfane. Per
`stream_aborts` il comando esegue cinque run con il lock reale, applica la
mutazione non atomica, esegue tre run mutate e ripristina sempre il file dal
backup. In tre esecuzioni consecutive del comando le run protette sono state
15/15 con delta 100; le tre run mutate di ciascuna esecuzione sono fallite 9/9,
con valori osservati 3, 4, 4; 2, 3, 2; e 3, 3, 4 su 100. Nella verifica del
watchdog WMI le tre run mutate hanno osservato 96, 86 e 89, sempre sotto 100.
Se all'avvio trova un
file mutato senza backup, ripristina la forma nota ma dichiara FAIL: senza hash
originale non può dimostrare la byte-identità. Il test dedicato
`python tests/mutation_interruption.py` uccide l'intero albero del comando a
mutazione applicata, verifica hash, transazione e watchdog, poi lascia una
transazione orfana e esegue il comando completo: deve comparire `RECOVERY:` e
`VERDICT: PASS`. La verifica permanente e più ampia è
`python tests/mutation_safety.py`: non è interattiva, non fa parte dei 93 test
pytest e prova in sequenza il watchdog out-of-tree contro il kill dell'albero,
la transazione orfana con backup, l'indisponibilità simulata del meccanismo
out-of-tree, una transazione stale con marker clean, il riuso del PID del padre
e il recovery concorrente protetto da lock. Ogni scenario verifica i due hash
e l'assenza di transazioni, backup, lock, watchdog e log residui; il comando
stampa `MUTATION_SAFETY: PASS` solo se tutti passano. In caso di errore il test
harness ripristina comunque i byte iniziali e ripulisce i residui. La modalità
out-of-tree assente si simula con la variabile test-only
`LLIGMA_MUTATION_DISABLE_OUT_OF_TREE=1`; in quel caso il comando di mutazione
deve uscire 2 prima di applicare la mutazione. Eseguita due volte di fila,
la verifica ha stampato `MUTATION_SAFETY: PASS` entrambe le volte, con la
seconda partenza da hash e transazioni già puliti; al termine i tre hash sono
rimasti quelli originali e non sono restati transazioni, lock, watchdog o log
di test. `tests/mutation_safety.py` e gli altri `tests/mutation_*.py` restano
script standalone; il controllo negativo è il test pytest che li orchestra.

### Controllo negativo automatico

Il controllo è automatizzato in `tests/test_mutation_negative_control.py`; si
esegue da solo o con pytest:

```bash
python tests/test_mutation_negative_control.py
python -m pytest tests/test_mutation_negative_control.py -q
```

Il test salva in memoria i sette file congelati, applica una regressione per
volta a `tests/mutation_transaction.py`, esegue `tests/mutation_safety.py` e
pretende exit 1 con il FAIL specifico. Il ripristino avviene in `finally` e
viene verificato contro gli hash; un canary separato dimostra che un ripristino
disattivato viene davvero rilevato. Non lascia transazioni, lock o log: al
termine gli hash sono quelli iniziali.

Nel caso `TREE_KILL` il FAIL deve nominare il watchdog ucciso da `taskkill /T`,
non un timeout o un traceback. Nel caso out-of-tree il comando non protetto può
uscire `0` oppure `1`, perché la suite E2E può fallire sotto carico prima della
mutazione, oppure può restare in esecuzione fino al timeout di 240 s del
safety. In ogni caso il safety deve uscire `1` e riportare `atteso exit 2,
ottenuto ...`. Accettare questi esiti non indebolisce il guard: il contratto è
che il comando venga fermato prima della mutazione con exit 2; un exit 0/1, o
un comando ancora vivo, dimostra che non è stato fermato.

La procedura manuale seguente resta come diagnostica per capire un FAIL:

1. **Regressione TREE_KILL.** Sostituire il ramo Windows che chiama
   `_start_wmi_watchdog(transaction, root)` con un normale `subprocess.Popen`
   del watchdog, senza `Win32_Process.Create` e senza nuova sessione. Il
   watchdog diventa così figlio del comando e `taskkill /T` lo uccide insieme
   all'albero. Eseguire `python tests/mutation_safety.py`: deve terminare con
   exit diverso da zero e stampare `TREE_KILL: FAIL`, oltre a
   `MUTATION_SAFETY: FAIL`; non deve arrivare a `TREE_KILL: PASS`. Il FAIL
   atteso nomina esplicitamente `watchdog non sopravvissuto a taskkill /T`, non
   un timeout di scenario, un errore di import o un traceback di
   orchestrazione. L'harness controlla la vita del watchdog mentre attende il
   ripristino: se muore prima che i file siano tornati agli hash originali,
   fallisce subito; il timeout di ripristino resta quello invariato.
2. **Guard out-of-tree ignorato.** Cambiare temporaneamente il confronto
   `LLIGMA_MUTATION_DISABLE_OUT_OF_TREE == "1"` in `== "0"`, così la variabile
   non blocca più la creazione della transazione. Eseguire il safety: deve
   stampare `OUT_OF_TREE_UNAVAILABLE: FAIL` con “atteso exit 2, ottenuto ...”
   (`0` o `1` a seconda che la suite non protetta passi o fallisca, oppure il
   timeout dopo 240 s se il comando resta in esecuzione) e
   `MUTATION_SAFETY: FAIL`; il comando di mutazione, partendo davvero, non deve
   essere scambiato per un successo del guard.
3. **Ripristino.** Ricopiare i byte originali, verificare che l'hash coincida
   con quello salvato, rimuovere transazioni, lock, backup e file temporanei,
   quindi eseguire due volte `python tests/mutation_safety.py` partendo da
   stato pulito. Devono comparire due `MUTATION_SAFETY: PASS` consecutivi; solo
   allora il meccanismo è considerato integro.

Eseguito su questa macchina, il safety ha prodotto exit 1 per entrambi i casi.
Con l'harness definitivo:

```text
TREE_KILL: FAIL: AssertionError: watchdog non sopravvissuto a taskkill /T: pid ... morto prima del ripristino (transazione=..., hash_restored=False)
OUT_OF_TREE_UNAVAILABLE: FAIL: AssertionError: out-of-tree assente: atteso exit 2, ottenuto 0 oppure 1 (o "un comando ancora in esecuzione dopo 240s")
MUTATION_SAFETY: FAIL
# il numero dipende dalla suite E2E non protetta: 0 se passa, 1 se fallisce
```

Il primo FAIL non è un timeout e non contiene un traceback di orchestrazione;
il secondo riporta l'exit reale del comando. Dopo il ripristino i sette hash
erano identici a quelli iniziali e non restavano transazioni, lock o processi
watchdog. La verifica automatica eseguita il 6 novembre 2026 su una copia
isolata ha prodotto `2 passed in 304.95 s`, exit 0, con gli hash iniziali
invariati. La suite completa nella stessa copia ha prodotto
**`93 passed in 348.19 s`**, exit 0, con 93 test raccolti; la sorgente non è
stata modificata durante la campagna.

Per il secondo invariante il comando esegue cinque run della suite
`tests/test_tutor_service.py` con pytest reale, poi sostituisce i tre
`with self._lock:` di `AnswerCache` con un blocco sempre vero e lancia tre run
mutate. Nelle stesse tre esecuzioni della campagna le run protette sono state
15/15, con 23/23 test passati ogni volta; le run mutate sono fallite 9/9 con
1 failed e 22 passed, sempre su
`test_cache_serializes_concurrent_mutations`: due scritture entrano nella cache
fuori dal lock. Il `finally` di `tests/mutation_cache_lock.py` e quello esterno
ripristinano sempre il file byte-identico al backup, con tre lock presenti e
mutazione assente. Anche questo percorso è coperto dalla stessa transazione:
il backup non viene cancellato finché il ripristino non è verificato, e il
watchdog resta attivo fino al clean marker.

Il test E2E di disconnessione anticipata chiude intenzionalmente la socket con
RST. `BaseHTTPRequestHandler` può tentare di leggere la richiesta successiva e
`socketserver` registra un traceback `ConnectionResetError`: è un rumore atteso
del client, non un errore applicativo. L'entrypoint test-only
`tests/tutor_server_instrumented.py` consuma esclusivamente questa eccezione
attesa in `handle_one_request` e chiude la connessione; qualsiasi altra
eccezione continua a propagarsi e gli assert su payload SSE, abort e log
restano invariati. Dopo la correzione 25/25 run E2E complete consecutive sono
passate, e le tre esecuzioni del comando sono tutte PASS.

Dopo le due fasi il comando esegue ancora due fault-injection pytest reali:
ciascuna applica la mutazione e avvia un vero subprocess `python -m pytest` sul
caso `tests/mutation_fault_case.py`. Il primo solleva `RuntimeError` nel test e
pytest esce con codice 1; il secondo solleva `KeyboardInterrupt` durante la run
e pytest esce con codice 2. In ognuna delle tre esecuzioni entrambi i fault
hanno ripristinato il file byte-identico al backup, con lock presente e
mutazione assente. Processo e porta vengono chiusi anche in caso di errore.

La suite completa conta **93 test** (il numero corrente si verifica con
`python -m pytest tests --collect-only -q`). I controlli coprono la copia profonda della
cache, la cache delle astensioni SSE, la scrittura atomica dell'indice, il
limite `Host` sul loopback, la chiusura delle connessioni con corpo non letto,
l'espansione delle query (baseline 0/39, espansa 39/39, scheda attesa in prima
posizione, default acceso, cache e limite 800 invariati), il report unico
`--compare` e il comportamento del client web.

Il client web ha finalmente una prova eseguibile, non solo sintattica:
`tests/test_tutor_web_client.py` invoca `tests/web_client_harness.js`, che
estrae lo script inline da `web/tutor/index.html` e lo esegue in Node con un
DOM e un `fetch` minimi. Lo scenario verifica che la cronologia venga aggiornata
solo dopo un evento `done`, che uno stream interrotto non produca un turno
assistant e che un evento `error`, anche seguito da `done`, non lasci una
risposta parziale nella cronologia. Usa solo moduli standard di Node e Python:
nessuna dipendenza nuova.

Misure riportate (backend estrattivo, CPU, corpus di 34 schede / 78 passaggi):

| Metrica | Valore | Lettura |
|---|---|---|
| recall@3 | **100%** (30/30) | il passaggio giusto è sempre nei primi tre |
| copertura media | **1,00** (30/30) | quota di parole chiave attese presenti nella risposta |
| astensione | **4/4** | non risponde mai alle 4 domande fuori ambito |
| latenza media | **~0,5 ms** | p95 1,1–1,8 ms tra benchmark e valutazioni |
| cache hit rate | 50% | misurato con due giri di richieste |

Separazione dei punteggi BM25, che è ciò che rende possibile la soglia:

| | min | mediana | max |
|---|---|---|---|
| in dominio | 5,93 | 15,77 | 24,95 |
| fuori dominio | 2,72 | 3,70 | 4,45 |

Le due distribuzioni non si sovrappongono: la soglia `min_score: 2.5` più il
vincolo sui termini è calibrata su questo corpus. Se il corpus cresce di un
ordine di grandezza, **ricalibrala**: `evaluate_tutor.py --scores` esiste per
questo.

### Limiti noti, misurati

La copertura è 1,00 sulle 30 domande in ambito, compresi i due casi che prima
erano a 0,00: "come si passa dal DNA alla proteina" e "in che anno è ambientato
il romanzo di Manzoni". Il recupero era già corretto; sbagliava la selezione
delle frasi. Due correzioni, entrambe nel backend estrattivo:

* una domanda che chiede una data ammette anche frasi senza parole in comune,
  purché contengano un anno e provengano da un passaggio quasi pertinente;
* quando una frase compare in due chunk consecutivi (overlap), la sua
  occorrenza successiva viene usata per aggiungere la continuazione: è così che
  "nella trascrizione" porta a "nella traduzione" e a "codone".

Il limite resta strutturale: l'estrattivo può solo selezionare frasi già
presenti, non sintetizzare un fatto che richieda di combinarne due o di
calcolarlo. Su domande che chiedono un'inferenza il tutor può quindi ancora
astenersi o fermarsi alla frase più vicina; il modello distillato serve a
superare questo limite, non a colmarlo nel riferimento. La copertura misura
parole chiave attese, non correttezza semantica: un 1,00 su un eval curato non
è una garanzia su domande nuove.

Una domanda in ambito ("che cosa è l'humus e perché è importante") condivide una
sola parola di contenuto con la scheda; viene recuperata dal tentativo
ristretto per materia, ed è il motivo per cui quel tentativo esiste.

Il recupero in dominio non garantisce la scheda in testa quando la domanda
condivide due termini con una scheda e uno solo con quella giusta: «come
respirano le piante?» non si astiene più — prima la forma verbale `respirano`
non coincideva con `respirazione` — ma mette davanti `bio.fotosintesi`, che
contiene sia `respir` sia `piante`, invece di `bio.respirazione`, che contiene
solo `respir` e viene scartata dal vincolo dei due termini. Il caso è lasciato
visibile: correggerlo con una regola ad hoc significherebbe calibrare il
retriever sulla domanda che si è appena letta.

---

## 7. Uso responsabile a scuola

* **Solo loopback per default.** `server.host: 127.0.0.1`: per esporlo sulla rete
  didattica serve un reverse proxy con autenticazione e quota, non una modifica
  a questo file.
* **Solo il sito stesso può chiamare le API.** Un POST con header `Origin` di un
  altro sito viene rifiutato con 403: una pagina qualsiasi aperta nel browser
  dello studente non può usare la chiave del tutor su localhost. I client non
  browser (curl, script) non mandano `Origin` e restano ammessi.
* **Host loopback controllato.** Con il bind predefinito il server accetta solo
  `127.0.0.1`, `localhost` e `::1` nell'header `Host`, salvo una lista esplicita
  in `server.allowed_hosts`. Questo chiude il DNS rebinding, che il solo
  confronto Origin/Host non intercetta. Le richieste rifiutate con un corpo non
  letto chiudono la connessione, quindi i byte residui non possono essere
  interpretati come una seconda richiesta. Le risposte includono anche
  `frame-ancestors 'none'` e `X-Frame-Options: DENY`.
* **Nessuna chiave nel repository.** `OPENAI_API_KEY` vive in `.env`, che è in
  `.gitignore`. Se una chiave è stata condivisa in chat o in un file, va
  revocata e rigenerata: una chiave incollata in un documento è una chiave
  compromessa, indipendentemente da dove finisce. La procedura completa,
dall'uso alla revoca, è nella sezione 5.
* **Nessun dato dello studente salvato.** Il servizio non scrive log delle
  domande su disco e non mantiene sessioni: la cache è in memoria e muore con il
  processo.
* **Le risposte citano sempre la fonte**, e il prompt di sistema vieta
  esplicitamente di usare conoscenza esterna al contesto. Non è una garanzia
  formale, è una riduzione strutturale della superficie di errore.

---

## 8. Cosa questo sistema non è

* Non è un modello addestrato da zero: è retrieval più un piccolo modello
  distillato. Se l'obiettivo è "un LLigMa da zero", questo è un altro progetto,
  con altri costi.
* Non risponde su materie fuori dalle tre previste, e lo dichiara.
* Non sostituisce il docente: per trattamenti fitosanitari, salute o decisioni
  aziendali il prompt rimanda esplicitamente all'insegnante o al tecnico.

---

## 9. Prossimi passi sensati, in ordine di rapporto valore/costo

1. **Popolare le schede** con le dispense reali dell'istituto: è la leva
   principale, e non richiede addestramento.
2. **Addestrare lo studente** sul dataset di distillazione e confrontarlo con
   l'estrattivo sullo stesso `eval.jsonl`. Se non migliora la copertura, non
   serve.
3. **Estendere il dizionario di sinonimi**: `lligma/tutor/expansion.py` e
   `configs/tutor.yaml` sono il punto di ingresso; ogni nuovo sinonimo va
   misurato su `eval-paraphrase.jsonl` e non deve spostare le domande fuori
   ambito.
4. **Quantizzazione e draft model per la decodifica speculativa**, quando il
   modello locale è in produzione: la decodifica speculativa richiede un
   modello bozza allineato al vocabolario e va misurata, non presunta.
