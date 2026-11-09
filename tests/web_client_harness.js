'use strict';

// Eseguibile con Node, senza dipendenze: estrae lo script inline del client e
// lo esegue contro un DOM/fetch minimale. Non e' un browser vero, ma copre la
// logica che il test sintattico non poteva vedere: done aggiorna la cronologia,
// stream interrotto e evento error non creano turni assistant parziali.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT = path.resolve(__dirname, '..');
const HTML_PATH = path.join(ROOT, 'web', 'tutor', 'index.html');
const HTML = fs.readFileSync(HTML_PATH, 'utf8');
const scripts = [...HTML.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/gi)];
if (scripts.length !== 1) {
  throw new Error(`attesi 1 script inline, trovati ${scripts.length}`);
}
const CLIENT_SCRIPT = scripts[0][1];

function makeElement(tagName = 'div') {
  const listeners = new Map();
  const classes = new Set();
  const element = {
    tagName: String(tagName).toUpperCase(),
    textContent: '',
    innerHTML: '',
    className: '',
    style: {},
    dataset: {},
    value: '',
    disabled: false,
    title: '',
    children: [],
    _listeners: listeners,
    classList: {
      add(...names) {
        names.forEach((name) => classes.add(name));
        element.className = [...classes].join(' ');
      },
      remove(...names) {
        names.forEach((name) => classes.delete(name));
        element.className = [...classes].join(' ');
      },
      contains(name) {
        return classes.has(name);
      },
    },
    appendChild(child) {
      element.children.push(child);
      return child;
    },
    addEventListener(type, handler) {
      if (!listeners.has(type)) listeners.set(type, []);
      listeners.get(type).push(handler);
    },
    setAttribute(name, value) {
      element[name] = String(value);
    },
    querySelector(selector) {
      if (selector === '.risposta') {
        if (!element._response) element._response = makeElement('div');
        return element._response;
      }
      return null;
    },
    closest(selector) {
      return selector === '.chip' && Object.prototype.hasOwnProperty.call(element.dataset, 'materia')
        ? element
        : null;
    },
    scrollIntoView() {},
    focus() {},
    requestSubmit() {
      for (const handler of listeners.get('submit') || []) {
        handler({ preventDefault() {} });
      }
    },
  };
  return element;
}

function makeDom() {
  const ids = {
    stato: makeElement('div'),
    conversazione: makeElement('main'),
    vuoto: makeElement('div'),
    modulo: makeElement('form'),
    domanda: makeElement('input'),
    invia: makeElement('button'),
    materie: makeElement('div'),
  };
  const chips = ['', 'agronomia', 'biologia', 'cultura'].map((materia) => {
    const chip = makeElement('button');
    chip.dataset.materia = materia;
    return chip;
  });
  const examples = [
    'Perché si alternano le colture nello stesso campo?',
    'Cosa succede nella fase luminosa della fotosintesi?',
  ].map((text) => {
    const button = makeElement('button');
    button.textContent = text;
    return button;
  });
  const document = {
    getElementById(id) {
      return ids[id] || null;
    },
    querySelectorAll(selector) {
      if (selector === '.chip') return chips;
      if (selector === '.esempi button') return examples;
      return [];
    },
    createElement(tagName) {
      return makeElement(tagName);
    },
  };
  return { document, ids };
}

function sseBytes(events, chunkSize = 7) {
  const payload = events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join('');
  const bytes = new TextEncoder().encode(payload);
  const chunks = [];
  for (let index = 0; index < bytes.length; index += chunkSize) {
    chunks.push(bytes.slice(index, index + chunkSize));
  }
  return chunks;
}

function makeFetch(scenarios) {
  const calls = [];
  let scenarioIndex = 0;
  const fetch = async (url, options = {}) => {
    if (url === '/api/health') {
      return {
        ok: true,
        async json() {
          return { backend: 'extractive', chunks: 78, documents: 34 };
        },
      };
    }
    if (url !== '/api/stream') {
      throw new Error(`richiesta inattesa: ${url}`);
    }
    calls.push(JSON.parse(options.body || '{}'));
    const events = scenarios[scenarioIndex++] || [];
    const chunks = sseBytes(events);
    let cursor = 0;
    return {
      ok: true,
      body: {
        getReader() {
          return {
            async read() {
              if (cursor >= chunks.length) return { done: true, value: undefined };
              return { done: false, value: chunks[cursor++] };
            },
          };
        },
      },
    };
  };
  return { fetch, calls };
}

function makeClient(scenarios) {
  const dom = makeDom();
  const transport = makeFetch(scenarios);
  const context = {
    document: dom.document,
    fetch: transport.fetch,
    TextDecoder,
    TextEncoder,
    setTimeout,
    clearTimeout,
    console,
  };
  vm.runInNewContext(CLIENT_SCRIPT, context, { filename: HTML_PATH });
  return { ...dom, calls: transport.calls };
}

const tick = (milliseconds = 0) => new Promise((resolve) => setTimeout(resolve, milliseconds));

async function submit(client, question) {
  client.ids.domanda.value = question;
  const handlers = client.ids.modulo._listeners.get('submit') || [];
  assert.strictEqual(handlers.length, 1, 'il form deve avere un solo handler submit');
  handlers[0]({ preventDefault() {} });
  for (let attempt = 0; attempt < 1000; attempt += 1) {
    if (client.ids.invia.disabled === false) return;
    await tick(1);
  }
  throw new Error('il client non ha concluso la richiesta');
}

const normalEvents = (text) => [
  {
    type: 'meta',
    sources: [{ subject: 'Agronomia', title: 'Scheda', quote: 'Citazione', numbers: [1] }],
    backend: 'extractive',
    cached: false,
    abstained: false,
    subject: 'agronomia',
    subject_label: 'Agronomia',
  },
  { type: 'token', text },
  { type: 'done', latency_ms: 1.5, backend: 'extractive', cached: false, abstained: false },
];

async function doneUpdatesHistory() {
  const client = makeClient([normalEvents('Risposta completa.'), normalEvents('Seconda risposta.')]);
  await submit(client, 'Prima domanda');
  assert.deepStrictEqual(client.calls[0].history, [], 'la prima richiesta non deve avere cronologia');
  await submit(client, 'Seconda domanda');
  assert.deepStrictEqual(
    client.calls[1].history,
    [
      { role: 'user', content: 'Prima domanda' },
      { role: 'assistant', content: 'Risposta completa.' },
    ],
    'done deve aggiungere il turno completo alla cronologia',
  );
  console.log('ok done aggiorna la cronologia');
}

async function interruptedStreamDoesNotUpdateHistory() {
  const interrupted = [
    normalEvents('Risposta parziale.')[0],
    { type: 'token', text: 'Risposta parziale.' },
  ];
  const client = makeClient([interrupted, normalEvents('Risposta successiva.')]);
  await submit(client, 'Domanda interrotta');
  assert.deepStrictEqual(client.calls[0].history, []);
  await submit(client, 'Domanda successiva');
  assert.deepStrictEqual(client.calls[1].history, [], 'uno stream senza done non deve entrare in cronologia');
  console.log('ok stream interrotto non aggiorna la cronologia');
}

async function degradedReplacesPartialText() {
  const meta = normalEvents('')[0];
  const degraded = [
    meta,
    { type: 'token', text: 'Risposta parziale del modello.' },
    { type: 'degraded', message: 'Modello non disponibile: risposta estrattiva.' },
    { type: 'token', text: 'Risposta estrattiva completa.' },
    { type: 'done', latency_ms: 2.0, backend: 'extractive', cached: false, abstained: false },
  ];
  const client = makeClient([degraded, normalEvents('Dopo.')]);
  await submit(client, 'Domanda con modello rotto');
  const turno = client.ids.conversazione.children[0];
  assert.strictEqual(
    turno._response.innerHTML,
    'Risposta estrattiva completa.',
    'il testo parziale del modello non deve restare sopra la risposta estrattiva',
  );
  await submit(client, 'Domanda successiva');
  assert.deepStrictEqual(
    client.calls[1].history,
    [
      { role: 'user', content: 'Domanda con modello rotto' },
      { role: 'assistant', content: 'Risposta estrattiva completa.' },
    ],
    'la cronologia deve contenere solo la risposta estrattiva',
  );
  console.log('ok degraded sostituisce il testo parziale');
}

async function errorEventDoesNotUpdateHistory() {
  const failed = [
    normalEvents('Risposta parziale.')[0],
    { type: 'token', text: 'Risposta parziale.' },
    { type: 'error', message: 'Errore simulato' },
    { type: 'done', latency_ms: 1.0, backend: 'extractive', cached: false, abstained: false },
  ];
  const client = makeClient([failed, normalEvents('Risposta successiva.')]);
  await submit(client, 'Domanda fallita');
  await submit(client, 'Domanda successiva');
  assert.deepStrictEqual(client.calls[1].history, [], 'un evento error non deve lasciare un turno assistant');
  console.log('ok evento error non aggiorna la cronologia');
}

(async () => {
  await doneUpdatesHistory();
  await interruptedStreamDoesNotUpdateHistory();
  await degradedReplacesPartialText();
  await errorEventDoesNotUpdateHistory();
  console.log('web client harness: PASS');
})().catch((error) => {
  console.error(error.stack || error);
  process.exitCode = 1;
});