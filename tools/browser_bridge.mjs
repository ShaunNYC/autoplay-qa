/** Generic visible-UI browser transport. No game rules or provider access. */
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import readline from 'node:readline';
import path from 'node:path';

export const PROTOCOL = 'autoplay-browser/1';
export const CLOCK_EPOCH = '2026-01-01T00:00:00.000Z';
const hash = value => createHash('sha256').update(value).digest('hex');
const fail = (code, message) => Object.assign(new Error(message), { code });

export function validateUrl(value) {
  let url;
  try { url = new URL(value); } catch { throw fail('invalid_url', 'A local HTTP URL is required.'); }
  if (url.protocol !== 'http:' || !['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname) || url.username || url.password) {
    throw fail('invalid_url', 'Only loopback HTTP origins without credentials are permitted.');
  }
  return url;
}

export function validateAction(action) {
  if (!action || typeof action !== 'object' || Array.isArray(action)) throw fail('invalid_action', 'Control must be an object.');
  const fields = { click: ['kind', 'target'], input: ['kind', 'target', 'value'], select: ['kind', 'target', 'value'], wait: ['kind', 'milliseconds'] }[action.kind];
  if (!fields || Object.keys(action).some(key => !fields.includes(key)) || fields.some(key => !(key in action))) throw fail('invalid_action', 'Invalid atomic control fields.');
  if (action.kind === 'wait') {
    if (!Number.isInteger(action.milliseconds) || action.milliseconds < 0 || action.milliseconds > 10000) throw fail('invalid_action', 'Wait must be 0–10000 milliseconds.');
  } else if (typeof action.target !== 'string' || !action.target || action.target.length > 96) throw fail('invalid_action', 'A visible control ID is required.');
  if (['input', 'select'].includes(action.kind) && (typeof action.value !== 'string' || action.value.length > 4000)) throw fail('invalid_action', 'Input must be at most 4000 characters.');
  return action;
}

/** Evaluate only rendered labels/text and editable values, never app JS/storage. */
function visibleDocument() {
  const visible = element => {
    const style = getComputedStyle(element);
    for (let parent = element.parentElement; parent; parent = parent.parentElement) {
      if (parent.tagName === 'DETAILS' && !parent.open && !parent.querySelector(':scope > summary')?.contains(element)) return false;
    }
    return style.visibility !== 'hidden' && style.display !== 'none' && element.getClientRects().length > 0 && !element.closest('[hidden],[inert]');
  };
  const normalize = value => String(value ?? '').replace(/\s+/g, ' ').trim();
  const label = element => {
    const references = (element.getAttribute('aria-labelledby') || '').split(/\s+/).filter(Boolean).map(id => document.getElementById(id)).filter(item => item && visible(item));
    const labels = [...(element.labels || [])].filter(visible);
    return normalize(element.getAttribute('aria-label') || references.map(item => item.innerText).join(' ') || labels.map(item => item.innerText.replace(element.innerText || '\0', '')).join(' ') || element.innerText || element.getAttribute('title') || element.getAttribute('placeholder') || (element.tagName === 'INPUT' && ['button', 'submit'].includes(element.type) ? element.value : ''));
  };
  const selector = element => {
    const parts = [];
    for (let current = element; current && current !== document.documentElement; current = current.parentElement) {
      const siblings = [...current.parentElement.children].filter(item => item.tagName === current.tagName);
      parts.unshift(`${current.tagName.toLowerCase()}:nth-of-type(${siblings.indexOf(current) + 1})`);
    }
    return `html > ${parts.join(' > ')}`;
  };
  const controls = [...document.querySelectorAll('button,input,textarea,select,a[href],[role="button"],[role="textbox"],[contenteditable="true"],summary')].filter(visible).filter(element => !(element.tagName === 'INPUT' && ['hidden', 'password'].includes(element.type))).map(element => {
    const tag = element.tagName.toLowerCase();
    const data = {
      locator: selector(element), tag,
      role: element.getAttribute('role') || ({button:'button',textarea:'textbox',select:'combobox',a:'link',summary:'button'}[tag] || (tag === 'input' ? element.type : 'textbox')),
      name: label(element), value: '', disabled: Boolean(element.disabled || element.getAttribute('aria-disabled') === 'true'),
    };
    if (['input','textarea','select'].includes(tag) && element.type !== 'file') data.value = element.value;
    if (element.isContentEditable) data.value = element.innerText;
    if (tag === 'select') data.options = [...element.options].map(option => ({ value: option.value, label: option.text, disabled: option.disabled }));
    if (tag === 'input' && ['checkbox','radio'].includes(element.type)) data.checked = element.checked;
    if (tag === 'input' && element.type === 'file') data.accept = element.accept;
    return data;
  });
  return { title: document.title, text: document.body?.innerText || '', controls };
}

export async function createBrowser({ url, playwrightModule, executablePath, actionTimeout = 5000, randomSeed = 0 }) {
  const target = validateUrl(url);
  const require = createRequire(import.meta.url);
  const playwright = playwrightModule ? await import(pathToFileURL(path.resolve(playwrightModule)).href) : require('playwright');
  const chromium = playwright.chromium || playwright.default?.chromium;
  const browser = await chromium.launch({ headless: true, ...(executablePath ? { executablePath } : {}), args: ['--disable-background-networking', '--disable-component-update', '--no-first-run'] });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, locale: 'en-US', timezoneId: 'UTC', serviceWorkers: 'block', acceptDownloads: true, reducedMotion: 'reduce' });
  // The test is confined to the one local fixture. No keys/cookies or external pages.
  await context.route('**/*', route => {
    const requestUrl = new URL(route.request().url());
    return requestUrl.origin === target.origin || ['blob:', 'data:'].includes(requestUrl.protocol) ? route.continue() : route.abort('blockedbyclient');
  });
  if (!Number.isSafeInteger(randomSeed) || randomSeed < 0 || randomSeed > 0xffffffff) { await browser.close(); throw fail('invalid_seed', 'Random seed must be a uint32.'); }
  await context.addInitScript(seed => {
    let state = seed || 0x6d2b79f5;
    const next = () => { state ^= state << 13; state ^= state >>> 17; state ^= state << 5; return (state >>> 0) / 4294967296; };
    Math.random = next;
    Object.defineProperty(crypto, 'randomUUID', { value: () => {
      const bytes = Array.from({length:16}, () => Math.floor(next() * 256));
      bytes[6] = (bytes[6] & 15) | 64; bytes[8] = (bytes[8] & 63) | 128;
      const hex = bytes.map(value => value.toString(16).padStart(2,'0')).join('');
      return `${hex.slice(0,8)}-${hex.slice(8,12)}-${hex.slice(12,16)}-${hex.slice(16,20)}-${hex.slice(20)}`;
    } });
  }, randomSeed);
  const page = await context.newPage();
  page.setDefaultTimeout(actionTimeout);
  const downloads = [];
  let fileChooser = null;
  let dialog = null;
  let nativeDialog = null;
  let dialogWaiter = null;
  let pendingOperation = null;
  let promptValue = '';
  let lastSnapshot = null;
  let tick = 0;
  let pendingDownloads = [];
  let downloadError = null;
  let registry = new Map();
  page.on('filechooser', chooser => { fileChooser = chooser; });
  page.on('dialog', item => { nativeDialog = item; dialog = { type: item.type(), message: item.message() }; promptValue = item.defaultValue(); dialogWaiter?.(true); });
  page.on('download', download => {
    const task = (async () => {
      const stream = await download.createReadStream();
      if (!stream) throw fail('download_failed', 'The browser download did not complete.');
      const chunks = []; let size = 0;
      for await (const chunk of stream) { size += chunk.length; if (size > 1024 * 1024) throw fail('download_too_large', 'Browser artifact exceeds 1 MiB.'); chunks.push(chunk); }
      const buffer = Buffer.concat(chunks);
      downloads.push({ id: `download-${downloads.length + 1}`, name: download.suggestedFilename(), sha256: hash(buffer), bytes: buffer.length, buffer });
    })().catch(error => { downloadError = error; });
    pendingDownloads.push(task);
  });
  try {
    await page.clock.install({ time: new Date(Date.parse(CLOCK_EPOCH) - 60000) });
    await page.clock.pauseAt(new Date(CLOCK_EPOCH));
    await page.goto(target.href, { waitUntil: 'networkidle', timeout: 15000 });
    for (let startup = 0; startup < 10; startup += 1) {
      await page.clock.runFor(100);
      await new Promise(resolve => setTimeout(resolve, 25));
    }
  } catch (error) { await browser.close(); throw error; }

  async function observe() {
    await Promise.all(pendingDownloads); pendingDownloads = [];
    if (downloadError) throw downloadError;
    if (nativeDialog) {
      if (!lastSnapshot) throw fail('initial_dialog', 'The initial fixture opened a native dialog before its first observation.');
      return { ...structuredClone(lastSnapshot), text: `${lastSnapshot.text}\nBrowser ${dialog.type}: ${dialog.message}`, dialog, tick,
        controls: [
          ...(dialog.type === 'prompt' ? [{id:'dialog-prompt',tag:'input',role:'text',name:'Browser prompt',value:promptValue,disabled:false}] : []),
          {id:'dialog-accept',tag:'button',role:'button',name:'OK',value:'',disabled:false},
          ...(dialog.type === 'alert' ? [] : [{id:'dialog-dismiss',tag:'button',role:'button',name:'Cancel',value:'',disabled:false}]),
        ] };
    }
    const snapshot = await page.evaluate(visibleDocument);
    if (Buffer.byteLength(snapshot.text, 'utf8') > 100000 || snapshot.controls.length > 500) throw fail('observation_too_large', 'Visible page exceeds the recording bound; no truncated evidence was returned.');
    registry = new Map();
    snapshot.controls = snapshot.controls.map(control => {
      const id = `c-${hash(control.locator).slice(0, 16)}`;
      registry.set(id, control);
      const { locator, ...publicControl } = control;
      return { id, ...publicControl };
    });
    if (fileChooser) snapshot.controls.push({ id: 'file-chooser', tag: 'file', role: 'file', name: 'Choose a previously downloaded file', value: '', disabled: false });
    const current = new URL(page.url());
    lastSnapshot = { kind: 'browser-visible-v1', url: current.pathname + current.hash, ...snapshot, controls: snapshot.controls, downloads: downloads.map(({buffer, ...metadata}) => metadata), dialog, tick };
    return structuredClone(lastSnapshot);
  }

  async function perform(operation) {
    const opened = new Promise(resolve => { dialogWaiter = resolve; });
    const execution = operation();
    const paused = await Promise.race([execution.then(() => false), opened]);
    dialogWaiter = null;
    if (paused) pendingOperation = execution;
  }
  async function act(action) {
    validateAction(action);
    if (nativeDialog) {
      if (action.kind === 'input' && action.target === 'dialog-prompt' && dialog.type === 'prompt') promptValue = action.value;
      else if (action.kind === 'click' && (action.target === 'dialog-accept' || (action.target === 'dialog-dismiss' && dialog.type !== 'alert'))) {
        const handle = nativeDialog;
        if (action.target === 'dialog-accept') await handle.accept(promptValue); else await handle.dismiss();
        nativeDialog = null; dialog = null;
        if (pendingOperation) await pendingOperation;
        pendingOperation = null;
        await page.clock.runFor(20);
      } else throw fail('invalid_action', 'Only the displayed native dialog controls are available.');
    } else if (action.kind === 'wait') await page.clock.runFor(action.milliseconds);
    else if (action.target === 'file-chooser' && fileChooser) {
      if (action.kind !== 'input') throw fail('invalid_action', 'File chooser accepts one downloaded artifact ID.');
      const artifact = downloads.find(item => item.id === action.value);
      if (!artifact) throw fail('invalid_action', 'Only a file downloaded in this session can be selected.');
      await fileChooser.setFiles({ name: artifact.name, mimeType: 'application/json', buffer: artifact.buffer });
      fileChooser = null;
      await page.clock.runFor(20);
    } else {
      const control = registry.get(action.target);
      if (!control) throw fail('stale_target', 'Target was not in the last visible observation.');
      const locator = page.locator(control.locator);
      if (!await locator.isVisible() || control.disabled) throw fail('stale_target', 'Target is no longer visibly enabled.');
      // Refuse structural IDs that have been reassigned to another visible label.
      const fresh = await page.evaluate(visibleDocument);
      const current = fresh.controls.find(item => item.locator === control.locator);
      if (!current || current.tag !== control.tag || current.role !== control.role || current.name !== control.name || current.disabled) throw fail('stale_target', 'Target changed after the last observation.');
      if (action.kind === 'click') await perform(() => locator.click({ timeout: actionTimeout }));
      else if (action.kind === 'input') {
        if (control.role === 'file') {
          const artifact = downloads.find(item => item.id === action.value);
          if (!artifact) throw fail('invalid_action', 'Only a file downloaded in this session can be selected.');
          await locator.setInputFiles({ name: artifact.name, mimeType: 'application/json', buffer: artifact.buffer });
        } else await perform(() => locator.fill(action.value, { timeout: actionTimeout }));
      } else await perform(() => locator.selectOption({ value: action.value }, { timeout: actionTimeout }));
      if (!nativeDialog) await page.clock.runFor(20);
    }
    await new Promise(resolve => setTimeout(resolve, 25));
    tick += 1;
    return observe();
  }
  return { observe, act, close: () => browser.close(), manifest: { protocol: PROTOCOL, browser: await browser.version(), viewport: { width:1440,height:1000 }, clock: { mode: 'controlled-native-timers', epoch: CLOCK_EPOCH, atomicSettleMs: 20, startupSettleMs: 1000 }, randomness: { mode: 'seeded-math-random-and-randomUUID', seed: randomSeed }, realEventSettleMs:25, nativeDialogPolicy:'model-chosen-controls', originPolicy: 'single-loopback-origin', storage: 'fresh-browser-context' } };
}

async function main() {
  const args = process.argv.slice(2); const value = name => args[args.indexOf(name) + 1];
  let instance;
  try { instance = await createBrowser({ url: value('--url'), playwrightModule: args.includes('--playwright-module') ? value('--playwright-module') : undefined, executablePath: args.includes('--executable') ? value('--executable') : undefined }); }
  catch { process.stderr.write('Local browser initialization failed.\n'); process.exitCode = 1; return; }
  const lines = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
  try {
    for await (const line of lines) {
      let request;
      try {
        if (Buffer.byteLength(line) > 1024 * 1024) throw fail('message_too_large', 'Request exceeds 1 MiB.');
        request = JSON.parse(line);
        if (!request || typeof request.id !== 'string' || !request.id) throw fail('invalid_request', 'Request ID is required.');
        let response;
        if (request.type === 'hello') response = { protocol: PROTOCOL, manifest: instance.manifest };
        else if (request.type === 'observe') response = { observation: await instance.observe() };
        else if (request.type === 'act') response = { observation: await instance.act(request.action) };
        else if (request.type === 'close') response = {};
        else throw fail('invalid_request', 'Unsupported browser request.');
        process.stdout.write(JSON.stringify({ id: request.id, type: request.type, ok: true, ...response }) + '\n');
        if (request.type === 'close') break;
      } catch (error) {
        // Never echo arbitrary browser errors, URLs, filesystem paths or source.
        process.stdout.write(JSON.stringify({ id: request?.id ?? null, type: 'error', ok: false, error: { code: error.code || 'browser_action_failed', message: error.code ? error.message : 'The browser operation failed; no action was retried.' } }) + '\n');
      }
    }
  } finally { await instance.close(); }
}
if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(new URL(import.meta.url).pathname)) await main();
