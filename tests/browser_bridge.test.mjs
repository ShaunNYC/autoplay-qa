import test from 'node:test';
import assert from 'node:assert/strict';
import { validateAction, validateUrl } from '../tools/browser_bridge.mjs';

test('browser URL confines execution to loopback and rejects credentials', () => {
  assert.equal(validateUrl('http://127.0.0.1:3210/#/mars').hostname, '127.0.0.1');
  for (const value of ['https://example.com','file:///etc/passwd','http://name:secret@localhost','http://localhost.evil.test','http://127.0.0.1.evil.test']) assert.throws(() => validateUrl(value));
});
test('atomic control schema rejects macros, arbitrary locators and extras', () => {
  for (const value of [{kind:'reset'},{kind:'click',target:'c-a',route:[]},{kind:'click',selector:'body'},{kind:'input',target:'c-a',value:[]},{kind:'wait',milliseconds:10001},{kind:'wait',milliseconds:NaN}]) assert.throws(() => validateAction(value));
  for (const value of [{kind:'click',target:'c-a'},{kind:'input',target:'c-a',value:'x'},{kind:'select',target:'c-a',value:'v'},{kind:'wait',milliseconds:0}]) assert.deepEqual(validateAction(value),value);
});
