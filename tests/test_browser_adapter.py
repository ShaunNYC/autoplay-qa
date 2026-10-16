"""Portable offline browser checks: real Chromium, local public HTML, no model.

Set AUTOPLAY_PLAYWRIGHT_MODULE for an external Playwright installation, or
install it in web/node_modules. Set AUTOPLAY_CHROME for another browser path.
Native checks skip when these optional dependencies are absent.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from autoplay_qa.browser_adapter import BrowserAdapter, LocalBrowserSite, validate_action
from autoplay_qa.protocol import ProtocolError

ROOT = Path(__file__).resolve().parents[1]
PLAYWRIGHT = Path(os.environ.get('AUTOPLAY_PLAYWRIGHT_MODULE', ROOT / 'web/node_modules/playwright/index.mjs'))
CHROME = Path(os.environ.get('AUTOPLAY_CHROME', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'))
HTML = '''<!doctype html><title>Offline visible fixture</title>
<p>Public help: enter a name, choose a size, save and restore.</p>
<p style="display:none">HIDDEN-DEVELOPER-SECRET</p><details><summary>Collapsed help</summary><button>Hidden collapsed control</button></details>
<label>Name<input id="name" value="initial"></label>
<label>Size<select id="size"><option value="small">Small</option><option value="large">Large</option></select></label>
<button id="save">Save session</button><button id="restore">Restore session</button>
<input id="upload" type="file" hidden><button disabled>Unavailable</button><button id="delayed">Start clock</button><button id="confirm">Confirm operation</button><p id="result"></p>
<script>
window.PRIVATE_GAME_STATE='PRIVATE-SECRET';
const nameInput=document.getElementById('name'),sizeInput=document.getElementById('size'),result=document.getElementById('result');
document.getElementById('save').onclick=()=>{const a=document.createElement('a');a.download='session.json';a.href=URL.createObjectURL(new Blob([JSON.stringify({name:nameInput.value,size:sizeInput.value})],{type:'application/json'}));a.click()};
document.getElementById('restore').onclick=()=>document.getElementById('upload').click();
document.getElementById('upload').onchange=async event=>{const data=JSON.parse(await event.target.files[0].text());nameInput.value=data.name;sizeInput.value=data.size;result.textContent='Restored';};
document.getElementById('confirm').onclick=()=>{result.textContent=confirm('Proceed?')?'Confirmed':'Canceled';};
document.getElementById('delayed').onclick=()=>setTimeout(()=>result.textContent='Timer complete',1000);
</script>'''

class BrowserSchemaTests(unittest.TestCase):
    def test_atomic_schema_rejects_unknown_extra_and_unbounded_controls(self):
        for action in [{'kind':'macro','target':'x'},{'kind':'wait','milliseconds':True},{'kind':'wait','milliseconds':-1},{'kind':'wait','milliseconds':10001},{'kind':'click','target':'a','script':'x'},{'kind':'input','target':'x','value':'a'*4001},{'kind':[]}]:
            with self.subTest(action=str(action)[:90]), self.assertRaises(ValueError):
                validate_action(action)
    def test_url_rejected_before_browser_launch(self):
        for url in ['https://example.com','file:///tmp/a','http://user:password@localhost','http://localhost.attacker.test']:
            with self.assertRaises(ValueError): BrowserAdapter(url)
    def test_wait_validation_copies_input(self):
        action={'kind':'wait','milliseconds':10}
        self.assertEqual(validate_action(action),action)
        self.assertIsNot(validate_action(action),action)

@unittest.skipUnless(PLAYWRIGHT.is_file() and CHROME.is_file(), 'Configured local Playwright/Chromium unavailable')
class NativeBrowserTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='browser-adapter-')
        self.addCleanup(self.temp.cleanup)
        Path(self.temp.name,'index.html').write_text(HTML)
        self.site=LocalBrowserSite(self.temp.name)
        self.addCleanup(self.site.close)
    def browser(self):
        value=BrowserAdapter(self.site.url,playwright_module=PLAYWRIGHT,executable_path=CHROME)
        self.addCleanup(value.close)
        return value
    def control(self,observation,name):
        return next(item['id'] for item in observation['controls'] if item['name']==name)
    def test_real_input_select_download_restore_and_fresh_replay(self):
        browser=self.browser()
        initial=browser.observe()
        encoded=json.dumps(initial)
        self.assertNotIn('HIDDEN-DEVELOPER-SECRET',encoded)
        self.assertNotIn('PRIVATE-SECRET',encoded)
        self.assertNotIn('upload',encoded)
        self.assertFalse(any(item['name']=='Hidden collapsed control' for item in initial['controls']))
        self.assertEqual(browser.source_manifest['backend'],'native-playwright-browser')
        self.assertEqual(browser.source_manifest['browser_transport']['clock']['mode'],'controlled-native-timers')
        steps=[]
        def act(action):
            after=browser.act(action);steps.append((deepcopy(action),deepcopy(after)));return after
        after=act({'kind':'input','target':self.control(initial,'Name'),'value':'fixture saved value'})
        after=act({'kind':'select','target':self.control(after,'Size'),'value':'large'})
        after=act({'kind':'click','target':self.control(after,'Save session')})
        self.assertEqual(after['downloads'][0]['name'],'session.json')
        self.assertGreater(after['downloads'][0]['bytes'],0)
        after=act({'kind':'input','target':self.control(after,'Name'),'value':'changed'})
        after=act({'kind':'click','target':self.control(after,'Restore session')})
        self.assertIn('file-chooser',[item['id'] for item in after['controls']])
        after=act({'kind':'input','target':'file-chooser','value':after['downloads'][0]['id']})
        after=act({'kind':'wait','milliseconds':20})
        self.assertIn('Restored',after['text'])
        self.assertEqual(next(item['value'] for item in after['controls'] if item['name']=='Name'),'fixture saved value')
        after=act({'kind':'click','target':self.control(after,'Start clock')})
        self.assertNotIn('Timer complete',after['text'])
        after=act({'kind':'wait','milliseconds':1000})
        self.assertIn('Timer complete',after['text'])
        browser.close()
        fresh=self.browser()
        self.assertNotEqual(fresh.source_manifest['instance_id'],browser.source_manifest['instance_id'])
        self.assertEqual(fresh.observe(),initial)
        for action,expected in steps:
            self.assertEqual(fresh.act(action),expected)
    def test_unobserved_target_stops_without_retry_and_closes(self):
        browser=self.browser()
        with self.assertRaises(ProtocolError) as caught: browser.act({'kind':'click','target':'invented'})
        self.assertEqual(caught.exception.code,'stale_target')
        with self.assertRaises(ProtocolError) as caught: browser.observe()
        self.assertEqual(caught.exception.code,'closed')
    def test_file_selection_cannot_read_local_path(self):
        browser=self.browser();observation=browser.observe()
        browser.act({'kind':'click','target':self.control(observation,'Restore session')})
        with self.assertRaises(ProtocolError) as caught: browser.act({'kind':'input','target':'file-chooser','value':'/etc/passwd'})
        self.assertEqual(caught.exception.code,'invalid_action')
    def test_native_dialog_requires_separate_model_control(self):
        browser=self.browser();observation=browser.observe()
        after=browser.act({'kind':'click','target':self.control(observation,'Confirm operation')})
        self.assertEqual(after['dialog'],{'type':'confirm','message':'Proceed?'})
        self.assertEqual(after['tick'],1)
        self.assertEqual([item['name'] for item in after['controls']],['OK','Cancel'])
        self.assertNotIn('Confirmed',after['text'])
        after=browser.act({'kind':'click','target':'dialog-accept'})
        self.assertEqual(after['tick'],2)
        self.assertIsNone(after['dialog'])
        self.assertIn('Confirmed',after['text'])
    def test_returned_observation_is_detached(self):
        browser=self.browser();observation=browser.observe();observation['controls'].clear()
        self.assertTrue(browser.observe()['controls'])

if __name__=='__main__': unittest.main()
