// Run with PLAYWRIGHT_MODULE pointing to an installed Playwright package.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs'), assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});
 try {
  const page=await browser.newPage(), calls=[], receipts=new Map(), errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.route('http://localhost:9876/**',async route=>{
   const url=new URL(route.request().url()), path=url.pathname;
   const json=data=>route.fulfill({json:data});
   if(path==='/')return route.fulfill({contentType:'text/html',body:fs.readFileSync('core/platform/playground.html','utf8')});
   if(path==='/activity.js')return route.fulfill({contentType:'text/javascript',body:fs.readFileSync('core/platform/activity.js','utf8')});
   if(path==='/playground/config')return json({model:'test',userId:'legacy-researcher-owner',tenants:[{id:'a',key:'a'},{id:'b',key:'b'}]});
   if(path==='/v1/usage/summary')return json({totals:{calls:0,knownInputTokens:0,knownOutputTokens:0}});
   if(path==='/v1/usage/events')return json({items:[]});
   if(path==='/v1/turns'){
    const body=route.request().postDataJSON();calls.push(body);
    const result={tenantId:body.tenantId,requestId:body.requestId,sessionId:body.sessionId,content:body.message.startsWith('Draft a concise')?'Trade SPY using a 20-bar momentum signal. No future data.':'Discussion response',proposal:null,validation:null,evidence:body.evidence};
    if(body.action==='research')result.evidence=[{id:'paper',title:'Paper',text:'Evidence'}];
    if(body.action==='build')result.proposal={baseDraftRevision:body.component.revision,baseContentHash:body.component.contentHash,explanation:'Updated',changes:[{before:body.component.sourceCode,after:'class Example: pass'}]};
    if(['build','validate'].includes(body.action))result.validation={target:'draft',revision:body.component.revision,contentHash:body.component.contentHash,checks:[],diagnostics:[]};
    receipts.set(body.requestId,{status:'succeeded',stage:body.action,progress:[],result});return json(result);
   }
   const id=path.split('/')[3];
   if(path.endsWith('/events'))return route.fulfill({contentType:'text/event-stream',body:'event: complete\ndata: '+JSON.stringify({tenantId:'a',requestId:id,status:'succeeded'})+'\n\n'});
   if(receipts.has(id))return json(receipts.get(id));
   return route.fulfill({status:404,json:{error:{message:'Not found'}}});
  });
  // Mocked localhost keeps Web Crypto available without a real provider call.
  await page.goto('http://localhost:9876/');
  async function send(id,message){if(message!==undefined)await page.locator('#message').fill(message);await page.locator('#'+id).click();await page.waitForFunction(()=>!document.querySelector('#send').disabled);}
  await send('send','Discuss momentum');assert.equal(calls.at(-1).action,'chat');
  await page.locator('#query').fill('momentum');await send('research','Find evidence');assert.equal(calls.at(-1).action,'research');
  await send('send','What are the limitations?');assert.equal(calls.at(-1).action,'chat');assert.equal(calls.at(-1).evidence[0].id,'paper');
  await send('summarize');assert.match(await page.locator('#spec').inputValue(),/20-bar/);
  const count=calls.length;await page.locator('#message').fill('Pending decision');await page.locator('#build').click();assert.equal(calls.length,count);assert.match(await page.locator('#status').innerText(),/pending message/);
  await page.locator('#message').fill('');await send('build');assert.equal(calls.at(-1).action,'build');assert.match(calls.at(-1).message,/20-bar/);
  await send('send','Explain the change');assert.equal(calls.at(-1).action,'chat');assert.equal(await page.locator('#proposal').isVisible(),true);
  await page.locator('#apply').click();await send('revise','Change lookback to 30');assert.equal(calls.at(-1).action,'build');
  await send('validate');assert.equal(calls.at(-1).action,'validate');
  await page.locator('#tenant').selectOption('b');assert.equal(await page.locator('#spec').inputValue(),'');
  await page.locator('#tenant').selectOption('a');assert.match(await page.locator('#spec').inputValue(),/20-bar/);
  await page.reload();await page.waitForFunction(()=>!document.querySelector('#send').disabled);assert.match(await page.locator('#spec').inputValue(),/20-bar/);
  assert.deepEqual(errors,[]);console.log('Discussion, research, specification, confirmation, revision, validation, persistence and tenant isolation passed.');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
