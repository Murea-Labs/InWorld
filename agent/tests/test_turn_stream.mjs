import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import vm from 'node:vm'
import ts from '../frontend/node_modules/typescript/lib/typescript.js'
const source = readFileSync(new URL('../frontend/src/api.ts', import.meta.url), 'utf8')
  .replace('import.meta.env.BASE_URL', "'/'")
const code = ts.transpileModule(source, { compilerOptions: {
  module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022,
} }).outputText
function setup(fetch) {
  // api.ts imports t() from ./i18n; strings pass through untranslated here.
  const require = (m) => { if (m === './i18n') return { t: (s) => s }; throw new Error(m) }
  const context = { exports:{}, require, fetch, TextDecoder, Error, JSON,
    crypto:{ randomUUID:()=> 'request-test' }, setTimeout:fn=>{fn();return 0} }
  vm.runInNewContext(code, context)
  return context.exports
}
const done = {type:'done',reply:'已抵达桥头',node:{id:'n1'},world:{id:'w1'}}
function stream(chunks) {
  return new Response(new ReadableStream({start(c) {
    for (const chunk of chunks) c.enqueue(new TextEncoder().encode(chunk))
    c.close()
  }}))
}
test('a fragmented stream with heartbeats delivers done without waiting for EOF', async()=>{
  const events=[]
  const api=setup(async()=>stream([': keep-alive\r','\n\r\n','data: '+JSON.stringify(done)+'\n','\n']))
  await api.streamTurn('test',{},e=>events.push(e))
  assert.equal(events.length,1)
  assert.equal(events[0].node.id,'n1')
})
test('lost POST response retrieves exactly its result without resubmitting generation', async()=>{
  const requests=[],events=[]
  const api=setup(async(url,init)=>{
    requests.push({url,init})
    if(init.method==='POST') throw new TypeError('Failed to fetch')
    return Response.json({status:'done',event:done})
  })
  await api.turn('w1','过桥',e=>events.push(e))
  assert.equal(requests.filter(r=>r.init.method==='POST').length,1)
  assert.equal(JSON.parse(requests[0].init.body).request_id,'request-test')
  assert.match(requests[1].url,/turn-status\/request-test$/)
  assert.equal(events.at(-1).type,'done')
})
test('EOF without terminal event polls the running job and recovers',async()=>{
  let gets=0,posts=0
  const api=setup(async(url,init)=>{
    if(init.method==='POST'){posts++;return stream(['data: {"type":"say","text":"过桥"}\n\n'])}
    return Response.json(++gets===1?{status:'running'}:{status:'done',event:done})
  })
  const events=[]
  await api.turn('w1','过桥',e=>events.push(e))
  assert.equal(posts,1);assert.equal(gets,2);assert.equal(events.at(-1).type,'done')
})
test('unreceived request fails clearly and is never automatically resubmitted',async()=>{
  let posts=0
  const api=setup(async(url,init)=>{
    if(init.method==='POST'){posts++;throw new TypeError('Failed to fetch')}
    return Response.json({status:'unknown'})
  })
  await assert.rejects(api.turn('w1','过桥',()=>{}),/无法确认本次请求状态/)
  assert.equal(posts,1)
})
test('model error is terminal and never mistaken for a transport disconnect',async()=>{
  let calls=0;const events=[]
  const api=setup(async()=>{calls++;return stream(['data: {"type":"error","message":"video error"}\n\n'])})
  await api.turn('w1','过桥',e=>events.push(e))
  assert.equal(calls,1);assert.equal(events[0].type,'error')
})
test('user abort never triggers recovery',async()=>{
  const controller=new AbortController();controller.abort()
  let calls=0
  const api=setup(async()=>{calls++;throw Object.assign(new Error('cancelled'),{name:'AbortError'})})
  await assert.rejects(api.turn('w1','过桥',()=>{},false,controller.signal),{name:'AbortError'})
  assert.equal(calls,1)
})
test('lost error event recovers the original model error without resubmitting',async()=>{
  let posts=0
  const events=[]
  const failure={type:'error',message:'GeminiError: Expecting value: line 1 column 1 (char 0)'}
  const api=setup(async(url,init)=>{
    if(init.method==='POST'){posts++;return stream([])}
    return Response.json({status:'failed',event:failure})
  })
  await api.turn('w1','过桥',e=>events.push(e))
  assert.equal(posts,1)
  assert.equal(events.at(-1).type,'error')
  assert.equal(events.at(-1).message,failure.message)
})
test('lost cancellation event is recovered as cancellation, not a network error',async()=>{
  let posts=0
  const events=[]
  const api=setup(async(url,init)=>{
    if(init.method==='POST'){posts++;throw new TypeError('Failed to fetch')}
    return Response.json({status:'cancelled',event:{type:'cancelled'}})
  })
  await api.turn('w1','过桥',e=>events.push(e))
  assert.equal(posts,1)
  assert.equal(events.at(-1).type,'cancelled')
})
