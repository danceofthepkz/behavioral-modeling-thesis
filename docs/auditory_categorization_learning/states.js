'use strict';
(async function stateViewer(){
 const find=id=>document.getElementById(id),mouse=find('state-mouse'),session=find('state-session'),slider=find('state-trial'),summary=find('state-summary');
 try{
  const response=await fetch('states.json');if(!response.ok)throw Error('State sequences could not load. Please reload.');
  const dataset=await response.json(),colors=['#440154','#21918c','#c5aa00'];let current,boundaries=[],start=0,end=0,scaleX,scaleY,cursor,selected;
  mouse.replaceChildren(...Object.keys(dataset.mice).map(id=>new Option(id,id)));mouse.value='GS029';mouse.disabled=false;session.disabled=false;slider.disabled=false;
  function inspect(value){
   const trial=Math.max(start,Math.min(end,Math.round(value))),state=current.states[trial],sessionIndex=d3.bisectRight(boundaries,trial);
   slider.value=trial;find('state-trial-value').textContent=trial.toLocaleString();
   find('state-detail').textContent=`Trial ${trial.toLocaleString()} · Session ${sessionIndex+1} · State ${state} · Posterior probability ${(100*current.confidence[trial]).toFixed(1)}%`;
   cursor.attr('x1',scaleX(trial)).attr('x2',scaleX(trial));selected.attr('cx',scaleX(trial)).attr('cy',scaleY(state)).attr('stroke',colors[state-1]);
  }
  function draw(){
   const which=session.value,all=which==='all',i=all?0:Number(which);start=all?0:(i?boundaries[i-1]:0);end=all?current.n-1:(boundaries[i]??current.n)-1;
   slider.min=start;slider.max=end;
   summary.textContent=all?`${current.n.toLocaleString()} trials · ${current.sessions.length} sessions`:`${(end-start+1).toLocaleString()} trials · Session ${i+1} of ${current.sessions.length}`;
   find('state-title').textContent=`${current.mouse}: inferred state sequence`;
   const el=find('state-chart'),w=Math.max(260,el.clientWidth),h=el.clientHeight,m={l:65,r:18,t:24,b:48},svg=d3.select(el);
   svg.selectAll('*').remove();svg.attr('viewBox',`0 0 ${w} ${h}`).attr('aria-label',`${current.mouse}: inferred states for trials ${start}–${end}`);
   svg.append('title').text(`${current.mouse}: inferred GLM-HMM states`);svg.append('desc').text(`Trials ${start} to ${end}. State 1, 2 or 3 is the most probable state on each trial. Use Inspect trial to read individual values.`);
   scaleX=d3.scaleLinear().domain([start-.5,end+.5]).range([m.l,w-m.r]);scaleY=d3.scaleLinear().domain([.7,3.3]).range([h-m.b,m.t]);
   svg.append('g').attr('transform',`translate(${m.l},0)`).call(d3.axisLeft(scaleY).tickValues([1,2,3]).tickFormat(d3.format('d')).tickSizeOuter(0));
   svg.append('g').attr('transform',`translate(0,${h-m.b})`).call(d3.axisBottom(scaleX).ticks(w<500?4:8).tickFormat(d3.format(',d')).tickSizeOuter(0));
   svg.selectAll('.domain,.tick line').attr('stroke','#b8c3c9');svg.selectAll('text').attr('font-family','Arial').attr('font-size',12).attr('fill','#64737c');
   svg.append('g').selectAll('line').data(boundaries.filter(t=>t>start&&t<=end)).join('line').attr('class','session-boundary').attr('x1',t=>scaleX(t-.5)).attr('x2',t=>scaleX(t-.5)).attr('y1',m.t).attr('y2',h-m.b).attr('stroke','#dce3e7').attr('stroke-width',1);
   svg.append('g').selectAll('circle').data(d3.range(start,end+1)).join('circle').attr('class','state-point').attr('data-trial',t=>t).attr('data-state',t=>current.states[t]).attr('cx',t=>scaleX(t)).attr('cy',t=>scaleY(current.states[t])).attr('r',1.65).attr('fill',t=>colors[current.states[t]-1]);
   svg.append('text').attr('x',(m.l+w-m.r)/2).attr('y',h-9).attr('text-anchor','middle').attr('font-size',12).attr('fill','#64737c').text('Training trial');
   svg.append('text').attr('transform',`translate(16,${(m.t+h-m.b)/2}) rotate(-90)`).attr('text-anchor','middle').attr('font-size',12).attr('fill','#64737c').text('Most likely state');
   cursor=svg.append('line').attr('class','state-cursor').attr('y1',m.t).attr('y2',h-m.b).attr('stroke','#64737c').attr('stroke-dasharray','3 3');
   selected=svg.append('circle').attr('class','state-cursor').attr('r',5).attr('fill','white').attr('stroke-width',2);
   svg.append('rect').attr('x',m.l).attr('y',m.t).attr('width',w-m.l-m.r).attr('height',h-m.t-m.b).attr('fill','transparent').on('pointermove',event=>{if(event.pointerType!=='touch')inspect(scaleX.invert(d3.pointer(event,el)[0]));}).on('click',event=>inspect(scaleX.invert(d3.pointer(event,el)[0])));
   inspect(Number(slider.value));
  }
  function chooseMouse(){
   current=dataset.mice[mouse.value];boundaries=d3.cumsum(current.sessions).slice(0,-1);
   session.replaceChildren(new Option('All sessions','all'),...current.sessions.map((_,i)=>new Option(`Session ${i+1}`,i)));slider.min=0;slider.max=current.n-1;slider.value=0;
   find('state-fit').textContent=current.mouse==='GS029'?'GS029 reproduces the individual notebook figure: the saved winning initialization (seed 301), 100 EM iterations, selected from 10 starts. Its likelihood, weights and occupancy match the saved notebook output at the reported precision.':'This sequence reproduces the saved 19-mouse batch fit: 75 EM iterations, using its recorded winning initialization from 3 starts. Likelihood, weights, transition probabilities and occupancy were checked against the saved results (absolute tolerance 0.000001).';
   draw();
  }
  mouse.onchange=chooseMouse;session.onchange=()=>{slider.min=0;slider.value=0;draw();};slider.oninput=()=>inspect(Number(slider.value));
  let resize;window.addEventListener('resize',()=>{clearTimeout(resize);resize=setTimeout(draw,120);});chooseMouse();
 }catch(error){summary.textContent=error.message;summary.setAttribute('role','alert');console.error(error);}
})();
