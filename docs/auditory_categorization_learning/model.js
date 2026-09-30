/* Direct port of simulateLearning and generateLearningCurves.
 * The random stream is exported from Python's random.Random(2021).
 * Preserve the source's extra stimulus draws even when stimuli are supplied.
 * The source computes N-1 trials; its uncomputed final array entry is excluded.
 */
(function(global){
 function simulate(cues,parameters,uniforms){
  let q=[parameters.init,-parameters.init],draw=0,correctionCount=0;const rows=[];
  for(let t=0;t<cues.length-1;t++){
   const cue=Number(cues[t]),p=1/(1+Math.exp(-5*((cue?q[1]:-q[0])+2*parameters.bias)));
   if(draw>=uniforms.length)throw Error('Reference random stream exhausted');
   const choice=Number(uniforms[draw++]>=1-p),reward=Number(choice===cue),before=q.slice(),delta=reward-q[choice];
   if(reward){draw++;correctionCount=0;}else{correctionCount++;if(correctionCount>3)draw++;}
   q[choice]+=parameters.alpha*delta;q[1-choice]-=parameters.coupling*parameters.alpha*delta;
   rows.push({cue,p,choice,reward,before,after:q.slice(),delta});
  }
  return rows;
 }
 function categoryCurves(cues,choices){
  const sums=[0,0],counts=[0,0],low=[],high=[];
  for(let t=0;t<choices.length;t++){
   const c=Number(cues[t]);sums[c]+=Number(choices[t]);counts[c]++;
   if(t>=200){const old=Number(cues[t-200]);sums[old]-=Number(choices[t-200]);counts[old]--;}
   low.push(counts[0]>=5?sums[0]/counts[0]:null);high.push(counts[1]>=5?sums[1]/counts[1]:null);
  }
  return {low,high};
 }
 const api={simulate,categoryCurves};if(typeof module!=='undefined'&&module.exports)module.exports=api;else global.AuditoryModel=api;
})(typeof window!=='undefined'?window:globalThis);
