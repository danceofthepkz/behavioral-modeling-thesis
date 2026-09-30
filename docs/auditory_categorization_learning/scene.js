/* Visual apparatus only. Choice, reward and learning are supplied by model.js. */
window.createMouseScene=function(){
 const host=document.querySelector('#mp-scene');let row={cue:0,choice:0,reward:0},phase=0,draw=()=>{};
 try{
 const T=window.THREE,scene=new T.Scene(),renderer=new T.WebGLRenderer({antialias:true});
 scene.background=new T.Color('#eef1f2');renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.shadowMap.enabled=true;renderer.shadowMap.type=T.PCFSoftShadowMap;renderer.outputColorSpace=T.SRGBColorSpace;renderer.toneMapping=T.ACESFilmicToneMapping;renderer.toneMappingExposure=1.3;host.prepend(renderer.domElement);renderer.domElement.setAttribute('role','img');renderer.domElement.setAttribute('aria-label','Head-fixed mouse with forepaw response wheel, lick spout and water reward');
 const camera=new T.PerspectiveCamera(33,1,.05,60);scene.add(new T.HemisphereLight(0xf6faff,0x555966,2));
 const key=new T.DirectionalLight(0xfff3e4,3.2);key.position.set(1,6,4);key.castShadow=true;key.shadow.mapSize.set(2048,2048);Object.assign(key.shadow.camera,{left:-4,right:4,top:4,bottom:-4});key.shadow.normalBias=.015;scene.add(key);const rim=new T.DirectionalLight(0xd4e6ff,2);rim.position.set(-3,3,-3);scene.add(rim);
 const mat=(color,roughness=.8,metalness=0)=>new T.MeshStandardMaterial({color,roughness,metalness});
 const fur=mat('#45413c'),headFur=mat('#514b43'),skin=mat('#ba918a',.7),innerEar=mat('#ae8781',.8),noseMat=mat('#aa8280',.5),black=mat('#090b0c',.12),steel=mat('#afbec6',.28,.8),dark=mat('#293941',.6),base=mat('#d8dfe2',.65);
 function mesh(g,m,parent=scene){const o=new T.Mesh(g,m);o.castShadow=true;o.receiveShadow=true;parent.add(o);return o;}
 function oval(parent,m,p,s){const o=mesh(new T.SphereGeometry(1,48,32),m,parent);o.position.set(...p);o.scale.set(...s);return o;}
 function box(parent,m,p,s){const o=mesh(new T.BoxGeometry(...s),m,parent);o.position.set(...p);return o;}
 function rod(parent,a,b,r,m){const av=new T.Vector3(...a),bv=new T.Vector3(...b),d=bv.clone().sub(av),o=mesh(new T.CylinderGeometry(r,r,d.length(),14),m,parent);o.position.copy(av.add(bv).multiplyScalar(.5));o.quaternion.setFromUnitVectors(new T.Vector3(0,1,0),d.normalize());return o;}
 function tube(parent,points,r,m){return mesh(new T.TubeGeometry(new T.CatmullRomCurve3(points.map(p=>new T.Vector3(...p))),64,r,10,false),m,parent);}
 const floor=mesh(new T.PlaneGeometry(30,30),mat('#eef1f2'));floor.rotation.x=-Math.PI/2;floor.position.y=-.1;box(scene,base,[-.35,-.035,0],[4.9,.12,2.7]);
 // Optical-table fixing points and a narrow body support.
 for(let x=-2.5;x<2;x+=.35)for(let z=-1.05;z<1.1;z+=.35){const screw=mesh(new T.CylinderGeometry(.021,.021,.006,8),steel);screw.position.set(x,.029,z);}
 box(scene,mat('#aeb9bf'),[-.7,.075,0],[2.1,.09,.85]);
 const mouse=new T.Group();scene.add(mouse);
 oval(mouse,fur,[-.72,.60,0],[.84,.52,.39]);oval(mouse,fur,[-.10,.62,0],[.49,.38,.30]);
 // Continuous tapered head: longitudinal cross sections avoid a spherical cartoon muzzle.
 const sections=[[-.08,.69,.25,.27],[.12,.75,.30,.28],[.35,.73,.27,.245],[.58,.67,.20,.18],[.78,.60,.115,.12],[.95,.565,.052,.058],[1.00,.56,.022,.031]],vertices=[],indices=[];
 sections.forEach(([x,y,ry,rz])=>{for(let j=0;j<48;j++){const a=j/48*Math.PI*2;vertices.push(x,y+Math.cos(a)*ry,Math.sin(a)*rz);}});
 for(let i=0;i<sections.length-1;i++)for(let j=0;j<48;j++){const a=i*48+j,b=i*48+(j+1)%48;indices.push(a,b,a+48,b,b+48,a+48);}const hg=new T.BufferGeometry();hg.setAttribute('position',new T.Float32BufferAttribute(vertices,3));hg.setIndex(indices);hg.computeVertexNormals();mesh(hg,headFur,mouse);
 oval(mouse,noseMat,[.997,.562,0],[.033,.028,.047]);const jaw=oval(mouse,headFur,[.72,.506,0],[.235,.057,.107]);
 const paws=[];
 for(const s of [-1,1]){
  const ear=oval(mouse,fur,[.09,1.005,s*.245],[.163,.215,.047]);ear.rotation.x=-s*.24;ear.rotation.z=-.20;
  const inner=oval(mouse,innerEar,[.107,1.01,s*.274],[.142,.189,.013]);inner.rotation.x=-s*.24;inner.rotation.z=-.20;
  const eye=oval(mouse,black,[.48,.805,s*.193],[.049,.054,.027]);eye.rotation.y=s*.18;oval(mouse,mat('#ffffff',.2),[.492,.824,s*.216],[.010,.009,.004]);
  oval(mouse,fur,[-1.08,.33,s*.28],[.36,.30,.24]);oval(mouse,skin,[-.91,.13,s*.39],[.22,.046,.068]);
  rod(mouse,[-.10,.58,s*.26],[.16,.33,s*.27],.066,fur);rod(mouse,[.16,.33,s*.27],[.42,.40,s*.23],.042,headFur);
  const paw=new T.Group();paw.position.set(.43,.39,s*.23);mouse.add(paw);oval(paw,skin,[0,0,0],[.105,.046,.060]);
  for(let i=0;i<4;i++){const z=(i-1.5)*.025;const toe=tube(paw,[[.015,-.006,z],[.09,-.02,z],[.12,-.052,z]],.011,skin);oval(paw,mat('#e4c9bd'),[.124,-.054,z],[.018,.006,.006]);}paws.push(paw);
  for(let k=0;k<7;k++){const offset=k-3;tube(mouse,[[.86,.568+offset*.009,s*.077],[.93+offset*.029,.59+offset*.027,s*.26],[.94+offset*.067,.61+offset*.046,s*(.43+Math.abs(offset)*.018)]],.0018,mat('#b8b5b0'));}
 }
 // Tapered, ringed tail follows the table rather than floating in space.
 const tailCurve=new T.CatmullRomCurve3([[-1.32,.32,0],[-1.62,.16,.11],[-2.06,.085,.24],[-2.51,.08,.20],[-2.79,.08,-.10]].map(p=>new T.Vector3(...p)));
 const tailGeometry=new T.TubeGeometry(tailCurve,96,.041,10,false),tailPos=tailGeometry.attributes.position;
 for(let i=0;i<=96;i++){const c=tailCurve.getPointAt(i/96),scale=1-.92*i/96;for(let j=0;j<=10;j++){const k=i*11+j;tailPos.setXYZ(k,c.x+(tailPos.getX(k)-c.x)*scale,c.y+(tailPos.getY(k)-c.y)*scale,c.z+(tailPos.getZ(k)-c.z)*scale);}}tailGeometry.computeVertexNormals();mesh(tailGeometry,skin,mouse);
 // Short direction-following strands over body, shoulders and head.
 let seed=81;const rand=()=>{seed=(1664525*seed+1013904223)>>>0;return seed/4294967296;};const strands=[],colors=[];
 for(const [cx,cy,cz,rx,ry,rz,count] of [[-.72,.60,0,.842,.522,.392,15000],[-.10,.62,0,.492,.382,.302,4500],[-1.08,.33,-.28,.362,.302,.242,2600],[-1.08,.33,.28,.362,.302,.242,2600]]){
  for(let i=0;i<count;i++){const v=rand()*2-1,a=rand()*Math.PI*2,b=Math.sqrt(1-v*v),x=cx+rx*b*Math.cos(a),y=cy+ry*v,z=cz+rz*b*Math.sin(a);if(y<.23)continue;const len=.007+rand()*.015;strands.push(x,y,z,x-len,y+len*.2,z+Math.sign(z)*len*.18);const shade=.035+rand()*.09;for(let k=0;k<2;k++)colors.push(shade*1.06,shade,shade*.88);}
 }
 for(let i=0;i<7000;i++){const u=rand()*(sections.length-1),n=Math.min(5,Math.floor(u)),t=u-n,a=rand()*Math.PI*2,p=sections[n].map((v,j)=>v*(1-t)+sections[n+1][j]*t),[x,y,ry,rz]=p;const yy=y+Math.cos(a)*(ry+.003),zz=Math.sin(a)*(rz+.003);if(yy<.50)continue;strands.push(x,yy,zz,x-.018,yy+.007,zz*1.022);const c=.04+rand()*.10;for(let k=0;k<2;k++)colors.push(c*1.08,c,c*.88);}
 const sg=new T.BufferGeometry();sg.setAttribute('position',new T.Float32BufferAttribute(strands,3));sg.setAttribute('color',new T.Float32BufferAttribute(colors,3));mouse.add(new T.LineSegments(sg,new T.LineBasicMaterial({vertexColors:true,transparent:true,opacity:.48})));
 // Forepaw wheel. Counterbalanced stimulus-to-turn mapping is illustrated, not inferred.
 const wheel=new T.Group();wheel.position.set(.45,.23,0);scene.add(wheel);const disk=mesh(new T.CylinderGeometry(.23,.23,.40,64),dark,wheel);disk.rotation.x=Math.PI/2;
 for(let i=0;i<36;i++){const a=i/36*Math.PI*2;rod(wheel,[.231*Math.cos(a),.231*Math.sin(a),-.2],[.231*Math.cos(a),.231*Math.sin(a),.2],.005,steel);}rod(scene,[.45,.23,-.48],[.45,.23,.48],.024,steel);for(const s of [-1,1])box(scene,steel,[.45,.14,s*.44],[.13,.22,.09]);
 // Headplate and lateral support posts keep the head stationary.
 box(scene,steel,[.055,.965,0],[.12,.022,.72]);for(const s of [-1,1]){rod(scene,[.055,.965,s*.36],[.055,.965,s*.69],.025,steel);rod(scene,[.055,.965,s*.69],[.055,.05,s*.69],.038,steel);box(scene,steel,[.055,.065,s*.69],[.28,.07,.24]);}
 // Reward reservoir, supply tubing and a stainless-steel lick spout at the mouth.
 const glass=new T.MeshPhysicalMaterial({color:'#d4e4e9',transparent:true,opacity:.24,roughness:.1,metalness:0,depthWrite:false});const waterMat=new T.MeshPhysicalMaterial({color:'#6dc4e9',transparent:true,opacity:.7,roughness:.12,metalness:.12,clearcoat:1});
 const reservoir=mesh(new T.CylinderGeometry(.14,.14,.62,36),glass);reservoir.position.set(1.56,1.12,-.65);const water=mesh(new T.CylinderGeometry(.125,.125,.37,36),waterMat);water.position.set(1.56,1.01,-.65);box(scene,steel,[1.56,1.45,-.65],[.31,.045,.31]);rod(scene,[1.76,.08,-.65],[1.76,1.5,-.65],.027,steel);rod(scene,[1.76,1.37,-.65],[1.56,1.37,-.65],.022,steel);
 tube(scene,[[1.56,.8,-.65],[1.57,.50,-.60],[1.34,.40,-.25],[1.17,.51,0]],.018,mat('#98adb7',.32));rod(scene,[1.19,.515,0],[1.047,.528,0],.018,steel);
 const droplet=oval(scene,waterMat,[1.035,.513,0],[.040,.052,.038]);const tongue=oval(mouse,mat('#cf8c90',.46),[.977,.505,0],[.068,.010,.032]);droplet.visible=tongue.visible=false;
 // A physical speaker replaces the previous decorative category screen.
 const speaker=new T.Group();speaker.position.set(1.74,.55,.65);speaker.rotation.y=-.28;scene.add(speaker);box(speaker,dark,[0,0,0],[.16,.5,.4]);const cone=mesh(new T.CylinderGeometry(.145,.12,.03,40),black,speaker);cone.rotation.z=Math.PI/2;cone.position.x=-.09;for(let i=0;i<7;i++)rod(speaker,[-.115,-.17+i*.055,-.16],[-.115,-.17+i*.055,.16],.003,steel);rod(scene,[1.74,.05,.65],[1.74,.32,.65],.03,steel);
 let azimuth=1.04,elevation=.40,dragging=false,px=0,py=0;host.addEventListener('pointerdown',e=>{if(e.target!==renderer.domElement)return;dragging=true;px=e.clientX;py=e.clientY;host.setPointerCapture(e.pointerId);});host.addEventListener('pointermove',e=>{if(!dragging)return;azimuth+=(e.clientX-px)*.008;elevation=Math.max(.12,Math.min(.9,elevation+(e.clientY-py)*.005));px=e.clientX;py=e.clientY;draw();});host.addEventListener('pointerup',()=>dragging=false);host.addEventListener('pointercancel',()=>dragging=false);
 document.querySelector('#scene-view').onchange=e=>{azimuth=e.target.value==='side'?1.57:1.04;elevation=e.target.value==='side'?.18:.40;draw();};
 draw=()=>{const move=Math.max(0,Math.min(1,(phase-.34)/.30)),direction=row.choice?1:-1;wheel.rotation.z=direction*move*Math.PI/6;paws.forEach(p=>{p.position.x=.43+Math.sin(move*Math.PI)*direction*.04;p.position.y=.39+Math.sin(move*Math.PI)*.022;});const rewarded=phase>=.72&&!!row.reward;droplet.visible=tongue.visible=rewarded;const lick=phase>=1?.65:(.5+.5*Math.sin((phase-.72)*Math.PI*24));tongue.scale.x=.028+.047*lick;droplet.scale.set(.035*(1-.28*lick),.046*(1-.18*lick),.035);jaw.position.y=.506-(rewarded?.009*lick:0);
 const distance=6.8*Math.max(1,.95/camera.aspect);camera.position.set(-.35+Math.cos(azimuth)*distance,.52+Math.sin(elevation)*distance,Math.sin(azimuth)*distance);camera.lookAt(-.35,.57,0);renderer.render(scene,camera);};
 new ResizeObserver(()=>{const w=host.clientWidth,h=host.clientHeight;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();draw();}).observe(host);draw();
 }catch(e){const p=document.createElement('p');p.className='load-error';p.textContent='3D view unavailable. Trial playback and model values remain available.';host.append(p);console.error(e);}
 return(r,t)=>{row=r;phase=t;draw();};
};
