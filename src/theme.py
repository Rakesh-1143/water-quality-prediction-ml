"""Design system for the Streamlit app: palettes, glass UI, 3D hero scenes, scroll animations.

Everything is plain CSS / vanilla JS (no external assets), so it works offline. Motion respects
`prefers-reduced-motion`. Text passed to the HTML helpers is escaped.
"""
import html
import itertools
import re

import matplotlib as mpl
import streamlit as st

PALETTES = {
    "Ocean": dict(primary="#0284c7", primary2="#06b6d4", accent="#f97316", ink="#0b2a3b", muted="#4a6b7c",
                  bg1="#f3faff", bg2="#dff3fb", blob1="rgba(14,165,233,.30)", blob2="rgba(6,182,212,.26)",
                  hue="0", scene="waves", warm="#f97316"),
    "Blue": dict(primary="#1d4ed8", primary2="#3b82f6", accent="#f59e0b", ink="#0b1b3a", muted="#4a5d85",
                 bg1="#f3f7ff", bg2="#dbe7ff", blob1="rgba(37,99,235,.28)", blob2="rgba(56,189,248,.24)",
                 hue="22", scene="rings", warm="#f59e0b"),
    "Lagoon": dict(primary="#0f766e", primary2="#22c55e", accent="#f59e0b", ink="#0a2e2a", muted="#46685f",
                   bg1="#f2fcf8", bg2="#dcf5ea", blob1="rgba(16,185,129,.28)", blob2="rgba(45,212,191,.26)",
                   hue="-25", scene="orbs", warm="#f59e0b"),
    "Aurora": dict(primary="#6366f1", primary2="#a855f7", accent="#ec4899", ink="#1b1b3a", muted="#5b5b85",
                   bg1="#f7f5ff", bg2="#e9e5ff", blob1="rgba(99,102,241,.30)", blob2="rgba(236,72,153,.22)",
                   hue="40", scene="helix", warm="#ec4899"),
}
DEFAULT_THEME = "Ocean"

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root{--wq-primary:__primary__;--wq-primary2:__primary2__;--wq-accent:__accent__;--wq-ink:__ink__;--wq-muted:__muted__;
--wq-glass:rgba(255,255,255,.62);--wq-border:rgba(255,255,255,.75);--wq-shadow:0 10px 30px rgba(11,42,59,.10);
--wq-ease:cubic-bezier(.2,.8,.2,1);--wq-hue:__hue__deg}
html,body,.stApp,[class*="st-"],button,input,textarea{font-family:'Plus Jakarta Sans',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif}
[data-testid="stIconMaterial"],[data-testid="stIconMaterial"] *{font-family:'Material Symbols Rounded','Material Symbols Outlined'!important}
[data-testid="stMain"]{scroll-behavior:smooth}
.stApp{background:linear-gradient(160deg,__bg1__ 0%,__bg2__ 100%) fixed;color:var(--wq-ink)}
.stApp::before,.stApp::after{content:"";position:fixed;z-index:0;pointer-events:none;border-radius:50%;filter:blur(70px)}
.stApp::before{width:46vw;height:46vw;left:-10vw;top:-12vw;background:__blob1__;animation:wq-float-a 22s ease-in-out infinite alternate}
.stApp::after{width:40vw;height:40vw;right:-8vw;bottom:-10vw;background:__blob2__;animation:wq-float-b 26s ease-in-out infinite alternate}
@keyframes wq-float-a{to{transform:translate3d(9vw,7vw,0) scale(1.15)}}
@keyframes wq-float-b{to{transform:translate3d(-8vw,-6vw,0) scale(1.2)}}
[data-testid="stHeader"]{background:transparent}
[data-testid="stMainBlockContainer"]{position:relative;z-index:1;padding-top:2.2rem;max-width:1180px;animation:wq-rise .8s var(--wq-ease) both}
@keyframes wq-rise{from{opacity:0;transform:translate3d(0,22px,0)}to{opacity:1;transform:none}}
h1,h2,h3{color:var(--wq-ink);letter-spacing:-.02em;font-weight:800}
[data-testid="stMainBlockContainer"] h3{position:relative;padding-bottom:.35rem}
[data-testid="stMainBlockContainer"] h3::after{content:"";position:absolute;left:0;bottom:0;height:3px;width:46px;border-radius:3px;
background:linear-gradient(90deg,var(--wq-primary),var(--wq-primary2));transition:width .6s var(--wq-ease)}
.wq-in h3::after,[data-testid="stMainBlockContainer"] h3:hover::after{width:90px}
/* sidebar */
[data-testid="stSidebar"]{background:linear-gradient(190deg,rgba(255,255,255,.78),rgba(255,255,255,.52));backdrop-filter:blur(18px);
border-right:1px solid var(--wq-border);box-shadow:8px 0 30px rgba(11,42,59,.06)}
[data-testid="stSidebarNavLink"]{border-radius:12px;transition:transform .25s var(--wq-ease),background .25s}
[data-testid="stSidebarNavLink"]:hover{transform:translateX(5px);background:rgba(255,255,255,.7)}
[data-testid="stSidebarNavLink"][aria-current="page"]{background:linear-gradient(90deg,var(--wq-primary),var(--wq-primary2));color:#fff!important;box-shadow:0 6px 16px rgba(2,132,199,.25)}
[data-testid="stSidebarNavLink"][aria-current="page"] span{color:#fff!important}
.wq-brand{display:flex;align-items:center;gap:.6rem;padding:.4rem .2rem 1rem}
.wq-brand i{width:38px;height:38px;border-radius:12px;display:grid;place-items:center;font-style:normal;font-size:1.3rem;color:#fff;
background:linear-gradient(135deg,var(--wq-primary),var(--wq-primary2));box-shadow:0 8px 18px rgba(2,132,199,.35);animation:wq-bob 4s ease-in-out infinite}
.wq-brand b{display:block;font-size:1rem;color:var(--wq-ink)}.wq-brand span{font-size:.72rem;color:var(--wq-muted)}
@keyframes wq-bob{50%{transform:translateY(-3px) rotate(-4deg)}}
/* glass cards: metrics, alerts, tables, images */
[data-testid="stMetric"],.wq-card{background:var(--wq-glass);backdrop-filter:blur(14px);border:1px solid var(--wq-border);border-radius:20px;
box-shadow:var(--wq-shadow);padding:1.1rem 1.25rem;transform-style:preserve-3d;transition:box-shadow .35s var(--wq-ease),transform .25s var(--wq-ease);position:relative;overflow:hidden}
[data-testid="stMetric"]::after,.wq-card::after{content:"";position:absolute;inset:0;pointer-events:none;opacity:0;transition:opacity .3s;
background:radial-gradient(260px circle at var(--gx,50%) var(--gy,0%),rgba(255,255,255,.75),transparent 60%)}
[data-testid="stMetric"]:hover,.wq-card:hover{box-shadow:0 22px 46px rgba(11,42,59,.18)}
[data-testid="stMetric"]:hover::after,.wq-card:hover::after{opacity:1}
[data-testid="stMetricValue"]{font-weight:800;background:linear-gradient(90deg,var(--wq-primary),var(--wq-primary2));-webkit-background-clip:text;background-clip:text;color:transparent}
[data-testid="stAlert"]{border-radius:16px;border:1px solid var(--wq-border);box-shadow:var(--wq-shadow);backdrop-filter:blur(10px)}
[data-testid="stDataFrame"],[data-testid="stImage"] img,[data-testid="stTable"]{border-radius:16px;overflow:hidden;box-shadow:var(--wq-shadow);border:1px solid var(--wq-border)}
[data-testid="stExpander"]{border-radius:16px;border:1px solid var(--wq-border);background:var(--wq-glass);box-shadow:var(--wq-shadow)}
/* buttons */
.stButton>button,[data-testid="stBaseButton-secondary"]{position:relative;overflow:hidden;border-radius:14px;border:0;font-weight:700;color:#fff;
background:linear-gradient(120deg,var(--wq-primary),var(--wq-primary2),var(--wq-primary));background-size:200% 100%;padding:.7rem 1rem;
box-shadow:0 10px 22px rgba(2,132,199,.28);transition:transform .25s var(--wq-ease),box-shadow .25s,background-position .6s}
.stButton>button p{color:#fff}
.stButton>button:hover{transform:translateY(-3px) scale(1.015);background-position:100% 0;box-shadow:0 16px 30px rgba(2,132,199,.38);color:#fff}
.stButton>button:active{transform:translateY(0) scale(.98)}
.stButton>button::after{content:"";position:absolute;top:0;left:-70%;width:50%;height:100%;transform:skewX(-20deg);background:rgba(255,255,255,.35);transition:left .6s var(--wq-ease)}
.stButton>button:hover::after{left:130%}
[data-testid="stPageLink"] a{border-radius:12px;font-weight:700;transition:transform .25s var(--wq-ease)}[data-testid="stPageLink"] a:hover{transform:translateX(4px)}
/* inputs */
[data-testid="stSlider"]{filter:hue-rotate(var(--wq-hue))}
[data-baseweb="tab-highlight"]{background:linear-gradient(90deg,var(--wq-primary),var(--wq-primary2))!important;height:3px!important;border-radius:3px}
[data-baseweb="tab"]{transition:transform .2s}[data-baseweb="tab"]:hover{transform:translateY(-2px)}
[data-baseweb="select"]>div,[data-baseweb="input"]>div{border-radius:12px}
/* scroll progress + reveal */
#wq-progress{position:fixed;top:0;left:0;height:4px;width:0;z-index:99999;background:linear-gradient(90deg,var(--wq-primary),var(--wq-primary2),var(--wq-accent));
box-shadow:0 0 12px var(--wq-primary2);transition:width .12s linear}
.wq-js .wq-pre{opacity:0;transform:translate3d(0,34px,0) scale(.97);filter:blur(4px)}
.wq-js .wq-pre,.wq-js .wq-in{transition:opacity .8s var(--wq-ease),transform .8s var(--wq-ease),filter .8s var(--wq-ease)}
.wq-js .wq-in{opacity:1;transform:none;filter:none}
/* hero */
.wq-hero{position:relative;height:var(--h,260px);border-radius:28px;overflow:hidden;margin-bottom:1.2rem;color:#fff;
background:linear-gradient(135deg,__primary__ 0%,__primary2__ 100%);box-shadow:0 26px 60px rgba(11,42,59,.25);transform-style:preserve-3d}
.wq-hero canvas{position:absolute;inset:0;width:100%;height:100%}
.wq-hero::before{content:"";position:absolute;inset:0;background:radial-gradient(70% 90% at 20% 0%,rgba(255,255,255,.22),transparent 60%),linear-gradient(90deg,rgba(6,24,38,.45),transparent 65%);z-index:1}
.wq-hero-content{position:relative;z-index:2;height:100%;display:flex;flex-direction:column;justify-content:center;padding:0 2.4rem;max-width:760px}
.wq-kicker{align-self:flex-start;font-size:.74rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;padding:.3rem .8rem;border-radius:99px;
background:rgba(255,255,255,.2);border:1px solid rgba(255,255,255,.4);backdrop-filter:blur(6px);animation:wq-rise .9s var(--wq-ease) .1s both}
.wq-hero h1{color:#fff;margin:.7rem 0 .4rem;font-size:clamp(1.8rem,3.6vw,3rem);line-height:1.08;text-shadow:0 4px 24px rgba(0,0,0,.25);animation:wq-rise .9s var(--wq-ease) .2s both}
.wq-hero p{color:rgba(255,255,255,.92);font-size:1.02rem;max-width:560px;margin:0;animation:wq-rise .9s var(--wq-ease) .35s both}
.wq-stats{display:flex;gap:2rem;margin-top:1.3rem;animation:wq-rise .9s var(--wq-ease) .5s both}
.wq-stats b{display:block;font-size:1.7rem;font-weight:800;line-height:1}.wq-stats span{font-size:.72rem;opacity:.85;text-transform:uppercase;letter-spacing:.08em}
/* feature cards + timeline + gauge + ladder + bars */
.wq-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:1rem;margin:.4rem 0 1rem;perspective:1000px}
.wq-card h4{margin:.5rem 0 .25rem;font-size:1.05rem;font-weight:800;color:var(--wq-ink)}.wq-card p{margin:0;color:var(--wq-muted);font-size:.9rem;line-height:1.5}
.wq-ico{width:46px;height:46px;border-radius:14px;display:grid;place-items:center;font-size:1.4rem;color:#fff;transform:translateZ(30px);
background:linear-gradient(135deg,var(--wq-primary),var(--wq-primary2));box-shadow:0 10px 20px rgba(2,132,199,.3)}
.wq-timeline{position:relative;display:grid;grid-template-columns:repeat(auto-fit,minmax(112px,1fr));gap:.8rem;margin:1rem 0 1.5rem}
.wq-timeline::before{content:"";position:absolute;top:22px;left:4%;right:4%;height:3px;border-radius:3px;background:linear-gradient(90deg,var(--wq-primary),var(--wq-primary2),var(--wq-accent));
transform-origin:left;transform:scaleX(0);transition:transform 1.6s var(--wq-ease) .2s}
.wq-in .wq-timeline::before,.wq-in.wq-timeline::before{transform:scaleX(1)}body:not(.wq-js) .wq-timeline::before{transform:none}
.wq-step{position:relative;text-align:center}.wq-step i{position:relative;z-index:1;display:grid;place-items:center;width:46px;height:46px;margin:0 auto .5rem;border-radius:50%;
font-style:normal;font-weight:800;color:#fff;background:linear-gradient(135deg,var(--wq-primary),var(--wq-primary2));box-shadow:0 0 0 6px rgba(255,255,255,.7),0 10px 22px rgba(2,132,199,.3);animation:wq-pulse 3.2s ease-in-out infinite}
.wq-step:nth-child(2) i{animation-delay:.4s}.wq-step:nth-child(3) i{animation-delay:.8s}.wq-step:nth-child(4) i{animation-delay:1.2s}.wq-step:nth-child(5) i{animation-delay:1.6s}.wq-step:nth-child(6) i{animation-delay:2s}
@keyframes wq-pulse{50%{box-shadow:0 0 0 12px rgba(255,255,255,.25),0 10px 22px rgba(2,132,199,.3)}}
.wq-step b{display:block;font-size:.92rem;color:var(--wq-ink)}.wq-step span{font-size:.78rem;color:var(--wq-muted)}
.wq-gauge{display:flex;align-items:center;gap:1.2rem;padding:1rem 1.2rem}
.wq-ring{--wq-p:0;width:124px;height:124px;border-radius:50%;flex:none;transform:translateZ(24px);filter:drop-shadow(0 10px 16px rgba(2,132,199,.28));
background:conic-gradient(from 0deg,var(--wq-primary),var(--wq-primary2) calc(var(--wq-p)*3.6deg),rgba(11,42,59,.10) calc(var(--wq-p)*3.6deg));
-webkit-mask:radial-gradient(farthest-side,transparent calc(100% - 15px),#000 calc(100% - 14px));mask:radial-gradient(farthest-side,transparent calc(100% - 15px),#000 calc(100% - 14px));
animation:wq-ring 1.7s var(--wq-ease) .15s forwards}@keyframes wq-ring{to{--wq-p:var(--v)}}
.wq-gauge .pct{font-size:2.3rem;font-weight:800;line-height:1;background:linear-gradient(90deg,var(--wq-primary),var(--wq-primary2));-webkit-background-clip:text;background-clip:text;color:transparent}
.wq-gauge .lbl{color:var(--wq-muted);font-size:.85rem;margin-top:.3rem}
.wq-ladder{display:flex;gap:.6rem;margin:.6rem 0 1rem;perspective:900px}.wq-ladder div{flex:1;text-align:center;padding:.9rem .4rem;border-radius:16px;font-weight:700;font-size:.9rem;
background:var(--wq-glass);border:1px solid var(--wq-border);color:var(--wq-muted);transition:transform .5s var(--wq-ease),box-shadow .5s;transform:rotateX(14deg) translateZ(0)}
.wq-ladder div.on{color:#fff;background:linear-gradient(135deg,var(--wq-primary),var(--wq-primary2));transform:rotateX(0) translateZ(26px) scale(1.07);box-shadow:0 18px 34px rgba(2,132,199,.4)}
.wq-bars{display:grid;gap:.7rem;margin:.4rem 0 1rem}.wq-bar{display:grid;grid-template-columns:170px 1fr 54px;align-items:center;gap:.8rem;font-size:.88rem;color:var(--wq-ink);font-weight:600}
.wq-bar em{height:14px;border-radius:99px;background:rgba(11,42,59,.08);overflow:hidden;font-style:normal}.wq-bar i{display:block;height:100%;width:var(--w);border-radius:99px;transform-origin:left;
background:linear-gradient(90deg,var(--wq-primary),var(--wq-primary2));transition:transform 1.3s var(--wq-ease) var(--d,0s)}.wq-bar.top i{background:linear-gradient(90deg,var(--wq-accent),#fbbf24)}
.wq-js .wq-bar i{transform:scaleX(0)}.wq-js .wq-in .wq-bar i{transform:scaleX(1)}.wq-bar b{text-align:right}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}.wq-js .wq-pre{opacity:1;transform:none;filter:none}}
@media (max-width:760px){.wq-hero-content{padding:0 1.2rem}.wq-bar{grid-template-columns:100px 1fr 46px}.wq-ladder{flex-wrap:wrap}}
"""

# Runs once per browser tab: scroll progress bar, scroll-reveal, 3D tilt on cards.
_GLOBAL_JS = r"""
(function(){
 const d=document; if(window.__wq){window.__wq.scan&&window.__wq.scan();return;}
 const reduce=matchMedia('(prefers-reduced-motion: reduce)').matches;
 const scroller=()=>d.querySelector('[data-testid="stMain"]');
 d.body.classList.add('wq-js');
 try{CSS.registerProperty({name:'--wq-p',syntax:'\x3cnumber>',inherits:false,initialValue:'0'});}catch(e){}
 const bar=d.createElement('div');bar.id='wq-progress';d.body.appendChild(bar);
 const onScroll=()=>{const s=scroller();if(!s)return;const m=s.scrollHeight-s.clientHeight;bar.style.width=(m>0?100*s.scrollTop/m:0)+'%';};
 const bind=()=>{const s=scroller();if(s&&!s.__wq){s.__wq=1;s.addEventListener('scroll',onScroll,{passive:true});}};
 let io=null,seen=new WeakSet(),batch=0;
 const mkIO=()=>{const s=scroller();io=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){const el=e.target;
   el.style.transitionDelay=(Math.min(batch++,6)*70)+'ms';el.classList.remove('wq-pre');el.classList.add('wq-in');io.unobserve(el);
   setTimeout(()=>{el.style.transitionDelay='';batch=0;},900);}});},{root:s||null,threshold:.08,rootMargin:'0px 0px -6% 0px'});};
 const scan=()=>{bind();if(!io)mkIO();
  d.querySelectorAll('[data-testid="stElementContainer"],[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]').forEach(el=>{
   if(seen.has(el)||el.querySelector('.wq-hero'))return;seen.add(el);
   if(reduce){el.classList.add('wq-in');return;}
   const r=el.getBoundingClientRect(),s=scroller(),vh=s?s.clientHeight:innerHeight;
   if(r.top<vh*0.9&&r.bottom>0&&el.closest('[data-testid="stSidebar"]')){return;}
   if(el.closest('[data-testid="stSidebar"]'))return;
   el.classList.add('wq-pre');io.observe(el);});
  onScroll();};
 window.__wq={scan};
 new MutationObserver(()=>{clearTimeout(window.__wqT);window.__wqT=setTimeout(scan,60);}).observe(d.body,{childList:true,subtree:true});
 scan();setTimeout(scan,400);
 // 3D tilt following the pointer
 let cur=null;
 const reset=el=>{if(el){el.style.transform='';}};
 d.addEventListener('mousemove',e=>{if(reduce)return;
  const el=e.target.closest&&e.target.closest('.wq-card,[data-testid="stMetric"]');
  if(cur&&cur!==el){reset(cur);cur=null;}
  if(!el)return;cur=el;const r=el.getBoundingClientRect(),x=(e.clientX-r.left)/r.width,y=(e.clientY-r.top)/r.height;
  el.style.transform='perspective(800px) rotateX('+((.5-y)*10).toFixed(2)+'deg) rotateY('+((x-.5)*12).toFixed(2)+'deg) translateZ(8px)';
  el.style.setProperty('--gx',(x*100)+'%');el.style.setProperty('--gy',(y*100)+'%');},{passive:true});
 d.addEventListener('mouseleave',()=>{reset(cur);cur=null;},true);
})();
"""

# Animated 3D scenes drawn on a 2D canvas with a manual perspective projection.
_SCENE_JS = r"""
(function(){
 const c=document.getElementById('__ID__');if(!c)return;const ctx=c.getContext('2d');
 const P=['__C1__','__C2__','__C3__'];const reduce=matchMedia('(prefers-reduced-motion: reduce)').matches;
 let W=0,H=0,mx=0,my=0,t=0,sy=0;const dpr=Math.min(window.devicePixelRatio||1,2);
 const size=()=>{const r=c.getBoundingClientRect();W=r.width;H=r.height;c.width=W*dpr;c.height=H*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);};size();addEventListener('resize',size);
 const host=c.parentElement;host.addEventListener('mousemove',e=>{const r=host.getBoundingClientRect();mx=(e.clientX-r.left)/r.width-.5;my=(e.clientY-r.top)/r.height-.5;});
 const sc=document.querySelector('[data-testid="stMain"]');if(sc)sc.addEventListener('scroll',()=>{sy=sc.scrollTop;},{passive:true});
 const rot=(x,y,z,ax,ay)=>{let c1=Math.cos(ay),s1=Math.sin(ay),X=x*c1+z*s1,Z=-x*s1+z*c1;let c2=Math.cos(ax),s2=Math.sin(ax),Y=y*c2-Z*s2;Z=y*s2+Z*c2;return[X,Y,Z];};
 const pr=(x,y,z,f,ox,oy)=>{const k=f/(f+z);return[ox+x*k,oy+y*k,k];};
 const dot=(x,y,r,col,a)=>{ctx.globalAlpha=a;ctx.fillStyle=col;ctx.beginPath();ctx.arc(x,y,r,0,6.2832);ctx.fill();};
 const scenes={
 waves(){const N=30,S=.55,pts=[];const ax=.95+sy*.0007+my*.25,ay=mx*.5+Math.sin(t*.2)*.08;
  for(let i=0;i<=N;i++){pts[i]=[];for(let j=0;j<=N;j++){const x=(i-N/2)*S,z=(j-N/2)*S,y=Math.sin(x*1.1+t)*Math.cos(z*1.1+t*.8)*.55+Math.sin((x+z)*.7+t*1.3)*.25;
   const q=rot(x,y,z,ax,ay);pts[i][j]=pr(q[0]*64,q[1]*64,q[2]*64+300,520,W*.7,H*.58);}}
  ctx.lineWidth=1.1;for(let i=0;i<=N;i++){for(let j=0;j<=N;j++){const p=pts[i][j];const a=Math.max(.14,Math.min(.95,(p[2]-.5)*1.9));ctx.strokeStyle=P[(i+j)%3];ctx.globalAlpha=a;
   if(i<N){ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(pts[i+1][j][0],pts[i+1][j][1]);ctx.stroke();}
   if(j<N){ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(pts[i][j+1][0],pts[i][j+1][1]);ctx.stroke();}
   if((i*7+j*3)%11===0)dot(p[0],p[1],2.2*p[2],'#fff',a);}}},
 orbs(){const pts=scenes._o||(scenes._o=Array.from({length:520},(_,i)=>{const k=i+.5,ph=Math.acos(1-2*k/520),th=Math.PI*(1+Math.sqrt(5))*k;return[Math.cos(th)*Math.sin(ph),Math.cos(ph),Math.sin(th)*Math.sin(ph),i%3];}));
  const ax=my*.5+sy*.0006,ay=t*.35+mx*.9,R=118+Math.sin(t)*6;const out=[];
  pts.forEach(p=>{const q=rot(p[0]*R,p[1]*R,p[2]*R,ax,ay);const o=pr(q[0],q[1],q[2]+260,440,W*.8,H*.5);out.push([o[0],o[1],o[2],q[2],p[3]]);});
  out.sort((a,b)=>b[3]-a[3]);out.forEach(o=>dot(o[0],o[1],2.6*o[2],P[o[4]],Math.max(.15,Math.min(1,o[2]*.85))));
  for(let r=0;r<3;r++){for(let k=0;k<90;k++){const a=k/90*6.2832+t*(.5+r*.2);const q=rot(Math.cos(a)*(165+r*34),0,Math.sin(a)*(165+r*34),ax+.9+r*.35,ay*.4);const o=pr(q[0],q[1],q[2]+260,440,W*.8,H*.5);dot(o[0],o[1],1.8*o[2],'#fff',.45*o[2]);}}},
 rings(){const ax=.5+my*.5+sy*.0006,ay=t*.3+mx*.8,cx=W*.8,cy=H*.5;
  const g=ctx.createRadialGradient(cx,cy,0,cx,cy,70);g.addColorStop(0,'rgba(255,255,255,.55)');g.addColorStop(1,'rgba(255,255,255,0)');ctx.globalAlpha=1;ctx.fillStyle=g;ctx.fillRect(cx-70,cy-70,140,140);
  const items=[];for(let r=0;r < 5;r++){const R=58+r*27,tx=ax+r*.55+Math.sin(t*.4+r)*.3,ty=ay*(1+r*.25)+t*.18*r;
   for(let k=0;k < 110;k++){const a=k/110*6.2832,q=rot(Math.cos(a)*R,0,Math.sin(a)*R,tx,ty),o=pr(q[0],q[1],q[2]+260,460,cx,cy);items.push([o,r,q[2]]);}}
  items.sort((a,b)=>b[2]-a[2]);items.forEach(it=>dot(it[0][0],it[0][1],2.6*it[0][2],P[it[1]%3],Math.max(.2,Math.min(1,it[0][2]*.9))));},
 helix(){const n=70,ay=t*.5+mx*.8,ax=.35+my*.4+sy*.0005;const items=[];
  for(let i=0;i<n;i++){const y=(i/n-.5)*330,a=i*.34+t;for(let s=0;s<2;s++){const ph=a+s*Math.PI,q=rot(Math.cos(ph)*84,y,Math.sin(ph)*84,ax,ay);const o=pr(q[0],q[1],q[2]+260,460,W*.8,H*.5);items.push([o,s,i,q[2]]);}
   if(i%2===0){const q1=rot(Math.cos(a)*84,y,Math.sin(a)*84,ax,ay),q2=rot(Math.cos(a+Math.PI)*84,y,Math.sin(a+Math.PI)*84,ax,ay);const o1=pr(q1[0],q1[1],q1[2]+260,460,W*.8,H*.5),o2=pr(q2[0],q2[1],q2[2]+260,460,W*.8,H*.5);
    ctx.strokeStyle='#fff';ctx.globalAlpha=.22;ctx.lineWidth=1.2;ctx.beginPath();ctx.moveTo(o1[0],o1[1]);ctx.lineTo(o2[0],o2[1]);ctx.stroke();}}
  items.sort((a,b)=>b[3]-a[3]);items.forEach(it=>dot(it[0][0],it[0][1],6*it[0][2],P[it[1]],Math.max(.25,Math.min(1,it[0][2]))));}
 };
 const draw=scenes['__SCENE__'];
 const frame=()=>{if(!document.body.contains(c))return;t+=reduce?0:.016;ctx.clearRect(0,0,W,H);ctx.lineCap='round';draw();ctx.globalAlpha=1;requestAnimationFrame(frame);};
 frame();
})();
"""

_ids = itertools.count(1)


def safe_js(code):
    """st.html's HTML parser treats a `<` that is not followed by a space or `=` as the start of a
    tag and breaks the script, so every less-than operator gets surrounding spaces."""
    return re.sub(r"<(?![ =])", " < ", code.replace("<=", " <= "))


def current():
    """Name of the selected palette (kept in session state so it follows across pages)."""
    name = st.session_state.get("design_theme") or st.session_state.get("_wq_theme") or DEFAULT_THEME
    return name if name in PALETTES else DEFAULT_THEME


def _css(p):
    css = _CSS
    for key, value in p.items():
        css = css.replace(f"__{key}__", value)
    return css


def style_matplotlib(p=None):
    """Charts follow the palette: clean spines, soft grid, palette colour cycle."""
    p = p or PALETTES[current()]
    mpl.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3d9e6",
        "axes.labelcolor": p["ink"], "xtick.color": p["muted"], "ytick.color": p["muted"],
        "text.color": p["ink"], "axes.titleweight": "bold", "axes.titlesize": 11, "axes.grid": True,
        "grid.color": "#e3eef4", "grid.linewidth": .8, "axes.axisbelow": True, "font.size": 9,
        "axes.prop_cycle": mpl.cycler(color=[p["primary"], p["accent"], p["primary2"], "#64748b", "#14b8a6"]),
        "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
    })


def colors():
    """Palette colours for charts: primary, warm accent, secondary."""
    p = PALETTES[current()]
    return p["primary"], p["warm"], p["primary2"]


def apply(brand_title="Water Quality ML", brand_sub="SHAP-ANN · XGBoost · LSTM"):
    """Inject the design system. Call right after `st.set_page_config` on every page."""
    st.sidebar.markdown(f'<div class="wq-brand"><i>💧</i><div><b>{html.escape(brand_title)}</b>'
                        f'<span>{html.escape(brand_sub)}</span></div></div>', unsafe_allow_html=True)
    if "design_theme" not in st.session_state and "_wq_theme" in st.session_state:
        st.session_state["design_theme"] = st.session_state["_wq_theme"]   # widget state is reset on page change
    st.sidebar.segmented_control("Design theme", list(PALETTES), default=DEFAULT_THEME,
                                 key="design_theme", selection_mode="single",
                                 help="Switch between three visual designs (palette + 3D scene).")
    st.session_state["_wq_theme"] = current()
    p = PALETTES[current()]
    st.markdown(f"<style>{_css(p)}</style>", unsafe_allow_html=True)
    st.html(f"<script>{safe_js(_GLOBAL_JS)}</script>", unsafe_allow_javascript=True)
    style_matplotlib(p)


def hero(kicker, title, subtitle, stats=None, height=260, scene=None):
    """Full-width hero banner with an animated 3D canvas scene behind the text."""
    p = PALETTES[current()]
    cid = f"wq-canvas-{next(_ids)}"
    stats_html = "".join(f"<div><b>{html.escape(str(v))}</b><span>{html.escape(k)}</span></div>"
                         for k, v in (stats or {}).items())
    js = (_SCENE_JS.replace("__ID__", cid).replace("__SCENE__", scene or p["scene"])
          .replace("__C1__", "#ffffff").replace("__C2__", "#bae6fd").replace("__C3__", "#fde68a"))
    st.html(f'<div class="wq-hero" style="--h:{int(height)}px"><canvas id="{cid}"></canvas>'
            f'<div class="wq-hero-content"><span class="wq-kicker">{html.escape(kicker)}</span>'
            f'<h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p>'
            f'{"<div class=wq-stats>" + stats_html + "</div>" if stats_html else ""}</div></div>'
            f'<script>{safe_js(js)}</script>', unsafe_allow_javascript=True)


def feature_cards(items):
    """Grid of 3D-tilt cards. items = [(icon, title, text), ...]."""
    cards = "".join(f'<div class="wq-card"><div class="wq-ico">{html.escape(i)}</div><h4>{html.escape(t)}</h4>'
                    f'<p>{html.escape(x)}</p></div>' for i, t, x in items)
    st.html(f'<div class="wq-grid">{cards}</div>')


def timeline(steps):
    """Pipeline steps with an animated connector. steps = [(title, caption), ...]."""
    body = "".join(f'<div class="wq-step"><i>{n}</i><b>{html.escape(t)}</b><span>{html.escape(c)}</span></div>'
                   for n, (t, c) in enumerate(steps, 1))
    st.html(f'<div class="wq-timeline">{body}</div>')


def gauge(value, label):
    """Animated ring gauge for a probability in [0, 1]."""
    value = max(0.0, min(1.0, float(value)))
    st.html(f'<div class="wq-card wq-gauge"><div class="wq-ring" style="--v:{value * 100:.1f}"></div>'
            f'<div><div class="pct">{value:.0%}</div><div class="lbl">{html.escape(label)}</div></div></div>')


def ladder(levels, active):
    """Row of quality levels with the predicted one lifted in 3D."""
    body = "".join(f'<div class="{"on" if i == active else ""}">{html.escape(str(n))}</div>' for i, n in enumerate(levels))
    st.html(f'<div class="wq-ladder">{body}</div>')


def leaderboard(scores, fmt="{:.1%}"):
    """Animated horizontal bars. scores = {name: value in [0, 1]}; the best one is highlighted."""
    best = max(scores.values())
    rows = "".join(
        f'<div class="wq-bar{" top" if v == best else ""}"><span>{html.escape(str(k))}</span>'
        f'<em><i style="--w:{max(0.0, min(1.0, v)) * 100:.1f}%;--d:{n * .12:.2f}s"></i></em><b>{fmt.format(v)}</b></div>'
        for n, (k, v) in enumerate(scores.items()))
    st.html(f'<div class="wq-bars">{rows}</div>')
