"""Linked macro and local-volume visualizations for H3-W11."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _json_for_html(value: Any) -> str:
    return json.dumps(value, separators=(",", ":")).replace("</", "<\\/")


def _display_step_sample(steps: list[dict[str, Any]], limit: int = 20000) -> list[dict[str, Any]]:
    if len(steps) <= limit:
        return steps
    interactions = [row for row in steps if bool(row.get("is_interaction"))]
    keep_interactions = interactions[: limit // 2]
    interaction_ids = {id(row) for row in keep_interactions}
    others = [row for row in steps if id(row) not in interaction_ids]
    remaining = limit - len(keep_interactions)
    sampled = [others[round(i * (len(others) - 1) / max(1, remaining - 1))] for i in range(remaining)]
    selected = keep_interactions + sampled
    selected.sort(key=lambda row: (int(row.get("geant4_event_id", 0)), int(row.get("event_step_index", 0))))
    return selected


def write_geant4_visualizations(
    values: dict[str, dict[str, Any]], run_output_dir: Path, output: Path,
    events: list[dict[str, Any]], escaped: list[dict[str, Any]], steps: list[dict[str, Any]],
    *, material: str, density_g_cm3: float,
) -> list[dict[str, Any]]:
    density_applied = material == "HADROS3_H_HE"
    density_policy = "configured local DIS density applied to custom H/He material" if density_applied else "built-in Geant4 NIST material density; configured local density is metadata only"
    dis_rows = _read_jsonl(run_output_dir / "DIS" / "dis_accepted_interactions.jsonl")
    dis_by_id = {str(row.get("interaction_id")): row for row in dis_rows}
    mass_msun = float(values.get("black_hole", {}).get("mass_msun", 3.0))
    rg_mm = 6.67430e-8 * mass_msun * 1.98847e33 / (2.99792458e10**2) * 10.0
    half_size_mm = float(values.get("geant4", {}).get("patch_half_size_mm", 10.0))
    steps_by_event: dict[int, list[dict[str, Any]]] = {}
    escaped_by_event: dict[int, list[dict[str, Any]]] = {}
    for row in steps:
        steps_by_event.setdefault(int(row["geant4_event_id"]), []).append(row)
    for row in escaped:
        escaped_by_event.setdefault(int(row["geant4_event_id"]), []).append(row)

    sites: list[dict[str, Any]] = []
    display_steps = _display_step_sample(steps)
    display_counts: dict[int, int] = {}
    for row in display_steps:
        event_id = int(row["geant4_event_id"])
        display_counts[event_id] = display_counts.get(event_id, 0) + 1
    for index, event in enumerate(events):
        event_id = int(event["geant4_event_id"])
        interaction_id = str(event.get("interaction_id") or "")
        dis = dis_by_id.get(interaction_id)
        exact = dis is not None
        r = float(dis["interaction_r_rg"]) if exact else float(values["analytic_torus"]["r_peak_rg"])
        theta = float(dis["interaction_theta_rad"]) if exact else math.pi / 2.0
        phi = float(dis["interaction_phi_rad"]) if exact else 2.0 * math.pi * index / max(1, len(events))
        xyz = [r * math.sin(theta) * math.cos(phi), r * math.sin(theta) * math.sin(phi), r * math.cos(theta)]
        event_steps = steps_by_event.get(event_id, [])
        sites.append({
            "geant4_event_id": event_id,
            "event_generation_event_id": event.get("event_generation_event_id"),
            "interaction_id": interaction_id or None,
            "global_position_available": exact,
            "position_source": "DIS accepted interaction" if exact else "schematic fixture fallback",
            "r_rg": r, "theta_rad": theta, "phi_rad": phi, "position_xyz_rg": xyz,
            "density_g_cm3": float(dis.get("interaction_rho_g_cm3", density_g_cm3)) if exact else density_g_cm3,
            "density_applied_to_material": density_applied, "material_density_policy": density_policy,
            "material": material, "patch_half_size_mm": half_size_mm,
            "patch_half_size_rg": half_size_mm / rg_mm,
            "recorded_steps": len(event_steps),
            "visualized_steps": display_counts.get(event_id, 0),
            "interaction_steps": sum(bool(row.get("is_interaction")) for row in event_steps),
            "tracks": len({int(row["track_id"]) for row in event_steps}),
            "escaped_particles": len(escaped_by_event.get(event_id, [])),
            "deposited_gev": float(event.get("deposited_gev", 0.0)),
            "local_view": f"geant4_event_view.html?event={event_id}",
        })

    site_payload = {
        "coordinate_system": "Cartesian display derived from Boyer-Lindquist-like r, theta, phi",
        "global_length_unit": "r_g", "local_length_unit": "mm", "rg_mm": rg_mm,
        "scale_warning": "The local Geant4 box is an inset and is not drawn to scale in the macro view.",
        "sites": sites,
    }
    (output / "geant4_sites.json").write_text(json.dumps(site_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    macro_data = {"black_hole": values["black_hole"], "torus": values["analytic_torus"], "cone": values["polar_cone"], "sites": sites}
    (output / "geant4_macro_sites_3d.html").write_text(_macro_html(macro_data), encoding="utf-8")
    local_data = {"material": material, "density_g_cm3": density_g_cm3, "density_applied_to_material": density_applied, "material_density_policy": density_policy, "half_size_mm": half_size_mm, "sites": sites, "steps": display_steps, "recorded_steps_total": len(steps), "display_steps_truncated": len(display_steps) < len(steps), "escaped": escaped}
    (output / "geant4_event_view.html").write_text(_local_html(local_data), encoding="utf-8")
    return sites


def _macro_html(data: dict[str, Any]) -> str:
    return """<!doctype html><html><head><meta charset="utf-8"><title>HADROS3 GEANT4 Macro Sites</title>
<style>html,body{margin:0;height:100%;overflow:hidden;background:#09111f;color:#e5eefc;font-family:system-ui}canvas{width:100%;height:100%;display:block;cursor:grab}.hud{position:fixed;left:14px;top:12px;max-width:430px;background:#0f172ade;border:1px solid #334155;border-radius:7px;padding:11px;line-height:1.35}.tip{position:fixed;display:none;pointer-events:none;background:#020617ed;border:1px solid #64748b;border-radius:5px;padding:7px;font-size:12px}button{position:fixed;right:14px;top:14px;padding:7px 10px}</style></head>
<body><canvas id="c"></canvas><button id="reset">Reset</button><div class="hud"><b>GEANT4 sites in the HADROS3 macro system</b><br>BH + analytic torus + polar funnels. Drag: rotate · wheel: zoom.<br><b>Click a green site</b> to open its millimetre-scale Geant4 volume.<br><small>The local box is an inset and is not drawn at the macro scale.</small></div><div id="tip" class="tip"></div><script>
const D=__DATA__,c=document.getElementById('c'),x=c.getContext('2d'),tip=document.getElementById('tip');let yaw=-.72,pitch=.38,zoom=1,drag=false,lx=0,ly=0,screenSites=[];
function rot(p){let cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),a=p[0]*cy-p[1]*sy,b=p[0]*sy+p[1]*cy;return[a,b*cp-p[2]*sp,b*sp+p[2]*cp]}
function lim(){return Math.max(D.torus.r_outer_rg,D.cone.r_max_rg)*1.18}function pr(p){let q=rot(p),s=Math.min(innerWidth,innerHeight)/(2*lim())*zoom;return[innerWidth/2+q[0]*s,innerHeight/2-q[1]*s,q[2],s]}
function line(ps,col,w=1,a=1){x.save();x.globalAlpha=a;x.strokeStyle=col;x.lineWidth=w;x.beginPath();ps.forEach((p,i)=>{p=pr(p);i?x.lineTo(p[0],p[1]):x.moveTo(p[0],p[1])});x.stroke();x.restore()}
function ring(radius,z,col,a){let p=[];for(let i=0;i<=96;i++){let q=2*Math.PI*i/96;p.push([radius*Math.cos(q),radius*Math.sin(q),z])}line(p,col,1,a)}
function draw(){x.clearRect(0,0,innerWidth,innerHeight);x.fillStyle='#09111f';x.fillRect(0,0,innerWidth,innerHeight);let L=Math.max(D.torus.r_outer_rg,D.cone.r_max_rg);line([[0,0,0],[L,0,0]],'#475569');line([[0,0,0],[0,L,0]],'#475569');line([[0,0,-L],[0,0,L]],'#64748b',1.2);let R=+D.torus.r_peak_rg,m=Math.min((D.torus.r_outer_rg-D.torus.r_inner_rg)/2,R*Math.tan(+D.torus.half_opening_angle_deg*Math.PI/180));for(let j=0;j<16;j++){let v=2*Math.PI*j/16;ring(R+m*Math.cos(v),m*Math.sin(v),'#f59e0b',.35)}for(let j=0;j<12;j++){let u=2*Math.PI*j/12,p=[];for(let k=0;k<=64;k++){let v=2*Math.PI*k/64;p.push([(R+m*Math.cos(v))*Math.cos(u),(R+m*Math.cos(v))*Math.sin(u),m*Math.sin(v)])}line(p,'#fbbf24',.8,.28)}let th=+D.cone.opening_angle_deg*Math.PI/180;for(const sg of(D.cone.draw_mode==='north_only'?[1]:[1,-1]))for(let i=0;i<16;i++){let ph=2*Math.PI*i/16;line([[D.cone.r_min_rg*Math.sin(th)*Math.cos(ph),D.cone.r_min_rg*Math.sin(th)*Math.sin(ph),sg*D.cone.r_min_rg*Math.cos(th)],[D.cone.r_max_rg*Math.sin(th)*Math.cos(ph),D.cone.r_max_rg*Math.sin(th)*Math.sin(ph),sg*D.cone.r_max_rg*Math.cos(th)]],'#38bdf8',.8,.3)}let bh=pr([0,0,0]),rh=1+Math.sqrt(1-Math.min(.998,Math.abs(+D.black_hole.spin_a))**2);x.fillStyle='#000';x.beginPath();x.arc(bh[0],bh[1],Math.max(5,rh*bh[3]),0,2*Math.PI);x.fill();screenSites=[];for(const s of D.sites){let p=pr(s.position_xyz_rg);screenSites.push({x:p[0],y:p[1],s});x.fillStyle=s.interaction_steps?'#22c55e':'#a3e635';x.strokeStyle='#dcfce7';x.lineWidth=2;x.beginPath();x.arc(p[0],p[1],7,0,2*Math.PI);x.fill();x.stroke()}}
function near(e){return screenSites.map(q=>({...q,d:Math.hypot(e.clientX-q.x,e.clientY-q.y)})).sort((a,b)=>a.d-b.d)[0]}
c.onmousedown=e=>{drag=true;lx=e.clientX;ly=e.clientY;c.style.cursor='grabbing'};onmouseup=()=>{drag=false;c.style.cursor='grab'};onmousemove=e=>{if(drag){yaw+=(e.clientX-lx)*.008;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-ly)*.008));lx=e.clientX;ly=e.clientY;draw()}let q=near(e);if(q&&q.d<13){tip.style.display='block';tip.style.left=e.clientX+12+'px';tip.style.top=e.clientY+12+'px';tip.innerHTML=`<b>Event ${q.s.geant4_event_id}</b><br>${q.s.interaction_id||'fixture'}<br>r=${q.s.r_rg.toFixed(4)} r_g<br>${q.s.interaction_steps} interaction steps`;c.style.cursor='pointer'}else{tip.style.display='none';if(!drag)c.style.cursor='grab'}};c.onclick=e=>{let q=near(e);if(q&&q.d<13)window.open(q.s.local_view,'_blank','noopener')};c.onwheel=e=>{e.preventDefault();zoom=Math.max(.3,Math.min(8,zoom*Math.exp(-e.deltaY*.001)));draw()};reset.onclick=()=>{yaw=-.72;pitch=.38;zoom=1;draw()};onresize=()=>{let d=devicePixelRatio||1;c.width=innerWidth*d;c.height=innerHeight*d;x.setTransform(d,0,0,d,0,0);draw()};onresize();
</script></body></html>""".replace("__DATA__", _json_for_html(data))


def _local_html(data: dict[str, Any]) -> str:
    return """<!doctype html><html><head><meta charset="utf-8"><title>HADROS3 GEANT4 Local Volume</title>
<style>html,body{margin:0;height:100%;overflow:hidden;background:#07101b;color:#e5eefc;font-family:system-ui}#wrap{height:100%;display:grid;grid-template-columns:minmax(0,1fr) 310px}canvas{width:100%;height:100%;cursor:grab}.side{padding:14px;background:#0f172a;border-left:1px solid #334155;overflow:auto}.legend{font-size:13px;line-height:1.45}.warn{color:#fde68a}select,button{padding:6px;margin:4px 0;width:100%}code{color:#a7f3d0}</style></head><body><div id="wrap"><canvas id="c"></canvas><aside class="side"><h2>Local GEANT4 volume</h2><label>Event</label><select id="event"></select><button id="reset">Reset view</button><div id="info" class="legend"></div></aside></div><script>
const D=__DATA__,c=document.getElementById('c'),x=c.getContext('2d'),sel=document.getElementById('event');let yaw=-.7,pitch=.42,zoom=1,drag=false,lx=0,ly=0;for(const s of D.sites){let o=document.createElement('option');o.value=s.geant4_event_id;o.textContent=`Event ${s.geant4_event_id} — ${s.interaction_id||'fixture'}`;sel.appendChild(o)}let wanted=new URLSearchParams(location.search).get('event');if(wanted!==null)sel.value=wanted;
function rows(){let id=+sel.value;return D.steps.filter(q=>q.geant4_event_id===id)}function rot(p){let cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),a=p[0]*cy-p[1]*sy,b=p[0]*sy+p[1]*cy;return[a,b*cp-p[2]*sp,b*sp+p[2]*cp]}function pr(p){let q=rot(p),s=Math.min(c.clientWidth,c.clientHeight)/(2.7*D.half_size_mm)*zoom;return[c.clientWidth/2+q[0]*s,c.clientHeight/2-q[1]*s,q[2]]}function line(a,b,col,w=1,alpha=1){a=pr(a);b=pr(b);x.save();x.globalAlpha=alpha;x.strokeStyle=col;x.lineWidth=w;x.beginPath();x.moveTo(a[0],a[1]);x.lineTo(b[0],b[1]);x.stroke();x.restore()}
function draw(){x.clearRect(0,0,c.clientWidth,c.clientHeight);x.fillStyle='#07101b';x.fillRect(0,0,c.clientWidth,c.clientHeight);let h=D.half_size_mm,V=[];for(let a of[-h,h])for(let b of[-h,h])for(let d of[-h,h])V.push([a,b,d]);let E=[[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]];for(let e of E)line(V[e[0]],V[e[1]],'#64748b',1,.75);for(let q of rows()){let secondary=q.parent_track_id!==0,col=q.is_interaction?'#fb7185':(secondary?'#fbbf24':'#38bdf8');line(q.pre_position_local_mm,q.post_position_local_mm,col,q.is_interaction?2.5:1.3,.9);if(q.is_interaction){let p=pr(q.post_position_local_mm);x.fillStyle='#fb7185';x.beginPath();x.arc(p[0],p[1],4,0,2*Math.PI);x.fill()}}}
function info(){let id=+sel.value,s=D.sites.find(q=>q.geant4_event_id===id),r=rows(),proc={};for(let q of r)if(q.is_interaction)proc[q.process_name]=(proc[q.process_name]||0)+1;document.getElementById('info').innerHTML=`<p><b>${s.interaction_id||'Fixture event'}</b><br>Macro position: ${s.global_position_available?`r=${s.r_rg.toFixed(6)} r_g, θ=${s.theta_rad.toFixed(6)}, φ=${s.phi_rad.toFixed(6)}`:'not supplied by DIS fixture'}<br>Patch: ${(2*D.half_size_mm).toPrecision(5)} mm per side<br>Material: <code>${D.material}</code><br>Local density at this site: <code>${s.density_g_cm3.toExponential(6)} g/cm³</code><br><span class="${D.density_applied_to_material?'':'warn'}">${D.material_density_policy}</span></p><p>Recorded steps for event: ${s.recorded_steps}<br>Steps drawn: ${r.length}${D.display_steps_truncated?' (display sample)':''}<br>Tracks drawn: ${new Set(r.map(q=>q.track_id)).size}<br>Interactions drawn: ${r.filter(q=>q.is_interaction).length}<br>Escapes: ${s.escaped_particles}</p><p><span style="color:#38bdf8">━ primary</span><br><span style="color:#fbbf24">━ secondary</span><br><span style="color:#fb7185">● physical process</span></p><p>Processes in display: ${Object.entries(proc).map(([k,v])=>`${k} (${v})`).join(', ')||'none — straight transport/boundary only'}</p><p class="warn">Two-scale view: this millimetre box is not the macro BH/toro geometry. It is the local homogeneous tangent-frame approximation centered at the selected site.</p>`}sel.onchange=()=>{info();draw()};c.onmousedown=e=>{drag=true;lx=e.clientX;ly=e.clientY};onmouseup=()=>drag=false;onmousemove=e=>{if(!drag)return;yaw+=(e.clientX-lx)*.008;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-ly)*.008));lx=e.clientX;ly=e.clientY;draw()};c.onwheel=e=>{e.preventDefault();zoom=Math.max(.25,Math.min(10,zoom*Math.exp(-e.deltaY*.001)));draw()};reset.onclick=()=>{yaw=-.7;pitch=.42;zoom=1;draw()};onresize=()=>{let d=devicePixelRatio||1;c.width=c.clientWidth*d;c.height=c.clientHeight*d;x.setTransform(d,0,0,d,0,0);draw()};info();onresize();
</script></body></html>""".replace("__DATA__", _json_for_html(data))
