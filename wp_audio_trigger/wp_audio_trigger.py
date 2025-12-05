#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, math, argparse, datetime as dt, time, socket, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from collections import deque
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import butter, sosfilt
import paho.mqtt.client as mqtt

# ---------- Konstanten ----------
FCS_LOW  = [40,50,63,80,100,125,160,200,250,315]  # Trigger & CSV
FCS_FULL = [31.5,40,50,63,80,100,125,160,200,250,315,400,500,630,800,1000,1250,1600,2000,2500,3150,4000,5000,6300,8000,10000,12500,16000,20000]  # Spektrum
K = 2 ** (1/6)

def band_sos(fc, fs, order=4):
    lo = (fc / K) / (fs/2)
    hi = (fc * K) / (fs/2)
    return butter(order, [lo, hi], btype='bandpass', output='sos')

def a_corr(fc: float) -> float:
    f=float(fc); f2=f*f
    num=(12194**2)*(f2**2)
    den=(f2+20.6**2)*((f2+107.7**2)**0.5)*((f2+737.9**2)**0.5)*(f2+12194**2)
    return 20*math.log10(num/den)+2.0  # +2 dB ≈ IEC-Rundungskorrektur

def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

def load_cal(path):
    off=0.0; band={}
    try:
        with open(path) as f:
            d=json.load(f)
        off=float(d.get("offset_db",0.0))
        band={int(k):float(v) for k,v in d.get("band_corr_db",{}).items()}
        print(f"[wp-audio] Kalibrierung: offset_db={off} band_corr={band}")
    except Exception:
        print(f"[wp-audio] Keine/ungültige Kalibrierdatei: {path} (verwende 0 dB)")
    return off, band

def device_info():
    # Use fixed ID to prevent duplicate devices on container restart
    dev_id = "wp_audio_trigger_addon"
    return {
        "identifiers": [dev_id],
        "manufacturer": "WP Audio",
        "model": "Audio Trigger Add-on",
        "name": "WP Audio Trigger",
    }, dev_id

# ----------- Mini-Web-UI (Ingress) -----------
latest_payload = {"bands": [], "values": [], "weighting": "Z", "ts": now_utc(), "la80": None, "la160": None}

HTML = """<!DOCTYPE html><meta charset=utf-8>
<title>WP Audio</title>
<style>
body{font-family:system-ui,Segoe UI,Arial;margin:16px}
h2{margin:0 0 8px}
#head{display:flex;gap:16px;align-items:baseline;margin-bottom:8px}
#vals{font-weight:600}
#wrap{display:grid;grid-template-columns:repeat(29,1fr);gap:3px;height:46vh;align-items:end;background:#f5f7f9;padding:8px;border-radius:8px}
.bar{background:#4aa3a2;position:relative}
.bar::after{content:attr(data-v);position:absolute;top:-1.4em;left:0;font:11px/1.2 monospace;color:#333}
#labels{display:grid;grid-template-columns:repeat(29,1fr);gap:3px;margin-top:6px}
#labels div{font:11px/1.1 monospace;text-align:center;color:#555}
small{color:#666}
</style>
<div id=head>
  <h2>WP Audio – Live-Spektrum</h2>
  <div id=vals>LA80: – dB(A) · LA160: – dB(A)</div>
</div>
<small>Bewertung: <span id=wgt>–</span> · Zeit: <span id=ts>–</span></small>
<div id=wrap></div>
<div id=labels></div>
<script>
const wrap=document.getElementById('wrap'), labels=document.getElementById('labels');
const vspan=document.getElementById('vals'), wspan=document.getElementById('wgt'), tspan=document.getElementById('ts');
const es=new EventSource('sse'); let minDB=25, maxDB=100;
function init(bands){wrap.innerHTML='';labels.innerHTML='';bands.forEach(b=>{let d=document.createElement('div');d.className='bar';d.style.height='1%';wrap.appendChild(d);let l=document.createElement('div');l.textContent=b;labels.appendChild(l);});}
es.onmessage=(e)=>{let p=JSON.parse(e.data); if(!wrap.children.length) init(p.bands);
  wspan.textContent=p.weighting; tspan.textContent=p.ts;
  if(p.la80!=null && p.la160!=null) vspan.textContent=`LA80: ${p.la80.toFixed(1)} dB(A) · LA160: ${p.la160.toFixed(1)} dB(A)`;
  p.values.forEach((v,i)=>{let h=(v-minDB)/(maxDB-minDB); h=Math.max(0,Math.min(1,h)); let el=wrap.children[i]; el.style.height=(h*100)+'%'; el.setAttribute('data-v',v.toFixed(1));});
};
</script>"""

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/","/index.html","//","/index.htm"):
            self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8")
            self.send_header("Cache-Control","no-store"); self.end_headers(); self.wfile.write(HTML.encode("utf-8")); return
        if self.path.endswith("/sse") or self.path == "/sse":
            self.send_response(200)
            self.send_header("Content-Type","text/event-stream")
            self.send_header("Cache-Control","no-store")
            self.send_header("Connection","keep-alive")
            self.end_headers()
            self.wfile.write(f"data: {json.dumps(latest_payload)}\n\n".encode()); self.wfile.flush()
            try:
                while True:
                    self.wfile.write(b": ping\n\n"); self.wfile.flush(); time.sleep(15)
            except BrokenPipeError:
                return
        self.send_response(404); self.end_headers()

def start_http(port):
    srv = HTTPServer(("0.0.0.0", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[wp-audio] Web-UI läuft auf Ingress (Port {port})")

# --------------- Hauptprogramm ----------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mqtt-host",default="core-mosquitto"); ap.add_argument("--mqtt-port",type=int,default=1883)
    ap.add_argument("--mqtt-user",default=""); ap.add_argument("--mqtt-pass",default="")
    ap.add_argument("--topic-base",default="wp_audio")
    ap.add_argument("--thresh-a80",type=float,default=50.0); ap.add_argument("--thresh-a160",type=float,default=50.0)
    ap.add_argument("--hold-sec",type=int,default=2)
    ap.add_argument("--pre",type=int,default=20); ap.add_argument("--post",type=int,default=30)
    ap.add_argument("--samplerate",type=int,default=48000); ap.add_argument("--device",default="")
    ap.add_argument("--event-dir",default="/media/wp_audio/events"); ap.add_argument("--cal-file",default="/data/calibration.json")
    ap.add_argument("--publish-spectrum", type=lambda v:str(v).lower() in ("1","true","yes"), default=True)
    ap.add_argument("--spectrum-weighting", choices=["A","Z"], default="Z")
    ap.add_argument("--spectrum-interval", type=float, default=1.0)   # <-- jetzt float
    ap.add_argument("--ui-port", type=int, default=8099)
    args=ap.parse_args()

    # Web-UI
    start_http(args.ui_port)

    # MQTT
    connected = {"ok": False}
    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            connected["ok"] = True
            print("[wp-audio] MQTT verbunden")
            client.publish(f"{args.topic_base}/availability", "online", qos=1, retain=True)
            client.publish(f"{args.topic_base}/selftest", "ok", qos=1, retain=True)
            dev, dev_id = device_info(); disc = "homeassistant"
            cfg80 = {"name":"WP OctA 80 Hz","unique_id":f"{dev_id}_octA_80","state_topic":f"{args.topic_base}/octA_80",
                     "unit_of_measurement":"dB(A)","availability_topic":f"{args.topic_base}/availability",
                     "device":dev,"icon":"mdi:chart-bell-curve"}
            cfg160={"name":"WP OctA 160 Hz","unique_id":f"{dev_id}_octA_160","state_topic":f"{args.topic_base}/octA_160",
                    "unit_of_measurement":"dB(A)","availability_topic":f"{args.topic_base}/availability",
                    "device":dev,"icon":"mdi:chart-bell-curve"}
            cfgspec={"name":"WP Spectrum","unique_id":f"{dev_id}_spectrum","state_topic":f"{args.topic_base}/spectrum",
                     "value_template":"{{ value_json.ts }}","json_attributes_topic":f"{args.topic_base}/spectrum",
                     "availability_topic":f"{args.topic_base}/availability","device":dev,"icon":"mdi:waveform"}
            client.publish(f"{disc}/sensor/{dev_id}/octA_80/config",  json.dumps(cfg80),  qos=1, retain=True)
            client.publish(f"{disc}/sensor/{dev_id}/octA_160/config", json.dumps(cfg160), qos=1, retain=True)
            client.publish(f"{disc}/sensor/{dev_id}/spectrum/config", json.dumps(cfgspec), qos=1, retain=True)
        else:
            print(f"[wp-audio] MQTT connect failed rc={rc}")

    def on_disconnect(client, userdata, rc, properties=None):
        connected["ok"] = False
        print(f"[wp-audio] MQTT disconnected rc={rc}")

    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION1, protocol=mqtt.MQTTv311)
    client.on_connect = on_connect; client.on_disconnect = on_disconnect
    client.will_set(f"{args.topic_base}/availability", payload="offline", qos=1, retain=True)
    if args.mqtt_user: client.username_pw_set(args.mqtt_user, args.mqtt_pass)
    client.connect_async(args.mqtt_host, args.mqtt_port, 60); client.loop_start()
    t0=time.time()
    while not connected["ok"] and time.time()-t0<5: time.sleep(0.1)

    # Kalibrierung
    cal_off, band_corr = load_cal(args.cal_file)
    def spl_db(rms): return 20.0*np.log10(max(rms,1e-20)/20e-6)+cal_off

    # Ziel-SR & Filter-Builder
    fs_target = int(args.samplerate)
    # Blockzeit passend zum Publish-Intervall (stabil zwischen 0.10 s und 0.25 s)
    block_sec = max(0.10, min(0.25, float(args.spectrum_interval)))
    def build_filters(fs):
        sos_low  = {fc: band_sos(fc, fs) for fc in FCS_LOW}
        sos_full = {fc: band_sos(fc, fs) for fc in FCS_FULL}
        return sos_low, sos_full
    sos_low, sos_full = build_filters(fs_target)
    a_low    = {fc: a_corr(fc) for fc in FCS_LOW}

    pre_buf=deque(maxlen=max(1,int(args.pre/block_sec)))
    S = {"trig": False, "hold": 0, "post_left": 0, "peak80": -999.0, "peak160": -999.0,
         "cur_dir": None, "event_audio": [], "event_specs": []}
    os.makedirs(args.event_dir, exist_ok=True)

    def start_event():
        S["trig"]=True; S["post_left"]=args.post
        S["cur_dir"]=os.path.join(args.event_dir, now_utc()); os.makedirs(S["cur_dir"], exist_ok=True)
        S["event_audio"]=list(pre_buf); S["event_specs"]=[]; S["peak80"]=-999.0; S["peak160"]=-999.0
        print(f"[wp-audio] Event START {S['cur_dir']}")

    def end_event(current_fs):
        if not S["cur_dir"]: return
        audio=np.concatenate(S["event_audio"],axis=0) if S["event_audio"] else np.zeros(0,np.float32)
        wav=os.path.join(S["cur_dir"],"audio.flac"); sf.write(wav, audio, int(current_fs), format="FLAC")
        csv=os.path.join(S["cur_dir"],"spectrum.csv")
        with open(csv,"w") as f:
            f.write("ts,"+",".join([f"LZ_{fc}" for fc in FCS_LOW])+","+",".join([f"LA_{fc}" for fc in FCS_LOW])+"\n")
            for r in S["event_specs"]:
                f.write(r["ts"]+","+",".join(f"{r['LZ'][fc]:.2f}" for fc in FCS_LOW)+","+
                        ",".join(f"{r['LA'][fc]:.2f}" for fc in FCS_LOW)+"\n")
        payload={"dir":S["cur_dir"],"audio":wav,"spectrum_csv":csv,"start":os.path.basename(S["cur_dir"]),
                 "stop":now_utc(),"peak_A80":round(S["peak80"],2),"peak_A160":round(S["peak160"],2)}
        try: client.publish(f"{args.topic_base}/event", json.dumps(payload), qos=1)
        except: pass
        print(f"[wp-audio] Event ENDE {S['cur_dir']} (PeakA80={S['peak80']:.1f}, PeakA160={S['peak160']:.1f})")
        S.update({"trig":False,"cur_dir":None,"event_audio":[],"event_specs":[]})

    print(f"[wp-audio] Input-Device: {args.device if args.device else '(Default/Pulse)'}  SR={fs_target}")

    # -------- Audio-Stream öffnen (Default/Pulse oder explizit) --------
    devname = args.device.strip()

    def try_open(device, fs, block):
        return sd.InputStream(device=device if device else None,
                              channels=1,
                              samplerate=(fs if fs>0 else None),
                              blocksize=(block if block>0 else None),
                              dtype='float32')

    open_plan = []
    if devname:
        open_plan = [
            {"device": devname, "fs": fs_target, "block": int(fs_target*block_sec)},
            {"device": devname, "fs": 44100,    "block": int(44100*block_sec)},
            {"device": devname, "fs": 0,        "block": 0},
        ]
    else:
        open_plan = [
            {"device": None,  "fs": fs_target, "block": int(fs_target*block_sec)},
            {"device": None,  "fs": 0,         "block": 0},
        ]
        if os.environ.get("PULSE_SERVER"):
            open_plan += [
                {"device": "pulse", "fs": fs_target, "block": int(fs_target*block_sec)},
                {"device": "pulse", "fs": 0,         "block": 0},
            ]

    stream = None
    current_fs = fs_target
    block_samples = int(fs_target * block_sec)

    for i,op in enumerate(open_plan, 1):
        try:
            print(f"[wp-audio] Versuch {i}: device={op['device']!r} fs={op['fs']} block={op['block']}")
            stream = try_open(op["device"], op["fs"], op["block"])
            stream.start()
            current_fs = int(stream.samplerate)
            if current_fs != fs_target:
                print(f"[wp-audio] Hinweis: tatsächliche fs={current_fs} Hz – Filter werden angepasst.")
                sos_low, sos_full = build_filters(current_fs)
            block_samples = int(current_fs * block_sec)
            break
        except Exception as e:
            print(f"[wp-audio] Open fehlgeschlagen: {e}")
            stream = None
            time.sleep(0.2)

    if stream is None:
        raise RuntimeError("Konnte keinen Audio-Stream öffnen (alle Versuche fehlgeschlagen).")

    # -------- Haupt-Loop --------
    last_spec_pub = 0.0  # monotonic time
    try:
        while True:
            x, _ = stream.read(block_samples)
            if x.ndim == 2:
                x = x[:,0]
            else:
                x = x.reshape(-1)
            LZ={}; LA={}
            for fc,sos in sos_low.items():
                y=sosfilt(sos,x)
                lz=spl_db(np.sqrt(np.mean(y*y)))+band_corr.get(fc,0.0)
                la=lz+a_low[fc]
                LZ[fc]=lz; LA[fc]=la

            la80=LA[80]; la160=LA[160]

            # MQTT Live
            try:
                client.publish(f"{args.topic_base}/octA_80", f"{la80:.2f}", qos=0)
                client.publish(f"{args.topic_base}/octA_160", f"{la160:.2f}", qos=0)
            except: pass

            # UI Snapshot
            latest_payload.update({"la80": float(la80), "la160": float(la160)})

            # Pre-Buffer / Event-Aufnahme
            pre_buf.append(x.copy())
            rec={"ts":now_utc(),"LZ":LZ,"LA":LA}

            # volles Spektrum (optional, schneller dank kürzerer Blöcke)
            nowm = time.monotonic()
            if args.publish_spectrum and (nowm - last_spec_pub) >= float(args.spectrum_interval):
                vals=[]
                for fc,sos in sos_full.items():
                    y=sosfilt(sos,x)
                    lz=spl_db(np.sqrt(np.mean(y*y)))+band_corr.get(fc,0.0)
                    v = lz + (a_corr(fc) if args.spectrum_weighting=="A" else 0.0)
                    vals.append(v)
                payload={"bands":[str(int(fc)) if fc>=100 else str(fc) for fc in FCS_FULL],
                         "values":vals,"weighting":args.spectrum_weighting,"ts":now_utc()}
                latest_payload.update(payload)
                try: client.publish(f"{args.topic_base}/spectrum", json.dumps(payload), qos=0)
                except: pass
                last_spec_pub = nowm

            # Trigger-Logik
            over=(la80>=args.thresh_a80) or (la160>=args.thresh_a160)
            if not S["trig"]:
                if over:
                    S["hold"]+=block_sec
                    if S["hold"]>=args.hold_sec:
                        S["trig"]=True; S["post_left"]=args.post
                        S["cur_dir"]=os.path.join(args.event_dir, now_utc()); os.makedirs(S["cur_dir"], exist_ok=True)
                        S["event_audio"]=list(pre_buf); S["event_specs"]=[]; S["peak80"]=-999.0; S["peak160"]=-999.0
                        print(f"[wp-audio] Event START {S['cur_dir']}")
                        S["hold"]=0
                else:
                    S["hold"]=0
            else:
                S["event_audio"].append(x.copy()); S["event_specs"].append(rec)
                S["peak80"]=max(S["peak80"],la80); S["peak160"]=max(S["peak160"],la160)
                if over:
                    S["post_left"]=args.post
                else:
                    S["post_left"]-=block_sec
                    if S["post_left"]<=0:
                        end_event(current_fs)

    finally:
        try:
            stream.stop(); stream.close()
        except: pass

if __name__=="__main__":
    main()
