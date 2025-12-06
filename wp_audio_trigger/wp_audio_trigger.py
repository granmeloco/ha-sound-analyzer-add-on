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
FCS_LOW  = [40,50,63,80,100,125,160,200,250,315]  # Trigger & CSV (will be dynamically replaced)
FCS_FULL = [31.5,40,50,63,80,100,125,160,200,250,315,400,500,630,800,1000,1250,1600,2000,2500,3150,4000,5000,6300,8000,10000,12500,16000,20000]  # Spektrum (will be dynamically replaced)
K = 2 ** (1/6)

def get_octave_bands(band_type, min_freq=31.5, max_freq=20000):
    """Generate octave band center frequencies based on band type."""
    if band_type == "1octave":
        # 1-octave bands: 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000
        base_freqs = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    elif band_type == "2octave":
        # 1/2-octave bands
        base_freqs = [31.5, 44.7, 63, 89.1, 125, 177, 250, 354, 500, 707, 1000, 1414, 2000, 2828, 4000, 5657, 8000, 11314, 16000]
    else:  # "3octave" or default
        # 1/3-octave bands (full range)
        base_freqs = [31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000]
    
    # Filter by min/max frequency
    return [f for f in base_freqs if min_freq <= f <= max_freq]

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
trigger_config = {"triggers": []}  # Will be populated from args

HTML = """<!DOCTYPE html><meta charset=utf-8>
<title>Audio Analyzer settings</title>
<style>
body{font-family:Arial,sans-serif;margin:0;padding:20px;background:#fff;max-width:1000px;margin:0 auto}
h1{text-align:center;font-size:22px;margin-bottom:20px;font-weight:normal}
.section{background:#e8e8e8;padding:16px;margin-bottom:16px;border-radius:4px}
.section-title{font-weight:bold;margin-bottom:16px;font-size:14px}
.row{display:flex;gap:10px;margin-bottom:10px;align-items:center}
.radio-group{display:flex;align-items:center;gap:6px}
.radio-group input[type=radio]{margin:0}
.radio-group label{margin:0;font-size:13px;font-weight:normal}
.field-label{font-size:13px;white-space:nowrap}
input[type=text],input[type=number]{padding:5px 8px;border:1px solid #999;font-size:13px;width:100px;background:white}
.freq-range{font-size:11px;color:#666}
.trigger-grid{display:grid;grid-template-columns:30px 160px 200px 160px;gap:20px;align-items:center;margin-bottom:10px}
.trigger-grid.header{font-weight:normal;font-size:12px;margin-bottom:12px}
.trigger-grid input{width:150px}
.freq-input-row{display:flex;gap:25px;align-items:center;margin-top:10px}
.calib-grid{display:flex;gap:12px;align-items:flex-start}
.calib-col{display:flex;flex-direction:column;align-items:center;gap:4px}
.calib-col input{width:70px;text-align:center}
.calib-col span{font-size:11px}
.calib-label{font-size:13px;margin-right:8px;align-self:center}
button{background:#17a2b8;color:white;border:none;padding:8px 24px;border-radius:4px;cursor:pointer;font-size:13px;margin-top:10px}
button:hover{background:#138496}
#status{margin-top:10px;padding:8px;border-radius:4px;font-size:13px;display:none}
#status.success{background:#d4edda;color:#155724;display:block}
#status.error{background:#f8d7da;color:#721c24;display:block}
</style>
<h1>Audio Analyzer settings</h1>

<div class=section>
  <div class=section-title>Spectrum analyzer settings</div>
  <div class=row>
    <div class=radio-group>
      <input type=radio name=bands id=b1oct value=1octave>
      <label for=b1oct>1-octave bands</label>
    </div>
    <span class=freq-range>(31.5 - 63 - 125 - 250 - 500 - 1000 - 2000 - 4000 - 8000 - 16000)</span>
  </div>
  <div class=row>
    <div class=radio-group>
      <input type=radio name=bands id=b2oct value=2octave>
      <label for=b2oct>1/2-octave bands</label>
    </div>
    <span class=freq-range>(31.5 - 44.7 - 63 - 89.4 - 125 - 177 - 250 - 355 - 500 - 707 - 1000 - 1414 - 2000 - 2828 - 4000 - 5657 - 8000 - 11314 - 16000)</span>
  </div>
  <div class=row>
    <div class=radio-group>
      <input type=radio name=bands id=b3oct value=3octave checked>
      <label for=b3oct>1/3-octave bands</label>
    </div>
    <span class=freq-range>(1 - 1.25 - 1.6 - 2 - 2.5 - 3.15 - 4 - 5 - 6.3 - 8 - 10 - 12.5 - 16 - 20 - 25 - 31.5 - 40 - 50 - 63 - 80 - 100 - 125 - 160 - 200 - 250 - 315 - 400 - 500 - 630 - 800 - 1000 - 1250 - 1600 - 2000 - 2500 - 3150 - 4000 - 5000 - 6300 - 8000 - 10000 - 12500 - 16000 - 20000 Hz)</span>
  </div>
  <div class=freq-input-row>
    <span class=field-label>Min. Frequency [Hz]</span>
    <input type=number id=minFreq value=31.5 step=0.1>
    <span class=field-label>Max. Frequency [Hz]</span>
    <input type=number id=maxFreq value=20000 step=0.1>
  </div>
</div>

<div class=section>
  <div class=section-title>Audio trigger settings</div>
  <div class=trigger-grid class=header>
    <span></span>
    <span>Frequency [Hz]</span>
    <span>Min. amplitude [dBA]</span>
    <span>Min. duration [s]</span>
  </div>
  <div class=trigger-grid>
    <span class=field-label>1.</span>
    <input type=number id=t1freq placeholder="">
    <input type=number id=t1amp placeholder="" step=0.1>
    <input type=number id=t1dur placeholder="" step=0.1>
  </div>
  <div class=trigger-grid>
    <span class=field-label>2.</span>
    <input type=number id=t2freq placeholder="">
    <input type=number id=t2amp placeholder="" step=0.1>
    <input type=number id=t2dur placeholder="" step=0.1>
  </div>
  <div class=trigger-grid>
    <span class=field-label>3.</span>
    <input type=number id=t3freq placeholder="">
    <input type=number id=t3amp placeholder="" step=0.1>
    <input type=number id=t3dur placeholder="" step=0.1>
  </div>
  <div class=trigger-grid>
    <span class=field-label>4.</span>
    <input type=number id=t4freq placeholder="">
    <input type=number id=t4amp placeholder="" step=0.1>
    <input type=number id=t4dur placeholder="" step=0.1>
  </div>
  <div class=row>
    <span class=field-label>Logical constraint:</span>
    <div class=radio-group>
      <input type=radio name=logic id=logicAnd value=AND checked>
      <label for=logicAnd>AND</label>
    </div>
    <div class=radio-group>
      <input type=radio name=logic id=logicOr value=OR>
      <label for=logicOr>OR</label>
    </div>
  </div>
</div>

<div class=section>
  <div class=section-title>Recorded sound file settings</div>
  <div class=row>
    <span class=field-label>Storage location</span>
    <input type=text id=storageLocation style="width:400px" placeholder="/media/wp_audio/events">
  </div>
  <div class=row>
    <span class=field-label>Recording length [s]</span>
    <input type=number id=recLength placeholder="" step=1>
  </div>
</div>

<div class=section>
  <div class=section-title>Mic calibration settings</div>
  <div class=calib-grid>
    <span class=calib-label>+/-</span>
    <div class=calib-col>
      <input type=text id=cal63 placeholder="">
      <span>63Hz</span>
    </div>
    <div class=calib-col>
      <input type=text id=cal250 placeholder="">
      <span>250Hz</span>
    </div>
    <div class=calib-col>
      <input type=text id=cal1000 placeholder="">
      <span>1000Hz</span>
    </div>
    <div class=calib-col>
      <input type=text id=cal4000 placeholder="">
      <span>4000Hz</span>
    </div>
    <div class=calib-col>
      <input type=text id=cal16000 placeholder="">
      <span>16000Hz</span>
    </div>
  </div>
</div>

<button onclick="saveConfig()">Save Configuration</button>
<div id=status></div>

<script>
const statusDiv=document.getElementById('status');

// Load configuration
fetch('api/config').then(r=>r.json()).then(data=>{
  document.getElementById('minFreq').value=data.minFreq||31.5;
  document.getElementById('maxFreq').value=data.maxFreq||20000;
  if(data.bands==='1octave') document.getElementById('b1oct').checked=true;
  else if(data.bands==='2octave') document.getElementById('b2oct').checked=true;
  else document.getElementById('b3oct').checked=true;
  
  for(let i=1;i<=4;i++){
    document.getElementById(`t${i}freq`).value=data.triggers[i-1]?.freq||'';
    document.getElementById(`t${i}amp`).value=data.triggers[i-1]?.amp||'';
    document.getElementById(`t${i}dur`).value=data.triggers[i-1]?.duration||'';
  }
  
  if(data.logic==='OR') document.getElementById('logicOr').checked=true;
  else document.getElementById('logicAnd').checked=true;
  
  document.getElementById('storageLocation').value=data.storageLocation||'/media/wp_audio/events';
  document.getElementById('recLength').value=data.recLength||'';
  
  document.getElementById('cal63').value=data.calibration?.cal63||'';
  document.getElementById('cal250').value=data.calibration?.cal250||'';
  document.getElementById('cal1000').value=data.calibration?.cal1000||'';
  document.getElementById('cal4000').value=data.calibration?.cal4000||'';
  document.getElementById('cal16000').value=data.calibration?.cal16000||'';
}).catch(e=>console.error('Load error:',e));

function saveConfig(){
  const config={
    bands:document.getElementById('b1oct').checked?'1octave':document.getElementById('b2oct').checked?'2octave':'3octave',
    minFreq:parseFloat(document.getElementById('minFreq').value),
    maxFreq:parseFloat(document.getElementById('maxFreq').value),
    triggers:[
      {freq:parseInt(document.getElementById('t1freq').value)||0,amp:parseFloat(document.getElementById('t1amp').value)||0,duration:parseFloat(document.getElementById('t1dur').value)||0},
      {freq:parseInt(document.getElementById('t2freq').value)||0,amp:parseFloat(document.getElementById('t2amp').value)||0,duration:parseFloat(document.getElementById('t2dur').value)||0},
      {freq:parseInt(document.getElementById('t3freq').value)||0,amp:parseFloat(document.getElementById('t3amp').value)||0,duration:parseFloat(document.getElementById('t3dur').value)||0},
      {freq:parseInt(document.getElementById('t4freq').value)||0,amp:parseFloat(document.getElementById('t4amp').value)||0,duration:parseFloat(document.getElementById('t4dur').value)||0}
    ],
    logic:document.getElementById('logicOr').checked?'OR':'AND',
    storageLocation:document.getElementById('storageLocation').value,
    recLength:parseInt(document.getElementById('recLength').value)||0,
    calibration:{
      cal63:document.getElementById('cal63').value,
      cal250:document.getElementById('cal250').value,
      cal1000:document.getElementById('cal1000').value,
      cal4000:document.getElementById('cal4000').value,
      cal16000:document.getElementById('cal16000').value
    }
  };
  
  fetch('api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(config)})
  .then(r=>r.json())
  .then(data=>{
    if(data.success){
      statusDiv.style.display='block';
      if(data.restarting){
        statusDiv.textContent='Configuration saved! Add-on is restarting to apply changes...';
      }else{
        statusDiv.textContent='Configuration saved successfully!';
      }
      statusDiv.className='success';
      setTimeout(()=>{statusDiv.style.display='none';statusDiv.className='';},5000);
    }else{
      statusDiv.style.display='block';
      statusDiv.textContent='Error saving configuration';
      statusDiv.className='error';
    }
  }).catch(e=>{statusDiv.style.display='block';statusDiv.textContent='Connection error: '+e.message;statusDiv.className='error';});
}
</script>"""

class H(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Log HTTP requests for debugging
        print(f"[wp-audio] HTTP {format % args}", flush=True)
    
    def do_GET(self):
        print(f"[wp-audio] GET request: {self.path}", flush=True)
        try:
            if self.path in ("/","/index.html","//","/index.htm"):
                self.send_response(200)
                self.send_header("Content-Type","text/html; charset=utf-8")
                self.send_header("Cache-Control","no-store")
                self.end_headers()
                self.wfile.write(HTML.encode("utf-8"))
                return
            if self.path == "/api/triggers":
                self.send_response(200)
                self.send_header("Content-Type","application/json")
                self.send_header("Cache-Control","no-store")
                self.end_headers()
                self.wfile.write(json.dumps(trigger_config).encode("utf-8"))
                return
            if self.path == "/api/config":
                self.send_response(200)
                self.send_header("Content-Type","application/json")
                self.send_header("Cache-Control","no-store")
                self.end_headers()
                # Load saved config or return defaults
                config_file = "/data/analyzer_config.json"
                if os.path.exists(config_file):
                    with open(config_file, "r") as f:
                        config = json.load(f)
                else:
                    config = {
                        "bands": "3octave",
                        "minFreq": 31.5,
                        "maxFreq": 20000,
                        "triggers": trigger_config.get("triggers", []),
                        "logic": "AND",
                        "storageLocation": "/media/wp_audio/events",
                        "recLength": 60,
                        "calibration": {}
                    }
                self.wfile.write(json.dumps(config).encode("utf-8"))
                return
            if self.path.endswith("/sse") or self.path == "/sse":
                self.send_response(200)
                self.send_header("Content-Type","text/event-stream")
                self.send_header("Cache-Control","no-store")
                self.send_header("Connection","keep-alive")
                self.end_headers()
                self.wfile.write(f"data: {json.dumps(latest_payload)}\n\n".encode())
                self.wfile.flush()
                try:
                    while True:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        time.sleep(15)
                except (BrokenPipeError, ConnectionResetError):
                    return
            self.send_response(404)
            self.end_headers()
        except Exception as e:
            print(f"[wp-audio] HTTP GET error: {e}")
            try:
                self.send_response(500)
                self.end_headers()
            except:
                pass
    
    def do_POST(self):
        print(f"[wp-audio] POST request: {self.path}", flush=True)
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            
            if self.path == "/api/triggers":
                try:
                    data = json.loads(body)
                    print(f"[wp-audio] Received trigger save request: {len(body)} bytes", flush=True)
                    # Save just triggers
                    trigger_file = "/data/trigger_config.json"
                    os.makedirs(os.path.dirname(trigger_file), exist_ok=True)
                    with open(trigger_file, "w") as f:
                        json.dump(data, f, indent=2)
                    print(f"[wp-audio] Trigger configuration saved to {trigger_file}: {len(data)} triggers", flush=True)
                    self.send_response(200)
                    self.send_header("Content-Type","application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": True}).encode("utf-8"))
                    return
                except Exception as e:
                    print(f"[wp-audio] Trigger save error: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                self.send_response(400)
                self.send_header("Content-Type","application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e) if 'e' in locals() else "Unknown error"}).encode("utf-8"))
                return
            
            if self.path == "/api/config":
                try:
                    data = json.loads(body)
                    print(f"[wp-audio] Received config save request: {len(body)} bytes", flush=True)
                    # Save entire configuration
                    config_file = "/data/analyzer_config.json"
                    os.makedirs(os.path.dirname(config_file), exist_ok=True)
                    with open(config_file, "w") as f:
                        json.dump(data, f, indent=2)
                    print(f"[wp-audio] Configuration saved to {config_file}: {len(data.get('triggers', []))} triggers, logic={data.get('logic')}", flush=True)
                    
                    # Trigger add-on restart to apply configuration changes via Supervisor API
                    import threading
                    import urllib.request
                    def restart_addon():
                        import time
                        time.sleep(2)  # Wait 2 seconds to allow response to be sent
                        try:
                            token = os.environ.get("SUPERVISOR_TOKEN", "")
                            if token:
                                print("[wp-audio] Restarting add-on via Supervisor API...", flush=True)
                                req = urllib.request.Request(
                                    "http://supervisor/addons/self/restart",
                                    method="POST",
                                    headers={"Authorization": f"Bearer {token}"}
                                )
                                urllib.request.urlopen(req, timeout=5)
                                print("[wp-audio] Restart command sent successfully", flush=True)
                            else:
                                print("[wp-audio] WARNING: SUPERVISOR_TOKEN not available, cannot auto-restart", flush=True)
                        except Exception as e:
                            print(f"[wp-audio] Restart failed: {e}", flush=True)
                    threading.Thread(target=restart_addon, daemon=True).start()
                    
                    self.send_response(200)
                    self.send_header("Content-Type","application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": True, "restarting": True}).encode("utf-8"))
                    return
                except Exception as e:
                    print(f"[wp-audio] Config save error: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                self.send_response(400)
                self.send_header("Content-Type","application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e) if 'e' in locals() else "Unknown error"}).encode("utf-8"))
                return
            
            self.send_response(404)
            self.end_headers()
        except Exception as e:
            print(f"[wp-audio] HTTP POST error: {e}")
            try:
                self.send_response(500)
                self.end_headers()
            except:
                pass

def start_http(port):
    try:
        srv = HTTPServer(("0.0.0.0", port), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f"[wp-audio] Web-UI läuft auf Ingress (Port {port})", flush=True)
    except Exception as e:
        print(f"[wp-audio] FEHLER beim Starten des HTTP-Servers: {e}", flush=True)
        raise

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
    # Trigger configuration arguments
    for i in range(1, 6):
        ap.add_argument(f"--trigger-freq-{i}", type=int, default=0)
        ap.add_argument(f"--trigger-amp-{i}", type=float, default=0.0)
    args=ap.parse_args()

    # Load full analyzer configuration
    global trigger_config
    analyzer_config = {
        "bands": "3octave",  # default 1/3-octave
        "minFreq": 31.5,
        "maxFreq": 20000,
        "triggers": [],
        "logic": "OR",
        "storageLocation": args.event_dir,
        "recLength": args.post,
        "calibration": {"cal63": 0, "cal250": 0, "cal1000": 0, "cal4000": 0, "cal16000": 0}
    }
    
    # Load from persistent file if exists
    config_file = "/data/analyzer_config.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                saved_config = json.load(f)
                analyzer_config.update(saved_config)
                print(f"[wp-audio] Analyzer configuration loaded: {analyzer_config['bands']} bands, {len(analyzer_config['triggers'])} triggers, logic={analyzer_config['logic']}", flush=True)
        except Exception as e:
            print(f"[wp-audio] Error loading analyzer config: {e}", flush=True)
    
    # Legacy: initialize trigger_config from command-line args if no saved config
    trigger_config["triggers"] = analyzer_config.get("triggers", [
        {"freq": getattr(args, f"trigger_freq_{i}"), "amp": getattr(args, f"trigger_amp_{i}")}
        for i in range(1, 6)
    ])

    # Web-UI
    print(f"[wp-audio] Starting HTTP server on port {args.ui_port}...", flush=True)
    start_http(args.ui_port)
    print(f"[wp-audio] HTTP server started successfully", flush=True)

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

    # Load configured bands dynamically
    global FCS_LOW, FCS_FULL
    bands_config = analyzer_config.get("bands", "3octave")
    min_freq = analyzer_config.get("minFreq", 31.5)
    max_freq = analyzer_config.get("maxFreq", 20000)
    FCS_LOW = get_octave_bands(bands_config, min_freq, max_freq)
    FCS_FULL = FCS_LOW  # Use same bands for both trigger and spectrum
    print(f"[wp-audio] Using {bands_config} bands: {FCS_LOW}", flush=True)
    
    # Kalibrierung from config
    cal_off, band_corr = load_cal(args.cal_file)
    calibration = analyzer_config.get("calibration", {})
    if calibration:
        # Apply calibration offsets from UI config
        for freq_key, offset_str in calibration.items():
            try:
                freq = int(freq_key.replace("cal", ""))
                offset = float(offset_str) if offset_str else 0.0
                if offset != 0.0:
                    band_corr[freq] = band_corr.get(freq, 0.0) + offset
                    print(f"[wp-audio] Calibration: {freq} Hz += {offset:.2f} dB", flush=True)
            except: pass
    
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

            # Legacy support for 80/160 Hz if they exist in bands
            la80 = LA.get(80, 0.0)
            la160 = LA.get(160, 0.0)

            # MQTT Live (publish first band and any legacy frequencies)
            try:
                if 80 in LA:
                    client.publish(f"{args.topic_base}/octA_80", f"{la80:.2f}", qos=0)
                if 160 in LA:
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

            # Dynamic Trigger Evaluation
            triggers = analyzer_config.get("triggers", [])
            logic = analyzer_config.get("logic", "OR")
            trigger_results = []
            
            for trig in triggers:
                freq = trig.get("freq", 0)
                amp = trig.get("amp", 0)
                if freq > 0 and amp > 0 and freq in LA:
                    trigger_results.append(LA[freq] >= amp)
            
            # Apply logic (OR = any trigger, AND = all triggers)
            if logic == "AND":
                over = all(trigger_results) if trigger_results else False
            else:  # OR (default)
                over = any(trigger_results) if trigger_results else False
            # Use configured storage location and recording length
            storage_dir = analyzer_config.get("storageLocation", args.event_dir)
            rec_length = analyzer_config.get("recLength", args.post)
            
            if not S["trig"]:
                if over:
                    S["hold"]+=block_sec
                    if S["hold"]>=args.hold_sec:
                        S["trig"]=True; S["post_left"]=rec_length
                        S["cur_dir"]=os.path.join(storage_dir, now_utc()); os.makedirs(S["cur_dir"], exist_ok=True)
                        S["event_audio"]=list(pre_buf); S["event_specs"]=[]; S["peak80"]=-999.0; S["peak160"]=-999.0
                        print(f"[wp-audio] Event START {S['cur_dir']}")
                        S["hold"]=0
                else:
                    S["hold"]=0
            else:
                S["event_audio"].append(x.copy()); S["event_specs"].append(rec)
                S["peak80"]=max(S["peak80"],la80); S["peak160"]=max(S["peak160"],la160)
                if over:
                    S["post_left"]=rec_length
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
