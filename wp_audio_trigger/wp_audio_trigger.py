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

def c_corr(fc: float) -> float:
    f=float(fc); f2=f*f
    num=(12194**2)*(f2)
    den=(f2+20.6**2)*(f2+12194**2)
    return 20*math.log10(num/den)+0.06  # C-weighting correction

def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

def load_cal(path):
    off=0.0; band={}
    try:
        with open(path) as f:
            d=json.load(f)
        off=float(str(d.get("offset_db",0.0)).replace(",","."))
        # Preserve decimal center frequencies (e.g., 31.5 Hz)
        band = {}
        for k, v in d.get("band_corr_db", {}).items():
            try:
                kf = float(str(k).replace(",","."))
                vf = float(str(v).replace(",","."))
                band[kf] = vf
            except Exception:
                print(f"[wp-audio] WARNING: invalid calibration entry {k}={v}: ignored (must be a number, dot or comma allowed)")
        print(f"[wp-audio] Kalibrierung: offset_db={off} band_corr={band}")
    except Exception:
        print(f"[wp-audio] Keine/ungültige Kalibrierdatei: {path} (verwende 0 dB)")
    return off, band

def device_info():
    # Use fixed ID to prevent duplicate devices on container restart
    dev_id = "audio_trigger"
    return {
        "identifiers": [dev_id],
        "manufacturer": "Audio Trigger",
        "model": "Audio Trigger Add-on",
        "name": "Audio Trigger",
    }, dev_id

# ----------- Mini-Web-UI (Ingress) -----------
latest_payload = {"bands": [], "values": [], "weighting": "Z", "ts": now_utc(), "la80": None, "la160": None}
trigger_config = {"triggers": []}  # Will be populated from args
analyzer_config = {}  # Will be populated in main()

HTML = """<!DOCTYPE html><meta charset=utf-8>
<title>Sound Analyzer and trigger based Recorder</title>
<style>
body{font-family:Arial,sans-serif;margin:0;padding:20px;background:#fff;max-width:1000px;margin:0 auto}
h1{text-align:center;font-size:22px;margin-bottom:20px;font-weight:bold}
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
.trigger-grid.header{font-weight:normal;font-size:13px;margin-bottom:12px}
.trigger-grid input{width:150px}
.freq-input-row{display:flex;gap:25px;align-items:center;margin-top:10px}
.calib-grid{display:flex;gap:12px;align-items:flex-start}
.calib-col{display:flex;flex-direction:column;align-items:center;gap:4px}
.calib-col input{width:56px;text-align:center}
.calib-col span{font-size:11px}
.calib-label{font-size:13px;margin-right:8px;align-self:center}
button{background:#17a2b8;color:white;border:none;padding:8px 24px;border-radius:4px;cursor:pointer;font-size:13px;margin-top:10px}
button:hover{background:#138496}
.browse-btn{padding:5px 12px;margin-top:0;margin-left:8px;font-size:12px}
#status{margin-top:10px;padding:8px;border-radius:4px;font-size:13px;display:none}
#status.success{background:#d4edda;color:#155724;display:block}
#status.error{background:#f8d7da;color:#721c24;display:block}
</style>
<h1>Sound Analyzer and trigger based Recorder</h1>

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
        <span class=freq-range>(1 - 1.25 - 1.6 - 2 - 2.5 - 3.15 - 4.5 - 6.3 - 8 - 10 - 12.5 - 16 - 20 - 25 - 31.5 - 40 - 50 - 63 - 80 - 100 - 125 - 160 - 200 - 250 - 315 - 400 - 500 - 630 - 800 - 1000 - 1250 - 1600 - 2000 - 2500 - 3150 - 4000 - 5000 - 6300 - 8000 - 10000 - 12500 - 16000 - 20000 Hz)</span>
    </div>
  <div class=freq-input-row>
    <span class=field-label>Min. Frequency [Hz]</span>
    <input type=number id=minFreq value=31.5 step=0.1>
    <span class=field-label>Max. Frequency [Hz]</span>
    <input type=number id=maxFreq value=20000 step=0.1>
  </div>
    <div class=row style="margin-top:10px">
        <span class=field-label>Publish interval [s]</span>
        <input type=number id=publishInterval value=1 step=0.1 min=0.1>
    </div>
    <div class=row>
        <span class=field-label>Averaging period [s]</span>
        <input type=number id=averagingPeriod value=2 step=0.1 min=0.1>
    </div>
  <div class=row>
    <span class=field-label>dB weighting</span>
    <select id=dbWeighting style="padding:5px 8px;border:1px solid #999;font-size:13px;background:white">
      <option value="A" selected>A</option>
      <option value="Z">Z</option>
      <option value="C">C</option>
    </select>
  </div>
</div>

<div class=section>
    <div class=section-title>Audio trigger settings</div>
<div class=section>
    <div class=section-title>Recorded signal settings</div>
    <div style="margin-bottom:10px">Enter sensor variable names to record, separated by commas (e.g. temperature,humidity,pressure):</div>
    <div class=row>
        <span class=field-label>Sensors</span>
        <input type="text" id="sensorText" style="min-width:220px;padding:5px 8px;border:1px solid #999;font-size:13px;background:white" placeholder="temperature,humidity">
    </div>
</div>

  <div class=trigger-grid class=header>
    <span></span>
    <span>Frequency [Hz]</span>
    <span>Min. amplitude [dBA]</span>
    <span>Min. duration [s]</span>
  </div>
  <div class=trigger-grid>
    <span class=field-label>1.</span>
    <select id=t1freq style="padding:5px 8px;border:1px solid #999;font-size:13px;width:150px;background:white"></select>
    <input type=number id=t1amp placeholder="" step=0.1>
    <input type=number id=t1dur placeholder="" step=0.1>
  </div>
  <div class=trigger-grid>
    <span class=field-label>2.</span>
    <select id=t2freq style="padding:5px 8px;border:1px solid #999;font-size:13px;width:150px;background:white"></select>
    <input type=number id=t2amp placeholder="" step=0.1>
    <input type=number id=t2dur placeholder="" step=0.1>
  </div>
  <div class=trigger-grid>
    <span class=field-label>3.</span>
    <select id=t3freq style="padding:5px 8px;border:1px solid #999;font-size:13px;width:150px;background:white"></select>
    <input type=number id=t3amp placeholder="" step=0.1>
    <input type=number id=t3dur placeholder="" step=0.1>
  </div>
  <div class=trigger-grid>
    <span class=field-label>4.</span>
    <select id=t4freq style="padding:5px 8px;border:1px solid #999;font-size:13px;width:150px;background:white"></select>
    <input type=number id=t4amp placeholder="" step=0.1>
    <input type=number id=t4dur placeholder="" step=0.1>
  </div>
    <div class=trigger-grid>
        <span class=field-label>5.</span>
        <span>
            Prominent frequency
            <select id=promFreq1 style="padding:5px 8px;border:1px solid #999;font-size:13px;width:100px;background:white"></select>
            or
            <select id=promFreq2 style="padding:5px 8px;border:1px solid #999;font-size:13px;width:100px;background:white"></select>
            for <input type=number id=promDur placeholder="" step=0.1 style="width:60px"> s
        </span>
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
    <button class=browse-btn onclick="browseFolder()">Browse...</button>
  </div>
  <div class=row>
    <span class=field-label>Pre and Post Buffer Time [s]</span>
    <input type=number id=preBuffer placeholder="" step=1>
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
                <input type=text id=cal31_5 placeholder="">
                <span>31.5Hz</span>
            </div>
    <div class=calib-col>
      <input type=text id=cal63 placeholder="">
      <span>63Hz</span>
    </div>
            <div class=calib-col>
                <input type=text id=cal125 placeholder="">
                <span>125Hz</span>
            </div>
    <div class=calib-col>
      <input type=text id=cal250 placeholder="">
      <span>250Hz</span>
    </div>
            <div class=calib-col>
                <input type=text id=cal500 placeholder="">
                <span>500Hz</span>
            </div>
    <div class=calib-col>
      <input type=text id=cal1000 placeholder="">
      <span>1000Hz</span>
    </div>
            <div class=calib-col>
                <input type=text id=cal2000 placeholder="">
                <span>2000Hz</span>
            </div>
    <div class=calib-col>
      <input type=text id=cal4000 placeholder="">
      <span>4000Hz</span>
    </div>
            <div class=calib-col>
                <input type=text id=cal8000 placeholder="">
                <span>8000Hz</span>
            </div>
    <div class=calib-col>
      <input type=text id=cal16000 placeholder="">
      <span>16000Hz</span>
    </div>
  </div>
        <div style="font-size:13px;margin-top:6px;color:#333">1-octave calibration will automatically interpolate to selected analyzed octave</div>
</div>

<button onclick="saveConfig()">Save Configuration</button>
<div id=status></div>

<script>
const statusDiv=document.getElementById('status');

// Frequency band definitions
const freqBands={
  '1octave':[31.5,63,125,250,500,1000,2000,4000,8000,16000],
  '2octave':[31.5,44.7,63,89.1,125,177,250,354,500,707,1000,1414,2000,2828,4000,5657,8000,11314,16000],
  '3octave':[31.5,40,50,63,80,100,125,160,200,250,315,400,500,630,800,1000,1250,1600,2000,2500,3150,4000,5000,6300,8000,10000,12500,16000,20000]
};

function updateFrequencyDropdowns(){
    const bands=document.getElementById('b1oct').checked?'1octave':
                            document.getElementById('b2oct').checked?'2octave':'3octave';
    const minFreq=parseFloat(document.getElementById('minFreq').value)||31.5;
    const maxFreq=parseFloat(document.getElementById('maxFreq').value)||20000;
    const freqs=freqBands[bands].filter(f=>f>=minFreq && f<=maxFreq);
  
    for(let i=1;i<=4;i++){
        const sel=document.getElementById(`t${i}freq`);
        const currentVal=sel.value;
        sel.innerHTML='<option value="">-- None --</option>'+
            '<option value="sum">Summe</option>'+
            freqs.map(f=>`<option value="${f}">${f>=100?Math.round(f):f} Hz</option>`).join('');
        // Only restore value if it exists in the new frequency list or is 'sum'
        if(currentVal && (freqs.map(String).includes(currentVal) || currentVal==='sum')){
            sel.value=currentVal;
        }else{
            sel.value='';  // Reset to blank if frequency not available
        }
    }
    // Update prominent frequency dropdowns
    const promSel1=document.getElementById('promFreq1');
    const promSel2=document.getElementById('promFreq2');
    const promVal1=promSel1.value;
    const promVal2=promSel2.value;
    const options='<option value="">-- None --</option>'+freqs.map(f=>`<option value="${f}">${f>=100?Math.round(f):f} Hz</option>`).join('');
    promSel1.innerHTML=options;
    promSel2.innerHTML=options;
    if(promVal1 && freqs.map(String).includes(promVal1)){
        promSel1.value=promVal1;
    }else{
        promSel1.value='';
    }
    if(promVal2 && freqs.map(String).includes(promVal2)){
        promSel2.value=promVal2;
    }else{
        promSel2.value='';
    }
}

// Load configuration
fetch('api/config').then(r=>r.json()).then(data=>{
    // Set sensor text field with comma-separated sensors
    document.getElementById('sensorText').value = (typeof data.selected_sensors === 'string') ? data.selected_sensors : '';
  document.getElementById('minFreq').value=data.minFreq||31.5;
  document.getElementById('maxFreq').value=data.maxFreq||20000;
  document.getElementById('publishInterval').value=data.publishInterval||1;
    document.getElementById('dbWeighting').value=data.dbWeighting||'A';
    document.getElementById('averagingPeriod').value = data.averagingPeriod || 2;
  if(data.bands==='1octave') document.getElementById('b1oct').checked=true;
  else if(data.bands==='2octave') document.getElementById('b2oct').checked=true;
  else document.getElementById('b3oct').checked=true;
  
  updateFrequencyDropdowns();
  
    for(let i=1;i<=4;i++){
        document.getElementById(`t${i}freq`).value=data.triggers[i-1]?.freq||'';
        document.getElementById(`t${i}amp`).value=data.triggers[i-1]?.amp||'';
        document.getElementById(`t${i}dur`).value=data.triggers[i-1]?.duration||'';
    }
    // Prominent frequency trigger
    document.getElementById('promFreq1').value = data.prominentTrigger?.freq1 || '';
    document.getElementById('promFreq2').value = data.prominentTrigger?.freq2 || '';
    document.getElementById('promDur').value = data.prominentTrigger?.duration || '';
  
  if(data.logic==='OR') document.getElementById('logicOr').checked=true;
  else document.getElementById('logicAnd').checked=true;
  
  document.getElementById('storageLocation').value=data.storageLocation||'/media/wp_audio/events';
    document.getElementById('preBuffer').value=data.preBuffer||10;
  document.getElementById('recLength').value=data.recLength||'';
  
    document.getElementById('cal31_5').value=data.calibration?.cal31_5||'';
  document.getElementById('cal63').value=data.calibration?.cal63||'';
    document.getElementById('cal125').value=data.calibration?.cal125||'';
  document.getElementById('cal250').value=data.calibration?.cal250||'';
    document.getElementById('cal500').value=data.calibration?.cal500||'';
  document.getElementById('cal1000').value=data.calibration?.cal1000||'';
    document.getElementById('cal2000').value=data.calibration?.cal2000||'';
  document.getElementById('cal4000').value=data.calibration?.cal4000||'';
    document.getElementById('cal8000').value=data.calibration?.cal8000||'';
  document.getElementById('cal16000').value=data.calibration?.cal16000||'';
  
  // Add event listeners for band and frequency range changes
  document.getElementById('b1oct').addEventListener('change',updateFrequencyDropdowns);
  document.getElementById('b2oct').addEventListener('change',updateFrequencyDropdowns);
  document.getElementById('b3oct').addEventListener('change',updateFrequencyDropdowns);
  document.getElementById('minFreq').addEventListener('change',updateFrequencyDropdowns);
  document.getElementById('maxFreq').addEventListener('change',updateFrequencyDropdowns);
}).catch(e=>console.error('Load error:',e));

function saveConfig(){
    // Get sensors from text field
    const selectedSensors = document.getElementById('sensorText').value;

    const config={
    bands:document.getElementById('b1oct').checked?'1octave':document.getElementById('b2oct').checked?'2octave':'3octave',
    minFreq:parseFloat(document.getElementById('minFreq').value),
    maxFreq:parseFloat(document.getElementById('maxFreq').value),
    publishInterval:parseFloat(document.getElementById('publishInterval').value)||1,
    dbWeighting:document.getElementById('dbWeighting').value||'A',
    averagingPeriod:parseFloat(document.getElementById('averagingPeriod').value)||2,
        triggers:[
            {freq:document.getElementById('t1freq').value,amp:parseFloat(document.getElementById('t1amp').value)||0,duration:parseFloat(document.getElementById('t1dur').value)||0},
            {freq:document.getElementById('t2freq').value,amp:parseFloat(document.getElementById('t2amp').value)||0,duration:parseFloat(document.getElementById('t2dur').value)||0},
            {freq:document.getElementById('t3freq').value,amp:parseFloat(document.getElementById('t3amp').value)||0,duration:parseFloat(document.getElementById('t3dur').value)||0},
            {freq:document.getElementById('t4freq').value,amp:parseFloat(document.getElementById('t4amp').value)||0,duration:parseFloat(document.getElementById('t4dur').value)||0}
        ],
        prominentTrigger: {
            freq1: document.getElementById('promFreq1').value,
            freq2: document.getElementById('promFreq2').value,
            duration: parseFloat(document.getElementById('promDur').value)||0
        },
    logic:document.getElementById('logicOr').checked?'OR':'AND',
    storageLocation:document.getElementById('storageLocation').value,
    preBuffer:parseInt(document.getElementById('preBuffer').value)||10,
    recLength:parseInt(document.getElementById('recLength').value)||0,
        calibration:{
            cal31_5:document.getElementById('cal31_5').value,
            cal63:document.getElementById('cal63').value,
            cal125:document.getElementById('cal125').value,
            cal250:document.getElementById('cal250').value,
            cal500:document.getElementById('cal500').value,
            cal1000:document.getElementById('cal1000').value,
            cal2000:document.getElementById('cal2000').value,
            cal4000:document.getElementById('cal4000').value,
            cal8000:document.getElementById('cal8000').value,
            cal16000:document.getElementById('cal16000').value
        },
    selected_sensors: selectedSensors
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

function browseFolder(){
  const currentPath=document.getElementById('storageLocation').value||'/media/wp_audio/events';
  const newPath=prompt('Enter storage location path:',currentPath);
  if(newPath!==null&&newPath.trim()!==''){
    document.getElementById('storageLocation').value=newPath.trim();
  }
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
            if self.path == "/api/events":
                # List all recorded events
                storage_dir = analyzer_config.get("storageLocation", "/media/wp_audio/events")
                events = []
                try:
                    if os.path.exists(storage_dir):
                        for event_dir in sorted(os.listdir(storage_dir), reverse=True):
                            event_path = os.path.join(storage_dir, event_dir)
                            if os.path.isdir(event_path):
                                metadata_file = os.path.join(event_path, "event_metadata.json")
                                if os.path.exists(metadata_file):
                                    with open(metadata_file, "r") as f:
                                        metadata = json.load(f)
                                        events.append(metadata)
                except Exception as e:
                    print(f"[wp-audio] Error listing events: {e}", flush=True)
                self.send_response(200)
                self.send_header("Content-Type","application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control","no-store")
                self.end_headers()
                self.wfile.write(json.dumps(events).encode("utf-8"))
                return
            if self.path.startswith("/api/events/"):
                # Serve event files: /api/events/{event_id}/{filename}
                parts = self.path.split("/")
                if len(parts) >= 5:
                    event_id = parts[3]
                    filename = parts[4]
                    storage_dir = analyzer_config.get("storageLocation", "/media/wp_audio/events")
                    file_path = os.path.join(storage_dir, event_id, filename)
                    if os.path.exists(file_path) and os.path.isfile(file_path):
                        # Determine content type
                        if filename.endswith(".flac"):
                            content_type = "audio/flac"
                        elif filename.endswith(".csv"):
                            content_type = "text/csv"
                        elif filename.endswith(".json"):
                            content_type = "application/json"
                        else:
                            content_type = "application/octet-stream"
                        
                        self.send_response(200)
                        self.send_header("Content-Type", content_type)
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.send_header("Content-Length", str(os.path.getsize(file_path)))
                        self.end_headers()
                        
                        with open(file_path, "rb") as f:
                            self.wfile.write(f.read())
                        return
                self.send_response(404)
                self.end_headers()
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
    # Load full analyzer configuration FIRST, before argument parser
    global trigger_config, analyzer_config
    analyzer_config = {
        "bands": "3octave",  # default 1/3-octave
        "minFreq": 31.5,
        "maxFreq": 20000,
        "publishInterval": 1,
        "averagingPeriod": 2,
        "dbWeighting": "A",
        "triggers": [],
        "logic": "OR",
        "storageLocation": "/media/wp_audio/events",
        "preBuffer": 10,
        "recLength": 30,
        "calibration": {"cal31_5": 0, "cal63": 0, "cal125": 0, "cal250": 0, "cal500": 0, "cal1000": 0, "cal2000": 0, "cal4000": 0, "cal8000": 0, "cal16000": 0},
        "selected_sensors": []
    }
    
    # Load from persistent file if exists
    config_file = "/data/analyzer_config.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                saved_config = json.load(f)
                analyzer_config.update(saved_config)
                # Parse selected_sensors as comma-separated string if needed
                sel = analyzer_config.get("selected_sensors", [])
                if isinstance(sel, str):
                    analyzer_config["selected_sensors"] = [s.strip() for s in sel.split(",") if s.strip()]
                elif not isinstance(sel, list):
                    analyzer_config["selected_sensors"] = []
                print(f"[wp-audio] Analyzer configuration loaded: {analyzer_config['bands']} bands, {len(analyzer_config['triggers'])} triggers, logic={analyzer_config['logic']}", flush=True)
        except Exception as e:
            print(f"[wp-audio] Error loading analyzer config: {e}", flush=True)
    
    # Now create argument parser - only infrastructure/technical settings from config.yaml
    ap=argparse.ArgumentParser()
    ap.add_argument("--mqtt-host",default="core-mosquitto")
    ap.add_argument("--mqtt-port",type=int,default=1883)
    ap.add_argument("--mqtt-user",default="")
    ap.add_argument("--mqtt-pass",default="")
    ap.add_argument("--topic-base",default="wp_audio")
    ap.add_argument("--samplerate",type=int,default=48000)
    ap.add_argument("--device",default="")
    ap.add_argument("--cal-file",default="/data/calibration.json")
    ap.add_argument("--ui-port", type=int, default=8099)
    args=ap.parse_args()
    
    # All analysis settings come from analyzer_config (UI)
    args.pre = analyzer_config.get("preBuffer", 10)
    args.post = analyzer_config.get("preBuffer", 10)  # Use same value for post-buffer
    args.post = 30  # Fixed post-trigger time
    args.hold_sec = 2  # Fixed hold time (minimum duration for trigger to be active)
    args.event_dir = analyzer_config.get("storageLocation", "/media/wp_audio/events")
    args.publish_spectrum = True
    args.spectrum_weighting = analyzer_config.get("dbWeighting", "A")
    args.spectrum_interval = analyzer_config.get("publishInterval", 1.0)
    args.averaging_period = analyzer_config.get("averagingPeriod", 2.0)
    
    # Initialize trigger_config from saved config
    trigger_config["triggers"] = analyzer_config.get("triggers", [])

    # Web-UI
    print(f"[wp-audio] Starting HTTP server on port {args.ui_port}...", flush=True)
    start_http(args.ui_port)
    print(f"[wp-audio] HTTP server started successfully", flush=True)

    # MQTT
    connected = {"ok": False}
    record_spectrum = {"enabled": False}  # Control spectrum recording (default OFF)
    
    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            connected["ok"] = True
            print("[wp-audio] MQTT verbunden")
            client.publish(f"{args.topic_base}/availability", "online", qos=1, retain=True)
            client.publish(f"{args.topic_base}/selftest", "ok", qos=1, retain=True)
            
            # Subscribe to control topics
            client.subscribe(f"{args.topic_base}/record_spectrum/set")
            print(f"[wp-audio] Subscribed to {args.topic_base}/record_spectrum/set", flush=True)
            
            dev, dev_id = device_info(); disc = "homeassistant"
            cfgspec={"name":"AT Spectrum","unique_id":f"{dev_id}_at_spectrum","state_topic":f"{args.topic_base}/spectrum",
                     "value_template":"{{ value_json.ts }}","json_attributes_topic":f"{args.topic_base}/spectrum",
                     "availability_topic":f"{args.topic_base}/availability","device":dev,"icon":"mdi:waveform"}
            cfgspec_live={"name":"AT Spectrum Live","unique_id":f"{dev_id}_at_spectrum_live","state_topic":f"{args.topic_base}/spectrum_live",
                     "value_template":"{{ value_json.ts }}","json_attributes_topic":f"{args.topic_base}/spectrum_live",
                     "availability_topic":f"{args.topic_base}/availability","device":dev,"icon":"mdi:waveform"}
            
            # Event log sensor
            cfgevent={"name":"AT Event Log","unique_id":f"{dev_id}_at_event_log","state_topic":f"{args.topic_base}/event",
                     "value_template":"{{ value_json.start }}","json_attributes_topic":f"{args.topic_base}/event",
                     "availability_topic":f"{args.topic_base}/availability","device":dev,"icon":"mdi:calendar-clock"}
            
            # Create a switch to control spectrum recording
            cfgswitch={"name":"AT Record Spectrum","unique_id":f"{dev_id}_at_record_spectrum",
                      "state_topic":f"{args.topic_base}/record_spectrum/state",
                      "command_topic":f"{args.topic_base}/record_spectrum/set",
                      "payload_on":"ON","payload_off":"OFF",
                      "state_on":"ON","state_off":"OFF",
                      "optimistic":False,"qos":1,
                      "availability_topic":f"{args.topic_base}/availability",
                      "device":dev,"icon":"mdi:database"}
            
            # Delete old discovery configs (cleanup from previous versions)
            client.publish(f"{disc}/sensor/{dev_id}/spectrum/config", "", qos=1, retain=True)
            client.publish(f"{disc}/sensor/{dev_id}/spectrum_live/config", "", qos=1, retain=True)
            client.publish(f"{disc}/sensor/{dev_id}/octA_80/config", "", qos=1, retain=True)
            client.publish(f"{disc}/sensor/{dev_id}/octA_160/config", "", qos=1, retain=True)
            client.publish(f"{disc}/sensor/{dev_id}/wp_spectrum/config", "", qos=1, retain=True)
            client.publish(f"{disc}/sensor/{dev_id}/wp_spectrum_live/config", "", qos=1, retain=True)
            client.publish(f"{disc}/sensor/{dev_id}/event_log/config", "", qos=1, retain=True)
            client.publish(f"{disc}/switch/{dev_id}/record_spectrum/config", "", qos=1, retain=True)
            # Also clean up old wp_audio_trigger device configs
            client.publish(f"{disc}/sensor/wp_audio_trigger/wp_spectrum/config", "", qos=1, retain=True)
            client.publish(f"{disc}/sensor/wp_audio_trigger/wp_spectrum_live/config", "", qos=1, retain=True)
            client.publish(f"{disc}/sensor/wp_audio_trigger/event_log/config", "", qos=1, retain=True)
            client.publish(f"{disc}/switch/wp_audio_trigger/record_spectrum/config", "", qos=1, retain=True)
            
            # Publish current discovery configs
            client.publish(f"{disc}/sensor/{dev_id}/at_spectrum/config", json.dumps(cfgspec), qos=1, retain=True)
            client.publish(f"{disc}/sensor/{dev_id}/at_spectrum_live/config", json.dumps(cfgspec_live), qos=1, retain=True)
            client.publish(f"{disc}/sensor/{dev_id}/at_event_log/config", json.dumps(cfgevent), qos=1, retain=True)
            client.publish(f"{disc}/switch/{dev_id}/at_record_spectrum/config", json.dumps(cfgswitch), qos=1, retain=True)
            
            # Publish initial state
            client.publish(f"{args.topic_base}/record_spectrum/state", "ON" if record_spectrum["enabled"] else "OFF", qos=1, retain=True)
        else:
            print(f"[wp-audio] MQTT connect failed rc={rc}")

    def on_message(client, userdata, msg):
        try:
            payload = msg.payload.decode('utf-8')
            if msg.topic == f"{args.topic_base}/record_spectrum/set":
                if payload == "ON":
                    record_spectrum["enabled"] = True
                    print("[wp-audio] Spectrum recording ENABLED", flush=True)
                elif payload == "OFF":
                    record_spectrum["enabled"] = False
                    print("[wp-audio] Spectrum recording DISABLED", flush=True)
                # Echo state back
                client.publish(f"{args.topic_base}/record_spectrum/state", payload, qos=1, retain=True)
        except Exception as e:
            print(f"[wp-audio] MQTT message error: {e}", flush=True)
    
    def on_disconnect(client, userdata, rc, properties=None):
        connected["ok"] = False
        print(f"[wp-audio] MQTT disconnected rc={rc}")

    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION1, protocol=mqtt.MQTTv311)
    client.on_connect = on_connect; client.on_disconnect = on_disconnect; client.on_message = on_message
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
                # Support decimal keys like cal31_5 -> 31.5, and comma/point decimal
                key = freq_key.replace("cal", "").replace("_", ".")
                freq = float(key.replace(",", "."))
                offset = float(str(offset_str).replace(",", ".")) if offset_str else 0.0
                if offset != 0.0:
                    band_corr[freq] = band_corr.get(freq, 0.0) + offset
                    print(f"[wp-audio] Calibration: {freq} Hz += {offset:.2f} dB", flush=True)
            except Exception as e:
                print(f"[wp-audio] WARNING: invalid calibration entry {freq_key}={offset_str}: ignored ({e})", flush=True)
    
    def spl_db(rms): return 20.0*np.log10(max(rms,1e-20)/20e-6)+cal_off

    # Ziel-SR & Filter-Builder
    fs_target = int(args.samplerate)
    # Blockzeit = Publish-Intervall für synchrone Spektrum-Erfassung
    block_sec = float(args.spectrum_interval)
    def build_filters(fs):
        sos_low  = {fc: band_sos(fc, fs) for fc in FCS_LOW}
        sos_full = {fc: band_sos(fc, fs) for fc in FCS_FULL}
        return sos_low, sos_full
    sos_low, sos_full = build_filters(fs_target)
    a_low    = {fc: a_corr(fc) for fc in FCS_LOW}

    # Build interpolated calibration corrections for current target bands
    def build_interpolated_corr(bcorr: dict, targets: list) -> dict:
        if not bcorr:
            return {fc: 0.0 for fc in targets}
        # Sort calibration points by frequency
        cps = sorted(((float(f), float(v)) for f, v in bcorr.items()), key=lambda x: x[0])
        freqs = [f for f, _ in cps]
        logs  = [math.log10(f) for f in freqs]
        vals  = [v for _, v in cps]
        def interp(fc: float) -> float:
            lf = math.log10(fc)
            if lf <= logs[0]:
                return vals[0]
            if lf >= logs[-1]:
                return vals[-1]
            # Find interval
            for i in range(1, len(logs)):
                if lf <= logs[i]:
                    w = (lf - logs[i-1]) / (logs[i] - logs[i-1])
                    return vals[i-1] * (1.0 - w) + vals[i] * w
            return 0.0
        result = {}
        for fc in targets:
            result[fc] = bcorr.get(fc, interp(fc))
        return result

    corr_low  = build_interpolated_corr(band_corr, FCS_LOW)
    corr_full = build_interpolated_corr(band_corr, FCS_FULL)

    pre_buf=deque(maxlen=max(1,int(args.pre/block_sec)))
    spec_buf=deque(maxlen=max(1,int(args.pre/block_sec)))  # Ring buffer for spectrum data
    post_buf_audio=deque(maxlen=max(1,int(args.post/block_sec)))
    post_buf_spec=deque(maxlen=max(1,int(args.post/block_sec)))
    S = {"trig": False, "hold": 0, "post_left": 0, "peak80": -999.0, "peak160": -999.0,
        "cur_dir": None, "event_audio": [], "event_specs": [], "overall_dbA": [],
        "event_start_time": None, "actual_duration": 0, "recording_stopped": False,
        "hold_start_idx": None, "hold_start_time": None}
    if not args.event_dir:
        raise ValueError("event_dir is empty! Please set a valid storage location.")
    os.makedirs(args.event_dir, exist_ok=True)
    
    # Trigger logging
    trigger_log = []
    trigger_states = {}  # Track active triggers: {freq: {"start_time": ts, "start_amp": amp}}

    def start_event():
        S["trig"]=True; S["post_left"]=args.post
        S["cur_dir"]=os.path.join(args.event_dir, now_utc()); os.makedirs(S["cur_dir"], exist_ok=True)
        S["event_audio"]=list(pre_buf); S["event_specs"]=[]; S["peak80"]=-999.0; S["peak160"]=-999.0
        print(f"[wp-audio] Event START {S['cur_dir']}")

    def end_event(current_fs):
        if not S["cur_dir"]: return
        audio=np.concatenate(S["event_audio"],axis=0) if S["event_audio"] else np.zeros(0,np.float32)
        wav=os.path.join(S["cur_dir"],"audio.flac"); sf.write(wav, audio, int(current_fs), format="FLAC")
        csv = os.path.join(S["cur_dir"], "spectrum.csv")
        # Determine which weighting to record based on config
        weighting = analyzer_config.get("dbWeighting", "A")
        if weighting == "A":
            col_prefix = "LA"
        elif weighting == "Z":
            col_prefix = "LZ"
        elif weighting == "D":
            col_prefix = "LD"
        else:
            col_prefix = "LA"  # fallback
        with open(csv, "w") as f:
            f.write("ts," + ",".join([f"{col_prefix}_{fc}" for fc in FCS_LOW]) + f",{col_prefix}_sum\n")
            for r in S["event_specs"]:
                # Get the correct band values
                band_vals = r.get(col_prefix, {})
                vals = [band_vals.get(fc, 0.0) for fc in FCS_LOW]
                # Calculate sum value (energy sum, then dB)
                energies = [10 ** (v / 10) for v in vals]
                sum_val = 10 * math.log10(sum(energies)) if energies and sum(energies) > 0 else 0.0
                f.write(r["ts"] + "," + ",".join(f"{v:.2f}" for v in vals) + f",{sum_val:.2f}\n")
        
        # Save trigger log
        trigger_csv=os.path.join(S["cur_dir"],"trigger_log.csv")
        with open(trigger_csv,"w") as f:
            f.write("time,frequency,amplitude,duration\n")
            for entry in trigger_log:
                f.write(f"{entry['time']},{entry['frequency']},{entry['amplitude']},{entry['duration']}\n")
        
        # Calculate overall dB(A) statistics
        max_overall_dbA = max(S["overall_dbA"]) if S["overall_dbA"] else 0.0
        avg_overall_dbA = sum(S["overall_dbA"]) / len(S["overall_dbA"]) if S["overall_dbA"] else 0.0
        
        # Save comprehensive event metadata as JSON
        metadata = {
            "event_id": os.path.basename(S["cur_dir"]),
            "start_time": os.path.basename(S["cur_dir"]),
            "stop_time": now_utc(),
            "configuration": {
                "bands": analyzer_config.get("bands", "3octave"),
                "minFreq": analyzer_config.get("minFreq", 31.5),
                "maxFreq": analyzer_config.get("maxFreq", 20000),
                "triggers": analyzer_config.get("triggers", []),
                "logic": analyzer_config.get("logic", "OR"),
                "preBuffer": analyzer_config.get("preBuffer", 10),
                "recLength": analyzer_config.get("recLength", 30),
                "preAndPostBuffer": analyzer_config.get("preBuffer", 10)
            },
            "statistics": {
                "max_overall_dbA": round(max_overall_dbA, 2),
                "avg_overall_dbA": round(avg_overall_dbA, 2),
                "peak_A80": round(S["peak80"], 2),
                "peak_A160": round(S["peak160"], 2),
                "trigger_count": len(trigger_log),
                "actual_duration_seconds": round(S["actual_duration"], 2),
                "recorded_duration_seconds": len(S["event_specs"]) * block_sec if S["event_specs"] else 0
            },
            "files": {
                "audio": "audio.flac",
                "spectrum": "spectrum.csv",
                "trigger_log": "trigger_log.csv",
                "metadata": "event_metadata.json"
            },
            "triggers": trigger_log
        }
        
        metadata_file = os.path.join(S["cur_dir"], "event_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Simplified MQTT payload with key metadata
        payload = {
            "event_id": metadata["event_id"],
            "start": metadata["start_time"],
            "stop": metadata["stop_time"],
            "max_overall_dbA": metadata["statistics"]["max_overall_dbA"],
            "avg_overall_dbA": metadata["statistics"]["avg_overall_dbA"],
            "trigger_count": metadata["statistics"]["trigger_count"],
            "actual_duration": metadata["statistics"]["actual_duration_seconds"],
            "recorded_duration": metadata["statistics"]["recorded_duration_seconds"],
            "event_dir": S["cur_dir"]
        }
        try: client.publish(f"{args.topic_base}/event", json.dumps(payload), qos=1)
        except: pass
        print(f"[wp-audio] Event ENDE {S['cur_dir']} (Duration={S['actual_duration']:.1f}s, Recorded={len(S['event_specs']) * block_sec:.1f}s, Max dB(A)={max_overall_dbA:.1f}, Avg dB(A)={avg_overall_dbA:.1f}, Triggers={len(trigger_log)})")
        
        # Clear trigger log for next event
        trigger_log.clear()
        S.update({"trig":False,"cur_dir":None,"event_audio":[],"event_specs":[],"overall_dbA":[],"event_start_time":None,"actual_duration":0,"recording_stopped":False})

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
    last_spec_pub = time.monotonic()  # monotonic time
    spectrum_publish_buffer = []  # buffer for each publish interval (e.g. 125ms)
    spectrum_rolling_buffer = []  # rolling buffer for averaging period
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
                lz=spl_db(np.sqrt(np.mean(y*y)))+corr_low.get(fc,0.0)
                la=lz+a_low[fc]
                LZ[fc]=lz; LA[fc]=la

            # Legacy support for 80/160 Hz if they exist in bands
            la80 = LA.get(80, 0.0)
            la160 = LA.get(160, 0.0)

            # UI Snapshot (keep la80/la160 for internal use)
            latest_payload.update({"la80": float(la80), "la160": float(la160)})

            # Pre-Buffer / Event-Aufnahme
            pre_buf.append(x.copy())
            rec={"ts":now_utc(),"LZ":LZ,"LA":LA}
            spec_buf.append(rec)  # Always buffer spectrum data for events
            # Always fill post buffers as well (they will be used only after trigger ends)
            post_buf_audio.append(x.copy())
            post_buf_spec.append(rec)

            # --- Calculate spectrum for this publish interval ---
            spectrum_block = {}
            for fc, sos in sos_full.items():
                y = sosfilt(sos, x)
                lz = spl_db(np.sqrt(np.mean(y*y))) + corr_full.get(fc, 0.0)
                if args.spectrum_weighting == "A":
                    v = lz + a_corr(fc)
                elif args.spectrum_weighting == "C":
                    v = lz + c_corr(fc)
                else:
                    v = lz
                energy = 10 ** (v / 10)
                spectrum_block[fc] = energy
            spectrum_publish_buffer.append(spectrum_block)

            # Check if publish interval has elapsed
            now_mono = time.monotonic()
            if (now_mono - last_spec_pub) >= float(args.spectrum_interval):
                # Average over this publish interval
                avg_energies = {}
                for fc in FCS_FULL:
                    fc_energies = [buf[fc] for buf in spectrum_publish_buffer if fc in buf]
                    avg_energies[fc] = np.mean(fc_energies) if fc_energies else 0.0
                # Add to rolling buffer
                spectrum_rolling_buffer.append(avg_energies.copy())
                # Limit rolling buffer to N = averaging_period / publish_interval
                maxlen = int(round(float(args.averaging_period) / float(args.spectrum_interval)))
                if len(spectrum_rolling_buffer) > maxlen:
                    spectrum_rolling_buffer = spectrum_rolling_buffer[-maxlen:]
                # Calculate rolling average for output
                rolling_avg_energies = {}
                for fc in FCS_FULL:
                    fc_energies = [buf[fc] for buf in spectrum_rolling_buffer if fc in buf]
                    rolling_avg_energies[fc] = np.mean(fc_energies) if fc_energies else 0.0
                avg_vals = [round(10 * np.log10(rolling_avg_energies[fc]), 1) if rolling_avg_energies[fc] > 0 else 0.0 for fc in FCS_FULL]
                sum_level = 10 * np.log10(sum(rolling_avg_energies.values())) if rolling_avg_energies and sum(rolling_avg_energies.values()) > 0 else 0.0
                timestamp_avg = now_utc()
                payload_avg = {
                    "bands": [str(int(fc)) if fc >= 100 else str(fc) for fc in FCS_FULL],
                    "values": avg_vals,
                    "sum_level": round(sum_level, 1),
                    "weighting": args.spectrum_weighting,
                    "averaging_period": args.averaging_period,
                    "ts": timestamp_avg,
                    "time": timestamp_avg
                }
                latest_payload.update(payload_avg)
                # Publish rolling average as both live and averaged
                try:
                    client.publish(f"{args.topic_base}/spectrum_live", json.dumps(payload_avg), qos=0)
                except:
                    pass
                if record_spectrum["enabled"]:
                    try:
                        client.publish(f"{args.topic_base}/spectrum", json.dumps(payload_avg), qos=0)
                    except:
                        pass
                # Reset publish buffer and update last publish time
                spectrum_publish_buffer = []
                last_spec_pub = now_mono

            # Dynamic Trigger Evaluation
            triggers = analyzer_config.get("triggers", [])
            logic = analyzer_config.get("logic", "OR")
            trigger_results = []
            current_time = now_utc()
            
            # Debug: Show 250Hz amplitude every 10 seconds
            if not hasattr(start_event, 'last_amp_log'):
                start_event.last_amp_log = 0
            if time.time() - start_event.last_amp_log > 10:
                if 250 in LA:
                    print(f"[wp-audio] DEBUG: 250Hz amplitude = {LA[250]:.1f} dB(A), Triggers configured: {len(triggers)}", flush=True)
                    if triggers:
                        print(f"[wp-audio] DEBUG: Trigger config = {triggers}, Logic = {logic}", flush=True)
                start_event.last_amp_log = time.time()
            
            # Only evaluate triggers that are actually configured (freq or 'sum') and amp > 0
            active_trigger_count = sum(1 for t in triggers if (t.get("freq") and t.get("amp", 0) > 0))

            for trig in triggers:
                freq = trig.get("freq")
                amp = trig.get("amp", 0)
                if freq and amp > 0:
                    if freq == "sum":
                        # ...existing code for 'sum' trigger...
                        sum_level = latest_payload.get("sum_level")
                        is_triggered = sum_level is not None and sum_level >= amp
                        trigger_results.append(is_triggered)
                        if is_triggered:
                            print(f"[wp-audio] TRIGGER ACTIVATED: Summe @ {sum_level:.1f} dB (threshold {amp:.1f} dB)", flush=True)
                        # ...existing code for state tracking...
                        if is_triggered:
                            if "sum" not in trigger_states:
                                trigger_states["sum"] = {"start_time": current_time, "start_amp": sum_level}
                                print(f"[wp-audio] Trigger ACTIVE: Summe @ {sum_level:.1f} dB (threshold {amp:.1f} dB)", flush=True)
                        else:
                            if "sum" in trigger_states:
                                start_info = trigger_states["sum"]
                                try:
                                    from datetime import datetime
                                    start_dt = datetime.fromisoformat(start_info["start_time"].replace('Z', '+00:00'))
                                    end_dt = datetime.fromisoformat(current_time.replace('Z', '+00:00'))
                                    duration = (end_dt - start_dt).total_seconds()
                                except:
                                    duration = 0.0
                                log_entry = {
                                    "time": start_info["start_time"],
                                    "frequency": "sum",
                                    "amplitude": round(start_info["start_amp"], 2),
                                    "duration": round(duration, 2)
                                }
                                trigger_log.append(log_entry)
                                print(f"[wp-audio] Trigger logged: Summe, {start_info['start_amp']:.1f} dB, {duration:.2f}s", flush=True)
                                del trigger_states["sum"]
                    else:
                        try:
                            freq_val = float(freq)
                        except:
                            continue
                        if freq_val in LA:
                            is_triggered = LA[freq_val] >= amp
                            trigger_results.append(is_triggered)
                            if is_triggered:
                                print(f"[wp-audio] TRIGGER ACTIVATED: {freq_val}Hz @ {LA[freq_val]:.1f} dB (threshold {amp:.1f} dB)", flush=True)
                        else:
                            print(f"[wp-audio] WARNING: Trigger frequency {freq_val} Hz not found in current bands. Available: {sorted(LA.keys())}", flush=True)
                        # ...existing code for state tracking...
                        if is_triggered:
                            if freq_val not in trigger_states:
                                trigger_states[freq_val] = {"start_time": current_time, "start_amp": LA[freq_val]}
                                print(f"[wp-audio] Trigger ACTIVE: {freq_val} Hz @ {LA[freq_val]:.1f} dB (threshold {amp:.1f} dB)", flush=True)
                        else:
                            if freq_val in trigger_states:
                                start_info = trigger_states[freq_val]
                                try:
                                    from datetime import datetime
                                    start_dt = datetime.fromisoformat(start_info["start_time"].replace('Z', '+00:00'))
                                    end_dt = datetime.fromisoformat(current_time.replace('Z', '+00:00'))
                                    duration = (end_dt - start_dt).total_seconds()
                                except:
                                    duration = 0.0
                                log_entry = {
                                    "time": start_info["start_time"],
                                    "frequency": freq_val,
                                    "amplitude": round(start_info["start_amp"], 2),
                                    "duration": round(duration, 2)
                                }
                                trigger_log.append(log_entry)
                                print(f"[wp-audio] Trigger logged: {freq_val} Hz, {start_info['start_amp']:.1f} dB, {duration:.2f}s", flush=True)
                                del trigger_states[freq_val]

            # --- Prominent frequency trigger (No. 5) ---
            prom_trig = analyzer_config.get("prominentTrigger", {})
            prom_freq1 = prom_trig.get("freq1")
            prom_freq2 = prom_trig.get("freq2")
            prom_dur = prom_trig.get("duration", 0)
            prom_freq_vals = []
            try:
                if prom_freq1:
                    prom_freq_vals.append(float(prom_freq1))
                if prom_freq2 and prom_freq2 != prom_freq1:
                    prom_freq_vals.append(float(prom_freq2))
            except:
                prom_freq_vals = []
            if prom_freq_vals:
                if not hasattr(main, "prominent_buffer") or main.prominent_buffer.maxlen != max(1, int(prom_dur/block_sec)):
                    main.prominent_buffer = deque(maxlen=max(1, int(prom_dur/block_sec)))
                # Is any selected freq the max?
                is_prominent = False
                for pf in prom_freq_vals:
                    if pf in LA and all(LA[pf] > LA[f] for f in LA if f != pf):
                        is_prominent = True
                        break
                main.prominent_buffer.append(is_prominent)
                # Only trigger if it has been true for the whole duration
                if len(main.prominent_buffer) == main.prominent_buffer.maxlen and all(main.prominent_buffer):
                    trigger_results.append(True)
                    print(f"[wp-audio] TRIGGER ACTIVATED: Prominent frequency {prom_freq_vals} Hz for {prom_dur}s", flush=True)
                else:
                    trigger_results.append(False)
            
            # Apply logic (OR = any trigger, AND = all triggers)
            # If no valid triggers, do not trigger any event
            if active_trigger_count == 0:
                trigger_event = False
            elif logic == "AND":
                trigger_event = all(trigger_results) if trigger_results else False
            else:  # OR (default)
                trigger_event = any(trigger_results) if trigger_results else False
            
            # Determine required hold duration from active triggers
            # For AND logic: use maximum duration among all configured triggers
            # For OR logic: use minimum duration among active triggers
            required_duration = args.hold_sec  # fallback to default 2s
            trigger_durations = [
                trig.get("duration", 0)
                for trig in triggers
                if trig.get("amp", 0) > 0 and (
                    (trig.get("freq") == "sum") or
                    (trig.get("freq") and str(trig.get("freq")).replace('.', '', 1).isdigit())
                )
            ]
            if trigger_durations:
                if logic == "AND":
                    required_duration = max(trigger_durations) if trigger_durations else args.hold_sec
                else:  # OR
                    required_duration = min([d for d in trigger_durations if d > 0]) if any(d > 0 for d in trigger_durations) else args.hold_sec
            
            # Debug: show trigger evaluation result
            if len(trigger_results) > 0 and not S["trig"]:
                print(f"[wp-audio] Trigger results: {trigger_results}, Logic={logic}, Event={trigger_event}", flush=True)
            if trigger_event and S["hold"] == 0:
                print(f"[wp-audio] Trigger event started! Logic={logic}, Required duration: {required_duration}s", flush=True)
            
            # Use configured storage location and recording length
            storage_dir = analyzer_config.get("storageLocation", args.event_dir)
            rec_length = analyzer_config.get("recLength", args.post)
            
            if not S["trig"]:
                if trigger_event:
                    if S["hold"] == 0:
                        # Mark the index and time when the hold period starts
                        S["hold_start_idx"] = len(pre_buf)
                        S["hold_start_time"] = time.time()
                    S["hold"]+=block_sec
                    print(f"[wp-audio] Accumulating hold time: {S['hold']:.2f}s / {required_duration:.2f}s", flush=True)
                    if S["hold"]>=required_duration:
                        S["trig"]=True; S["post_left"]=args.post
                        # Calculate the start index for the event in the pre-buffer
                        pre_buf_list = list(pre_buf)
                        spec_buf_list = list(spec_buf)
                        # The event should start at (len(pre_buf) - int(args.pre/block_sec) + S["hold_start_idx"]) if possible
                        event_audio = pre_buf_list[S["hold_start_idx"]:] if S["hold_start_idx"] is not None else pre_buf_list
                        event_specs = spec_buf_list[S["hold_start_idx"]:] if S["hold_start_idx"] is not None else spec_buf_list
                        S["cur_dir"]=os.path.join(storage_dir, now_utc()); os.makedirs(S["cur_dir"], exist_ok=True)
                        S["event_audio"]=event_audio; S["event_specs"]=event_specs; S["peak80"]=-999.0; S["peak160"]=-999.0; S["overall_dbA"]=[]
                        S["event_start_time"]=S["hold_start_time"] if S["hold_start_time"] is not None else time.time(); S["actual_duration"]=0; S["recording_stopped"]=False
                        print(f"[wp-audio] Event START {S['cur_dir']} (Pre-buffer: {len(event_audio)} audio blocks, {len(event_specs)} spectrum blocks)")
                        S["hold"]=0
                        S["hold_start_idx"] = None
                        S["hold_start_time"] = None
                        # Clear post buffers at event start
                        post_buf_audio.clear()
                        post_buf_spec.clear()
                else:
                    S["hold"]=0
                    S["hold_start_idx"] = None
                    S["hold_start_time"] = None
            else:
                # Track actual event duration
                S["actual_duration"] = time.time() - S["event_start_time"] if S["event_start_time"] is not None else 0
                # Always append to event buffers during event
                S["event_audio"].append(x.copy()); S["event_specs"].append(rec)
                S["peak80"]=max(S["peak80"],la80); S["peak160"]=max(S["peak160"],la160)
                # Calculate overall dB(A) from all frequency bands (energy sum)
                total_energy = sum(10**(la/10) for la in LA.values())
                overall_dbA = 10 * np.log10(total_energy) if total_energy > 0 else 0.0
                S["overall_dbA"].append(overall_dbA)
                # If trigger ended, start/continue post-buffering
                if not trigger_event:
                    S["post_left"]-=block_sec
                    # Append all post-buffer data when post buffer is full or post_left <= 0
                    if S["post_left"] <= 0:
                        # Append the entire post buffer to the event
                        S["event_audio"].extend(list(post_buf_audio))
                        S["event_specs"].extend(list(post_buf_spec))
                        if not S["recording_stopped"]:
                            print(f"[wp-audio] DEBUG: Trigger ended, calling end_event, cur_dir={S['cur_dir']}, actual_duration={S['actual_duration']:.1f}s", flush=True)
                            end_event(current_fs)
                        else:
                            print(f"[wp-audio] Event tracking ended. Total duration: {S['actual_duration']:.1f}s", flush=True)
                            S["trig"]=False; S["hold"]=0
                    # Otherwise, keep filling the post buffer
                else:
                    # Trigger still active, reset post timer
                    S["post_left"]=args.post

    finally:
        try:
            stream.stop(); stream.close()
        except: pass

if __name__=="__main__":
    main()
