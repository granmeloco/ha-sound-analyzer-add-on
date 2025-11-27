#!/usr/bin/with-contenv bashio
set -euo pipefail

# --- Supervisor API Token (für Sensor/Device-Registration via HA) ---
if [ -f /var/run/s6/container_environment/SUPERVISOR_TOKEN ]; then
  export SUPERVISOR_TOKEN="$(cat /var/run/s6/container_environment/SUPERVISOR_TOKEN)"
  echo "[wp-audio] SUPERVISOR_TOKEN set for API access"
fi

# --- Konfiguration ---
MQTT_HOST="$(bashio::config 'mqtt_host')"
MQTT_PORT="$(bashio::config 'mqtt_port')"
MQTT_USER="$(bashio::config 'mqtt_user')"
MQTT_PASS="$(bashio::config 'mqtt_password')"
TOPIC_BASE="$(bashio::config 'topic_base')"

THRESH_A80="$(bashio::config 'thresh_a80')"
THRESH_A160="$(bashio::config 'thresh_a160')"
HOLD_SEC="$(bashio::config 'hold_sec')"
PRE_TRIGGER="$(bashio::config 'pre_trigger')"
POST_TRIGGER="$(bashio::config 'post_trigger')"
SR="$(bashio::config 'samplerate')"
DEVICE="$(bashio::config 'device')"     # leer = Pulse-Default
EVENT_DIR="$(bashio::config 'event_dir')"
CAL_FILE="$(bashio::config 'calibration_file')"

PUB_SPEC="$(bashio::config 'publish_spectrum' || true)"
SPEC_WGT="$(bashio::config 'spectrum_weighting' || true)"
SPEC_INT="$(bashio::config 'spectrum_interval' || true)"
UI_PORT="$(bashio::config 'ui_port' || true)"
PULSE_SRC_CFG="$(bashio::config 'pulse_source' || true)"

# Defaults für optionale Felder
case "${PUB_SPEC:-}" in ""|null) PUB_SPEC="true";; esac
case "${SPEC_WGT:-}" in ""|null) SPEC_WGT="Z";; esac
case "${SPEC_INT:-}" in ""|null) SPEC_INT="1";; esac
case "${UI_PORT:-}" in ""|null) UI_PORT="8099";; esac

mkdir -p "$EVENT_DIR"

# --- MQTT-Creds automatisch aus HA (falls leer) ---
if { [ -z "${MQTT_USER}" ] || [ -z "${MQTT_PASS}" ]; } && bashio::services "mqtt" > /dev/null; then
  MQTT_HOST="$(bashio::services 'mqtt' 'host')"
  MQTT_PORT="$(bashio::services 'mqtt' 'port')"
  MQTT_USER="$(bashio::services 'mqtt' 'username')"
  MQTT_PASS="$(bashio::services 'mqtt' 'password')"
  echo "[wp-audio] MQTT creds via HA service: ${MQTT_USER}@${MQTT_HOST}:${MQTT_PORT}"
fi

# --- Pulse vom Supervisor mounten & konfigurieren ---
if [ -S /run/audio/pulse/native ]; then
  export PULSE_SERVER="unix:/run/audio/pulse/native"
elif [ -S /run/audio/pulse.sock ]; then
  export PULSE_SERVER="unix:/run/audio/pulse.sock"
fi
echo "[wp-audio] Using PULSE_SERVER=${PULSE_SERVER:-<none>}"

# Cookie falls vorhanden
if [ -f /data/pulse_cookie ]; then
  export PULSE_COOKIE="/data/pulse_cookie"
  echo "[wp-audio] Using PULSE_COOKIE=${PULSE_COOKIE}"
fi

# Gewünschte Mic-Quelle erzwingen (keine .monitor / kein .mono-fallback)
if [ -n "${PULSE_SRC_CFG:-}" ] && [ "${PULSE_SRC_CFG}" != "null" ]; then
  export PULSE_SOURCE="${PULSE_SRC_CFG}"
  echo "[wp-audio] Forcing PULSE_SOURCE=${PULSE_SOURCE}"
fi

# Latency moderat
export PULSE_LATENCY_MSEC="${PULSE_LATENCY_MSEC:-60}"

# --- Diagnose: Socket & pactl/parec ---
ls -l /run/audio/pulse/native || true
if command -v pactl >/dev/null 2>&1; then
  echo "[wp-audio] pactl info:"
  pactl info || true

  if [ -n "${PULSE_SOURCE:-}" ]; then
    echo "[wp-audio] Set default/mute/volume on source"
    pactl set-default-source "${PULSE_SOURCE}" || true
    pactl set-source-mute   "${PULSE_SOURCE}" 0 || true
    pactl set-source-volume "${PULSE_SOURCE}" 100% || true
  fi

  echo "[wp-audio] Current source status:"
  pactl list sources | sed -n '/'"${PULSE_SOURCE:-}"'/,+30p' | egrep "Name:|State:|Mute:|Volume:|sample spec" || true

  # Kurzprobe: 0.25s Rohaufnahme (zeigt, ob Nicht-Null-Daten kommen)
  if command -v parec >/dev/null 2>&1 && [ -n "${PULSE_SOURCE:-}" ]; then
    echo "[wp-audio] Probe recording 0.25s raw from ${PULSE_SOURCE}"
    timeout 1 parec --device="${PULSE_SOURCE}" --raw --rate=44100 --channels=1 --format=s16le --record \
      | head -c 4096 > /tmp/probe.raw || true
    echo "[wp-audio] Probe size: $(stat -c%s /tmp/probe.raw 2>/dev/null || echo 0) bytes"
    od -An -t u1 -N 64 /tmp/probe.raw 2>/dev/null | head -n1 || true
  fi
fi

# --- Geräteübersicht (PortAudio) ---
python3 - <<'PY'
import sounddevice as sd
try:
    print("[wp-audio] Eingänge:")
    for i,d in enumerate(sd.query_devices()):
        if d.get("max_input_channels",0)>0:
            print(f"  {i}: {d['name']} (hostapi={d['hostapi']})")
except Exception as e:
    print("[wp-audio] Hinweis: Geräteabfrage fehlgeschlagen:", e)
PY

# --- Python starten ---
DEV_ARG="${DEVICE}"   # leer = Pulse-Default
if [ -z "${DEV_ARG// }" ]; then
  echo "[wp-audio] Device arg: '(empty -> Pulse-Default)'  SR=${SR}"
else
  echo "[wp-audio] Device arg (explicit): ${DEV_ARG}  SR=${SR}"
fi

exec python3 /app/wp_audio_trigger.py \
  --mqtt-host "${MQTT_HOST}" --mqtt-port "${MQTT_PORT}" \
  --mqtt-user "${MQTT_USER}" --mqtt-pass "${MQTT_PASS}" \
  --topic-base "${TOPIC_BASE}" \
  --thresh-a80 "${THRESH_A80}" --thresh-a160 "${THRESH_A160}" \
  --hold-sec "${HOLD_SEC}" --pre "${PRE_TRIGGER}" --post "${POST_TRIGGER}" \
  --samplerate "${SR}" --device "${DEV_ARG}" \
  --event-dir "${EVENT_DIR}" --cal-file "${CAL_FILE}" \
  --publish-spectrum "${PUB_SPEC}" --spectrum-weighting "${SPEC_WGT}" --spectrum-interval "${SPEC_INT}" \
  --ui-port "${UI_PORT}"
