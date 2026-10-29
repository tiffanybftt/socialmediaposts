#!/usr/bin/env bash
# Mix narration + music (ducked under the voice) onto the silent render.
# usage: tools/mix.sh [output.mp4]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-ai-summit-claude-training.mp4}

ffmpeg -y -loglevel error \
  -i build/video_silent.mp4 -i build/narration.wav -i build/music.wav \
  -filter_complex "
    [1:a]aresample=48000,pan=stereo|c0=c0|c1=c0,loudnorm=I=-16:TP=-1.5:LRA=7,asplit=2[nar][sc];
    [2:a]aresample=48000,volume=${MUSIC_VOL:-0.13}[mus];
    [mus][sc]sidechaincompress=threshold=0.015:ratio=${DUCK_RATIO:-6}:attack=30:release=700[duck];
    [nar][duck]amix=inputs=2:weights=1 1:normalize=0,alimiter=limit=0.95,apad[a]" \
  -map 0:v -map "[a]" -c:v copy -c:a aac -b:a 192k -movflags +faststart -shortest "$OUT"

ffprobe -v error -show_entries format=duration,size -of default=nw=1 "$OUT"
