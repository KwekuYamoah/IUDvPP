#!/bin/bash
find . -type f -name '*.m4a' -exec bash -c '
  for file; do
    wav_file="${file%.m4a}.wav"
    ffmpeg -i "$file" "$wav_file" && rm "$file"
  done
' bash {} +

