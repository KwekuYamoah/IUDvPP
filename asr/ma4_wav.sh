#!/bin/sh

find . -type f -name '*.m4a' -exec bash -c 'ffmpeg -i "$0" "${0/%m4a/wav}"' '{}' \;
find . -type f -name '*.m4a' -exec bash -c 'rm "$0"' '{}' \;