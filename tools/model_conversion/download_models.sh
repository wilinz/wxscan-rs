#!/usr/bin/env bash
# Download the published Caffe models, at the commit opencv_contrib's CMakeLists references.
set -euo pipefail
COMMIT=a8b69ccc738421293254aec5ddb38bd523503252
DIR="$(dirname "$0")/models"
mkdir -p "$DIR"
for f in detect.caffemodel detect.prototxt sr.caffemodel sr.prototxt; do
  echo "downloading $f"
  curl -fsSL -o "$DIR/$f" \
    "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/$COMMIT/$f"
done
# Matches the MD5 recorded in opencv_contrib/modules/wechat_qrcode/CMakeLists.txt
echo "238e2b2d6f3c18d6c3a30de0c31e23cf  $DIR/detect.caffemodel" | md5sum -c - 2>/dev/null \
  || echo "(skipping verification: md5sum not available)"
