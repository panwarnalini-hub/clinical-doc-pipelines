#!/bin/bash
apt-get update
apt-get install -y pandoc tesseract-ocr
echo "Installed pandoc + tesseract on all nodes"