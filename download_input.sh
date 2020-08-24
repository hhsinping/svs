#!/bin/bash
wget https://hhsinping.github.io/svs/link/labels.zip
unzip labels.zip
mv labels/* inputs
rm -r labels.zip labels
