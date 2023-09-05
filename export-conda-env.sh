#!/usr/bin/env sh
conda env export | grep -v '^prefix:' > environment.yaml 
