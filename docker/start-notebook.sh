#!/bin/bash
cd /tmp/sciknowmap/
exec screen -dmS ipython jupyter notebook --ip='*' --port 8888 --no-browser

