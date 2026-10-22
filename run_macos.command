#!/bin/bash
cd -- "$(dirname -- "$0")" || exit 1
bash run_mac_linux.sh
status=$?
if [ "$status" -ne 0 ]; then
  printf '\nLaunch failed. Read the error above. Press Enter to close.\n'
  read -r response
fi
exit "$status"
