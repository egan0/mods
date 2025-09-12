*** For testing ony on OpenScan Meanwhile Version ***

Must have:
- OpenScan Meanwhile os
- IMX519 camera

To install:
- ssh into OpenScan
- in the default '/home/pi' directory, run:
  - wget -L https://raw.github.com/egan0/mods/main/update_2.sh
  - sudo bash update_2.sh "update"
 
This will update three files:
- OpenScan.py
- fla.py
- flows.json

To remove:
 - sudo bash update_2.sh "remove"
