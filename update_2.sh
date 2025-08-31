#!/bin/bash

update() {
    #Backup original files
    cp /usr/lib/python3/dist-packages/OpenScan.py /usr/lib/python3/dist-packages/OpenScantmp.py
    cp /home/pi/OpenScan/files/fla.py /home/pi/OpenScan/files/flatmp.py
    cp /home/pi/OpenScan/settings/.node-red/flows.json /home/pi/OpenScan/settings/.node-red/flowstmp.json

    #Get the 3 udated files from github
    echo "Downloading updated files...."
    wget -L https://raw.github.com/egan0/mods/main/fla.py
    wget -L https://raw.github.com/egan0/mods/main/OpenScan.py
    wget -L https://raw.github.com/egan0/mods/main/flows.json

    #Copy the files to thier locations
    cp OpenScan.py /usr/lib/python3/dist-packages/OpenScan.py
    echo "OpenScan.py copied...."
    cp fla.py /home/pi/OpenScan/files/fla.py
    echo "fla.py copied...."
    cp flows.json /home/pi/OpenScan/settings/.node-red/flows.json
    echo "flows.json copied...."

    #Cleanup and remove the downloaded files
    echo "Deleting downloaded files...."
    rm -f OpenScan.py
    rm -f fla.py
    rm -f flows.json

    #Create autofocus settings file if it does not already exist
    if [ ! -f "/home/pi/OpenScan/settings/cam_AFmode" ]; then
        echo "true" > /home/pi/OpenScan/settings/cam_AFmode
        echo "cam_AFmode settings file created...."
    fi

    #Create resolution settings files if it does not already exist
    if [ ! -f "/home/pi/OpenScan/settings/cam_resX" ]; then
        echo "4656" > /home/pi/OpenScan/settings/cam_resX
        echo "cam_resX settings file created...."
    fi
    if [ ! -f "/home/pi/OpenScan/settings/cam_resY" ]; then
        echo "3496" > /home/pi/OpenScan/settings/cam_resY
        echo "cam_resX settings file created...."
    fi

    #Create auto stacking adjust settings file if it does not already exist
    if [ ! -f "/home/pi/OpenScan/settings/cam_stackadj" ]; then
        echo "0.5" > /home/pi/OpenScan/settings/cam_stackadj
        echo "cam_AFmode settings file created...."
    fi
}

restore() {
    cp /usr/lib/python3/dist-packages/OpenScantmp.py /usr/lib/python3/dist-packages/OpenScan.py
    cp /home/pi/OpenScan/files/flatmp.py /home/pi/OpenScan/files/fla.py
    cp /home/pi/OpenScan/settings/.node-red/flowstmp.json /home/pi/OpenScan/settings/.node-red/flows.json
}

case $1 in
  "update")
     echo "Updating...."
     update
     ;;
  "restore")
     echo "Restoring...."
     restore
     ;;
  *)
     echo "Unknown command: $command. Please use update or restore." ;;
esac

read -p "Reboot now (y/n)? " -n 1 -r
echo    
if [[ $REPLY =~ ^[Yy]$ ]]
then
    reboot
fi
