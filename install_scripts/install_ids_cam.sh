

CWD="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


mkdir -p "$SCRIPT_DIR"/../obs_cameras/assets
cd "$SCRIPT_DIR"/../obs_cameras/assets

if [ ! -d "/opt/ids-peak_2.18.1.0-132_amd64" ]; then
    echo "IDS Peak SDK not found. Installing..."
    curl -LO https://en.ids-imaging.com/files/downloads/ids-peak/software/linux-desktop/ids-peak_2.18.1.0-132_amd64.tgz
    tar -xvzf ids-peak_2.18.1.0-132_amd64.tgz
    sudo mv ids-peak_2.18.1.0-132_amd64 /opt
else
    echo "IDS Peak SDK already installed."
fi

# Add LD_LIBRARY_PATH to .bashrc if not already present
if ! grep -q 'LD_LIBRARY_PATH=/opt/ids-peak_2.18.1.0-132_amd64/lib' ~/.bashrc; then
    echo 'export LD_LIBRARY_PATH=/opt/ids-peak_2.18.1.0-132_amd64/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH' >> ~/.bashrc
    echo 'export GENICAM_GENTL64_PATH=/opt/ids-peak_2.18.1.0-132_amd64/lib/x86_64-linux-gnu/ids-peak/cti:$GENICAM_GENTL64_PATH' >> ~/.bashrc
fi



sudo /opt/ids-peak_2.18.1.0-132_amd64/share/ids-peak/scripts/ids_install_udev_rule.sh
