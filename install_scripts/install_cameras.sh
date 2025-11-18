
CWD="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAM_ASSETS_DIR="$SCRIPT_DIR/../obs_cameras/assets"

# Alvium install
curl -L "https://downloads.alliedvision.com/VimbaX/VimbaX_Setup-2025-3-Linux64.tar.gz" -o "$HOME/Downloads/VimbaX_Setup-2025-3-Linux64.tar.gz"
mkdir -p "$CAM_ASSETS_DIR"/VimbaX
tar -xf ~/Downloads/VimbaX_Setup-2025-3-Linux64.tar.gz -C "$CAM_ASSETS_DIR"/VimbaX --strip-components=1
sudo "$CAM_ASSETS_DIR"/VimbaX/cti/Install_GenTL_Path.sh

# ZWO install
sudo apt update
# mkdir -p "$SCRIPT_DIR"/../obs_cameras/assets
cd "$CAM_ASSETS_DIR"
curl -LO https://github.com/tusq-at-usq/obs-utils/releases/download/v0.1.0/ASI_linux_mac_SDK_V1.39.tar.bz2
tar -xf ./ASI_linux_mac_SDK_V1.39.tar.bz2
sudo install ./ASI_linux_mac_SDK_V1.39/lib/asi.rules /etc/udev/rules.d
cp ./ASI_linux_mac_SDK_V1.39/lib/x64/libASICamera2.so ./
sudo udevadm control --reload-rules
sudo udevadm trigger

rm -rf .ASI_linux_mac_SDK_V1.39 ASI_linux_mac_SDK_V1.39.tar.bz2
cd "$CWD"



echo "Camera installation complete."
