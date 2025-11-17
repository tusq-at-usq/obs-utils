

CWD="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing all components..."

# Install PC setup
echo "Installing PC setup..."
"$SCRIPT_DIR"/install_scripts/install_pc_setup.sh
# Install Certus
echo "Installing Certus..."
"$SCRIPT_DIR"/install_scripts/install_certus.sh
# Install Encoders
echo "Installing Encoders..."
"$SCRIPT_DIR"/install_scripts/install_encoders.sh
# Install Cameras
echo "Installing Cameras..."
"$SCRIPT_DIR"/install_scripts/install_cameras.sh

source ~/.env/obs/bin/activate
uv pip install $SCRIPT_DIR

reboot_on_confirm() {
  read -p "Setup complete. Reboot required for permission to take effect. Reboot now? (y/n): " choice
  case "$choice" in 
    y|Y ) echo "Rebooting..."; sudo reboot;;
    n|N ) echo "Reboot cancelled. Please reboot later to apply all changes.";;
    * ) echo "Invalid input. Please enter y or n.";;
  esac
}
