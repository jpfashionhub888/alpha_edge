#!/bin/bash

echo "Deploying AlphaEdge..."

# Update system
apt update && apt upgrade -y

# Install Python
apt install -y python3 python3-pip python3-venv git

# Create project directory
mkdir -p /root/alpha_edge
cd /root/alpha_edge

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy service file
cp alphaedge.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable alphaedge
systemctl start alphaedge

echo "AlphaEdge deployed!"
echo "Check status: systemctl status alphaedge"
echo "View logs: journalctl -u alphaedge -f"
