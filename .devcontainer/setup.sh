#!/bin/bash

set -e

echo "Setting up development environment..."

# Install Emscripten
echo "Installing Emscripten..."
cd /tmp
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest

# Add Emscripten to PATH for all sessions
echo 'source /tmp/emsdk/emsdk_env.sh' >> ~/.bashrc
echo 'source /tmp/emsdk/emsdk_env.sh' >> ~/.zshrc

# Source it for current session
source /tmp/emsdk/emsdk_env.sh

# Go back to workspace
cd /workspaces/wasm-blas-ts

# Install Node.js dependencies
echo "Installing Node.js dependencies..."
npm install

echo "Development environment setup complete!"
echo "Please run 'source /tmp/emsdk/emsdk_env.sh' to activate Emscripten in your current shell."
