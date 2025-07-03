#!/bin/bash

# Script to download HuggingFaceH4/ultrachat_200k dataset
# This script downloads the dataset in a format expected by load_ultra_dataset method

set -e  # Exit on any error

# Default download directory
DOWNLOAD_DIR="./data/ultrachat"
REPO_ID="HuggingFaceH4/ultrachat_200k"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            DOWNLOAD_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--output-dir DIR]"
            echo "  --output-dir DIR    Directory to download the dataset (default: ./ultrachat_200k)"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Downloading HuggingFaceH4/ultrachat_200k dataset to: $DOWNLOAD_DIR"

# Create output directory if it doesn't exist
mkdir -p "$DOWNLOAD_DIR"

# Method 1: Try using huggingface-cli (preferred method)
if command -v huggingface-cli &> /dev/null; then
    echo "Using huggingface-cli to download dataset..."
    huggingface-cli download "$REPO_ID" --repo-type dataset --local-dir "$DOWNLOAD_DIR" --local-dir-use-symlinks False
    echo "Download completed successfully using huggingface-cli!"
    
elif command -v git &> /dev/null && git lfs version &> /dev/null; then
    # Method 2: Use git clone with LFS support
    echo "huggingface-cli not found, using git clone with LFS..."
    echo "Cloning repository..."
    cd "$(dirname "$DOWNLOAD_DIR")"
    git clone "https://huggingface.co/datasets/$REPO_ID" "$(basename "$DOWNLOAD_DIR")"
    cd "$(basename "$DOWNLOAD_DIR")"
    git lfs pull
    echo "Download completed successfully using git clone!"
    
else
    # Method 3: Install huggingface-hub and try again
    echo "Neither huggingface-cli nor git with LFS found."
    echo "Attempting to install huggingface-hub..."
    
    if command -v pip &> /dev/null; then
        pip install huggingface-hub[cli]
        echo "Installed huggingface-hub, retrying download..."
        huggingface-cli download "$REPO_ID" --repo-type dataset --local-dir "$DOWNLOAD_DIR" --local-dir-use-symlinks False
        echo "Download completed successfully!"
    elif command -v pip3 &> /dev/null; then
        pip3 install huggingface-hub[cli]
        echo "Installed huggingface-hub, retrying download..."
        huggingface-cli download "$REPO_ID" --repo-type dataset --local-dir "$DOWNLOAD_DIR" --local-dir-use-symlinks False
        echo "Download completed successfully!"
    else
        echo "Error: Could not find pip or pip3 to install huggingface-hub"
        echo "Please install one of the following:"
        echo "  1. huggingface-hub: pip install huggingface-hub[cli]"
        echo "  2. git with LFS: git lfs install"
        echo "Then run this script again."
        exit 1
    fi
fi

# Verify the download
echo "Verifying download..."
if [ -d "$DOWNLOAD_DIR" ]; then
    PARQUET_COUNT=$(find "$DOWNLOAD_DIR" -name "*.parquet" | wc -l)
    echo "Found $PARQUET_COUNT parquet files in $DOWNLOAD_DIR"
    
    if [ "$PARQUET_COUNT" -gt 0 ]; then
        echo "✓ Dataset downloaded successfully!"
        echo "You can now use this dataset with the load_ultra_dataset method by passing the path: $DOWNLOAD_DIR"
    else
        echo "⚠ Warning: No parquet files found in the downloaded directory"
        echo "The dataset might still be downloading or there might be an issue"
    fi
else
    echo "✗ Error: Download directory not found"
    exit 1
fi
