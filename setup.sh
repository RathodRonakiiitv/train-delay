# Quick Start Script for Train Delay Prediction

echo "Setting up Indian Railways Train Delay Prediction..."

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if data exists
if [ ! -f "data/ir_train.csv" ]; then
    echo "ERROR: Data files not found!"
    echo "Please download from Kaggle and place in data/ folder:"
    echo "  - ir_train.csv"
    echo "  - ir_test.csv"
    echo "  - ir_sample_submission.csv"
    echo "  - ir_data_dictionary.csv"
    exit 1
fi

# Create necessary folders
mkdir -p submissions models

echo "Setup complete!"
echo ""
echo "To train models:"
echo "  cd notebooks && jupyter notebook full_pipeline.ipynb"
echo ""
echo "Or run from command line:"
echo "  python src/train.py"
