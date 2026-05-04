#!/bin/bash

echo "==============================="
echo "        AURORA DEMO"
echo "==============================="

echo ""
echo "Select a demo case:"
echo "1 → STABLE"
echo "2 → IRREGULAR"
echo "3 → LOW CONTRACTION"

read choice

if [ "$choice" == "1" ]; then
    FILE="examples/stable/patient029_4d.nii.gz"
elif [ "$choice" == "2" ]; then
    FILE="examples/irregular/patient094_4d.nii.gz"
elif [ "$choice" == "3" ]; then
    FILE="examples/low_contraction/patient008_4d.nii.gz"
else
    echo "Invalid choice"
    exit 1
fi

echo ""
echo "Running AURORA on: $FILE"

pip3 install -r requirements.txt

python3 src/build_visual_demo.py --input "$FILE"

echo ""
echo "✅ Output saved at:"
echo "aurora_results/demo_final/"
echo "==============================="
