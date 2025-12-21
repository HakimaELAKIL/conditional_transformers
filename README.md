# Conditional Transformers for Drug Design

This project uses conditional transformers to generate molecules for drug design.

## What it does
- Generates new drug-like molecules using AI
- Analyzes molecules in a step-by-step way (validation, novelty, uniqueness, conditions)
- Uses a small GPT model called nanoDrugGPT

## Files
- `application.py`: Flask web app for generating and analyzing molecules (main interface)
- `app.py`: Streamlit web app for generating and analyzing molecules (alternative)
- `cond_gpt_categorical_extended.pth`: Trained model file
- `vocab_dataset.json`: Vocabulary for molecule encoding
- `s_100_str_+1M_fixed.txt`: Reference dataset of molecules
- `paper_DrugGPT.pdf`: Research paper
- `test_generation.py`: Standalone script to test molecule generation

## Folders
- `datasets/`: Contains molecule datasets
- `Notebooks/`: Jupyter notebooks for experiments and training
- `static/`: CSS and JS files for the Flask interface
- `templates/`: HTML templates for the Flask interface
- `version1/`: Results from first version (plots)

## How to run

### Main Interface (Flask)
1. Install requirements: `pip install torch rdkit flask plotly pandas`
2. Run `python application.py`
3. Open http://localhost:5005 in your browser
4. Select parameters and generate molecules
5. View hierarchical analysis results

### Alternative Interface (Streamlit)
1. Install requirements: `pip install torch rdkit streamlit plotly pandas`
2. Run `streamlit run app.py`
3. Open the web interface in your browser

### Test Generation
1. Run `python test_generation.py` to generate 10 molecules and test validity

## Analysis levels
1. **Valid**: Check if molecules are chemically correct
2. **Novel**: Check if molecules are new (not in reference dataset)
3. **Unique**: Remove duplicates from novel molecules
4. **Conditions**: Check if molecules meet drug design rules (like Lipinski rules)

## Model Details
- Architecture: Conditional GPT with 4 layers, 128 embedding dimensions
- Vocabulary: 57 tokens for SMILES encoding
- Conditions: 10-dimensional vectors for drug design constraints
- Training: On a dataset of drug-like molecules

## Requirements
- Python 3.8+
- PyTorch
- RDKit
- Flask or Streamlit
- Plotly
- Pandas
