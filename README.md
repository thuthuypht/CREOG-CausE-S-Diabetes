# CREOG – Causal Reasoning over Ontology-Enriched Graphs with CausE-S

This repository provides a reference implementation of the CREOG framework
for an ontology-only setting, using a diabetes causal ontology (Diabetes_large.owl)
as input.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
pip install -r requirements.txt

## Running with the diabetes ontology

Place your ontology at:

- `data/Diabetes_large.owl`

Then run:

```bash
python creog_main.py --owl_path data/Diabetes_large.owl --output_dir outputs
