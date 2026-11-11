Dataset: Tox21 compound assay data, 7,831 source rows and 12 qualitative assay labels.
Raw archive: data/tox21.csv.gz
Source URL: https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz
Loader documentation and task definition: https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html
NIH Tox21 Challenge: https://tripod.nih.gov/tox21/challenge/data.jsp
The public CSV includes SMILES, mol_id, and the 12 NR/SR assay fields. Eight rows were excluded because RDKit could not parse or sanitize their SMILES; source rows are in excluded_invalid.csv. Missing assay results were preserved as missing.
Licence: redistributed by DeepChem/MoleculeNet under the MIT licence; the assay results come from the public US Tox21 programme (NIH, EPA, FDA). Cite Huang et al. (2016) and Wu et al. (2018).
