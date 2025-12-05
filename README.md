# chemical-analysis-project
"Chemical property analysis and visualization of compounds"



# create env (optional)
conda create -n chemproj python=3.10 -y
conda activate chemproj

# RDKit (recommended via conda)
conda install -c conda-forge rdkit -y

# other packages
pip install pandas numpy scikit-learn umap-learn matplotlib seaborn rdkit-pypi

