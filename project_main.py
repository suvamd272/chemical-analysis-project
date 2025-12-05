#!/usr/bin/env python3
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit import DataStructs

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
from scipy.stats import gaussian_kde
# ---------------------------------------
# Configuration
# ---------------------------------------
INPUT_TSV = "mtbs_tropical_annotations.tsv"
OUT_DIR = "analysis_output"
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SMILES_COL = "structure_smiles"
SUPERCLASS_COL = "structure_taxonomy_npclassifier_02superclass"
CLASS_COL = "structure_taxonomy_npclassifier_03class"
PATHWAY_COL = "structure_taxonomy_npclassifier_01pathway"

PALETTE = "tab20c"
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# ---------------------------------------
# Utility Functions
# ---------------------------------------
def safe_mol(smiles):
    if pd.isna(smiles) or not isinstance(smiles, str):
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except:
        return None



def get_fingerprint(mol, n_bits=2048):
    """Get Morgan fingerprint as bit vector"""
    if mol is None:
        return np.zeros(n_bits, dtype=np.int8)
    try:
        fp = rdmd.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.int8)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except:
        return np.zeros(n_bits, dtype=np.int8)

def compute_density_contours(x, y, ax, color='gray', alpha=0.3):
    """Add density contours to scatter plot"""
    try:
        # Calculate point density
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        
        # Sort points by density
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        
        # Plot contour
        ax.tricontourf(x, y, z, levels=10, alpha=alpha, cmap='Greys')
    except:
        pass

# ---------------------------------------
# 1. Load + Clean
# ---------------------------------------
print("Loading data...")
df = pd.read_csv(INPUT_TSV, sep='\t', dtype=str)

df = df[df[SMILES_COL].notna()]
df[SMILES_COL] = df[SMILES_COL].str.strip()
df = df[df[SMILES_COL] != ""]

df_unique = df.drop_duplicates(subset=[SMILES_COL]).reset_index(drop=True)
df_unique['ID'] = [f"S{i+1:04d}" for i in range(len(df_unique))]
print(f"Unique compounds: {len(df_unique)}")

# ---------------------------------------
# 2. Compute descriptors
# ---------------------------------------
print("Computing descriptors...")
descriptors = []
fps = []

for idx, row in df_unique.iterrows():
    mol = safe_mol(row[SMILES_COL])
    
    if mol is None:
        desc = {
            'ID': row['ID'],
            'MW': np.nan, 'LogP': np.nan, 'TPSA': np.nan,
            'HBD': np.nan, 'HBA': np.nan, 'RotBonds': np.nan,
            'Rings': np.nan, 'Fsp3': np.nan, 'Aromatic': np.nan,
            'Heteroatoms': np.nan, 'QED': np.nan
        }
        fps.append(np.zeros(2048, dtype=np.int8))
    else:
        # Compute QED (drug-likeness) if available
        try:
            qed = Descriptors.qed(mol)
        except:
            qed = np.nan
            
        desc = {
            'ID': row['ID'],
            'MW': Descriptors.MolWt(mol),
            'LogP': Crippen.MolLogP(mol),
            'TPSA': rdmd.CalcTPSA(mol),
            'HBD': rdmd.CalcNumHBD(mol),
            'HBA': rdmd.CalcNumHBA(mol),
            'RotBonds': rdmd.CalcNumRotatableBonds(mol),
            'Rings': rdmd.CalcNumRings(mol),
            'Fsp3': rdmd.CalcFractionCSP3(mol),
            'Aromatic': sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) / max(1, mol.GetNumAtoms()),
            'Heteroatoms': sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6)),
            'QED': qed
        }
        fps.append(get_fingerprint(mol))
    
    descriptors.append(desc)

desc_df = pd.DataFrame(descriptors)
df_merged = pd.merge(df_unique, desc_df, on='ID')
df_merged.to_csv(os.path.join(OUT_DIR, "compounds_with_descriptors.csv"), index=False)

# Fill missing values with median
desc_cols = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'Rings', 'Fsp3', 'Aromatic', 'Heteroatoms', 'QED']
for col in desc_cols:
    df_merged[col] = df_merged[col].fillna(df_merged[col].median())

# Prepare descriptor matrix for correlation
desc_matrix = df_merged[desc_cols].values
desc_scaled = StandardScaler().fit_transform(desc_matrix)



#-------------------ANALYSE and VISUALIZATION---------------------------

# ---------------------------------------
# 1. PCA LOADING
# ---------------------------------------
print("\n=== PCA Analysis ===")
pca = PCA(n_components=2)
pcs = pca.fit_transform(desc_scaled)

var_text = (
    f"PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}%\n"
    f"PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}%"
)

df_merged['PC1'] = pcs[:,0]
df_merged['PC2'] = pcs[:,1]

print(f"Total variance explained by PC1+PC2: {sum(pca.explained_variance_ratio_[:2])*100:.1f}%")
print(f"PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance")
print(f"PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}% of variance")

loadings_df = pd.DataFrame(pca.components_.T, index=desc_cols, columns=['PC1','PC2'])
fig_load, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))

for ax, pc in zip([ax1, ax2], ['PC1','PC2']):
    sorted_load = loadings_df[pc].sort_values()
    colors = ['green' if x>0 else 'red' for x in sorted_load]
    ax.barh(range(len(sorted_load)), sorted_load.values, color=colors)
    ax.set_yticks(range(len(sorted_load)))
    ax.set_yticklabels(sorted_load.index)
    ax.set_xlabel('Loading')
    ax.set_title(f'{pc} Loadings')
    ax.axvline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "pca_loadings.png"), dpi=600)
plt.close()

print("PCA loadings saved.")
scale_factor = 3.0
loadings_scaled = loadings_df.values * scale_factor

print(loadings_df)


def interpret_pc_axis(loadings, pc_name):
    """Create interpretable axis label based on loadings"""
    # Get top 3 positive and negative contributors
    pos = loadings.nlargest(3)
    neg = loadings.nsmallest(3)
    
    pos_str = " + ".join([f"{abs(val):.2f}×{name}" for name, val in pos.items()])
    neg_str = " + ".join([f"{abs(val):.2f}×{name}" for name, val in neg.items()])
    
    if len(pos) > 0 and len(neg) > 0:
        return f"{pc_name}: [{pos_str}] - [{neg_str}]"
    else:
        return pc_name

pc1_interpretation = interpret_pc_axis(loadings_df['PC1'], "PC1")
pc2_interpretation = interpret_pc_axis(loadings_df['PC2'], "PC2")

print(f"\nInterpretation of axes:")
print(f"PC1 represents: {pc1_interpretation}")
print(f"PC2 represents: {pc2_interpretation}")


# Save loadings to file
loadings_df.to_csv(os.path.join(OUT_DIR, "pca_loadings.csv"))

print("PCA loadings saved to: pca_loadings.csv")





# ---------------------------------------
# 2. Correlation heatmap btw different properities
# ---------------------------------------

fig_a, ax_a = plt.subplots(figsize=(6,5))
corr_matrix = np.corrcoef(desc_scaled.T)
dist_matrix = pdist(1 - np.abs(corr_matrix))
linkage_matrix = hierarchy.linkage(dist_matrix, method='average')
dendro_order = hierarchy.leaves_list(linkage_matrix)
corr_reordered = corr_matrix[dendro_order][:, dendro_order]
labels_reordered = np.array(desc_cols)[dendro_order]

im = ax_a.imshow(corr_reordered, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax_a)
ax_a.set_xticks(range(len(labels_reordered)))
ax_a.set_yticks(range(len(labels_reordered)))
ax_a.set_xticklabels(labels_reordered, rotation=45, ha='right')
ax_a.set_yticklabels(labels_reordered)
ax_a.set_title('Descriptor Correlation Heatmap')
for i in range(len(labels_reordered)):
    for j in range(len(labels_reordered)):
        ax_a.text(j,i,f'{corr_reordered[i,j]:.2f}',ha='center',va='center',
                  fontsize=6, color='white' if abs(corr_reordered[i,j])>0.5 else 'black')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "correlation_heatmap.png"), dpi=600)
plt.close()

print("correlation_heatmap saved.")


# ---------------------------------------
# 3. PCA 
# ---------------------------------------


print("Computing pca...")
fig, ax_b = plt.subplots(figsize=(6, 6))

top_classes = df_merged[SUPERCLASS_COL].value_counts().head(8).index.tolist()
df_plot = df_merged[df_merged[SUPERCLASS_COL].isin(top_classes)]

palette = sns.color_palette(PALETTE, n_colors=len(top_classes))

# Scatter points
for cls, col in zip(top_classes, palette):
    m = df_plot[SUPERCLASS_COL] == cls
    ax_b.scatter(df_plot[m]['PC1'], df_plot[m]['PC2'],
                 c=[col], label=cls, s=25, alpha=0.75,
                 edgecolors='k', linewidths=0.3)

# Loadings arrows
for i, desc in enumerate(desc_cols):
    ax_b.arrow(0, 0, loadings_scaled[i,0], loadings_scaled[i,1],
               head_width=0.10, head_length=0.10, fc='red', ec='red')
    ax_b.text(loadings_scaled[i,0]*1.15, loadings_scaled[i,1]*1.15,
              desc, color='black', fontsize=10)

ax_b.set_title("PCA — Chemicals + Descriptor Loadings")
ax_b.axhline(0, color='gray', lw=0.5)
ax_b.axvline(0, color='gray', lw=0.5)


ax_b.legend(
    handles=[],
    labels=[var_text],
    loc="lower right",
    fontsize=8,
    frameon=True,
)
# Add interpretation text box
interpretation = f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%): "
interpretation += "Size/Complexity axis\n"
interpretation += f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%): "
interpretation += "Polarity/H-bonding axis\n"
interpretation += "Arrows show descriptor contributions"

ax_b.text(0.02, 0.98, interpretation, transform=ax_b.transAxes,
          fontsize=9, verticalalignment='top',
          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.savefig(os.path.join(FIG_DIR, "pca.png"), dpi=600)
plt.close()


print("Saved pca as pca.png")
# ---------------------------------------
# 4. Top Scaffolds as Horizontal Bar Chart 
# ---------------------------------------

print("Computing scaffolds...")
def get_scaffold(smiles):
    mol = safe_mol(smiles)
    if mol is None:
        return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf, canonical=True)
    except:
        return None

df_merged['Scaffold'] = df_merged[SMILES_COL].apply(get_scaffold)

# --- FIX: remove None scaffolds ---
scaffolds_clean = df_merged['Scaffold'].dropna()
scaf_counts = scaffolds_clean.value_counts().head(10)

# Create standalone figure
fig_e, ax_e = plt.subplots(figsize=(20, 10))

y_pos = np.arange(len(scaf_counts))

ax_e.barh(
    y_pos,
    scaf_counts.values,
    color=sns.color_palette('muted', n_colors=len(scaf_counts))
)

# --- FIX: shorten very long scaffold SMILES for visibility ---
labels_short = [s if len(s) < 70 else s[:67] + "..." for s in scaf_counts.index]

ax_e.set_yticks(y_pos)
ax_e.set_yticklabels(labels_short, fontsize=10)

ax_e.set_xlabel("Number of Compounds")
ax_e.set_title("Top Scaffolds", fontsize=20, pad=20)   # <-- FIX: add padding

# Invert y-axis so highest bar is at top
ax_e.invert_yaxis()

# Add labels to bars
for i, v in enumerate(scaf_counts.values):
    ax_e.text(
        v + max(scaf_counts.values) * 0.01,
        i,
        str(v),
        va='center',
        fontsize=12
    )

# --- FIX: force layout to include title & ylabel ---
plt.tight_layout(rect=[0, 0, 0.95, 0.98])

plt.savefig(os.path.join(FIG_DIR, "scaffold_distribution.png"), dpi=600)
plt.close()

print("Saved scaffold bar plot as scaffold_distribution.png")



# ---------------------------------------
# 5. Class distribution
# ---------------------------------------

print("Computing class distribution..")
fig_g, ax_g = plt.subplots(figsize=(6,5))
class_dist = df_merged[SUPERCLASS_COL].value_counts().head(10)
y_pos = np.arange(len(class_dist))
ax_g.barh(y_pos, class_dist.values, color=sns.color_palette('muted', n_colors=len(class_dist)))
ax_g.set_yticks(y_pos)
ax_g.set_yticklabels(class_dist.index)
ax_g.set_xlabel('Number of Compounds')
ax_g.set_title('G) Top Chemical Classes')
ax_g.invert_yaxis()
for i, v in enumerate(class_dist.values):
    ax_g.text(v + max(class_dist.values)*0.01, i, str(v), va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "class_distribution.png"), dpi=600)
plt.close()


print("Saved class distribution as class_distribution.png")




# ---------------------------------------
# 5. UMAP plot for Drug-likeness, QED
# ---------------------------------------



print("Generating standalone UMAP plot...")

# Prepare fingerprint matrix
fp_matrix = np.array(fps)

# Run UMAP
reducer = umap.UMAP(
    n_components=2,
    random_state=42,
    n_neighbors=min(15, len(fp_matrix)-1),
    min_dist=0.1,
    metric='jaccard'
)
try:
    umap_emb = reducer.fit_transform(fp_matrix)
except:
    reducer = umap.UMAP(n_components=2, random_state=42, metric='cosine')
    umap_emb = reducer.fit_transform(fp_matrix)

df_merged['UMAP1'] = umap_emb[:, 0]
df_merged['UMAP2'] = umap_emb[:, 1]

# Create figure for ONLY UMAP
plt.figure(figsize=(10, 8))

qed_values = df_merged['QED'].fillna(0.5).values
scatter = plt.scatter(
    df_merged['UMAP1'], df_merged['UMAP2'],
    c=qed_values, cmap='RdYlGn', s=25,
    alpha=0.8, edgecolors='k', linewidth=0.2,
    vmin=0, vmax=1
)

plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.title("UMAP Structural Clustering\n(Color: Drug-likeness, QED)", fontweight="bold")

cbar = plt.colorbar(scatter, ticks=[0, 0.33, 0.66, 1.0])
cbar.set_ticklabels(["Poor", "Fair", "Good", "Excellent"])
cbar.set_label("QED Score")
# Optional: density contour (remove if not needed)
compute_density_contours(
    df_merged['UMAP1'].values,
    df_merged['UMAP2'].values,
    plt.gca(),
    alpha=0.2
)

# SAVE PNG
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "umap_plot.png"), dpi=300)
plt.close()

print("UMAP figure saved as umap_plot.png")

# ---------------------------------------
# 6. property_boxplot_top_classes function (for only top few classes)
# ---------------------------------------
def property_boxplot_top_classes(df, prop, class_col=CLASS_COL, n_classes=8, outprefix=None):
    # Get top N classes by count
    top_class_counts = df[class_col].value_counts().head(n_classes)
    top_classes = top_class_counts.index.tolist()
    
    # Filter data to only include top classes
    data = df[df[class_col].isin(top_classes)].copy()
    data = data[[prop, class_col]].dropna(subset=[prop, class_col])
    
    # Order classes by count (descending)
    class_order = data[class_col].value_counts().index
    
    plt.figure(figsize=(12, 10))  # Taller figure for horizontal plot
    
    # Create HORIZONTAL boxplot
    sns.boxplot(y=class_col, x=prop, data=data, order=class_order)
    
    # Shorten very long class names for display
    y_labels = []
    for label in class_order:
        if len(label) > 40:
            y_labels.append(label[:37] + "...")
        else:
            y_labels.append(label)
    
    plt.yticks(ticks=range(len(class_order)), labels=y_labels)
    plt.ylabel(class_col)
    plt.xlabel(prop)
    plt.title(f"{prop} distribution by {class_col} (Top {n_classes} classes)")
    plt.tight_layout()

    if outprefix is None:
        outprefix = f"boxplot_{prop}_by_top{n_classes}_classes"

    #svg = os.path.join(FIG_DIR, outprefix + ".svg")
    png = os.path.join(FIG_DIR, outprefix + ".png")
    #plt.savefig(svg)
    plt.savefig(png, dpi=300)
    plt.close()
    print(f"Property plot for top {n_classes} classes saved: {png}")

# -----------------------
# Generate Fsp3 boxplots for TOP CLASSES ONLY
# -----------------------


# Create boxplots for top 8 classes (or adjust n_classes as needed)
#property_boxplot_top_classes(df_merged, 'MW', class_col=CLASS_COL, n_classes=8, outprefix="MW_by_top_classes")
#property_boxplot_top_classes(df_merged, 'LogP', class_col=CLASS_COL, n_classes=8, outprefix="LogP_by_top_classes")
property_boxplot_top_classes(df_merged, 'Fsp3', class_col=CLASS_COL, n_classes=8, outprefix="Fsp3_by_top_classes")

print("Saved boxplots as boxplot_{prop}_by_top{n_classes}_classes.png")
print("\nAnalysis complete!")
print(f"Results saved to {OUT_DIR}")