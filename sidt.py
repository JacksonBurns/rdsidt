import numpy as np
from rdkit import Chem
from tqdm import tqdm
from sklearn.metrics import mean_squared_error


class SIDTExtender:
    """
    Uses RDKit RWMol to chemically edit graphs (add atoms, add bonds, ring closures)
    and generates canonical SMARTS to ensure unique features.
    """
    def __init__(self, allowed_atoms=None, allowed_bonds=[1, 2, 3]):
        if allowed_atoms is None:
            allowed_atoms = [6, 7, 8, 9, 16, 17]
        self.allowed_atoms = allowed_atoms
        # Single, Double, Triple
        self.allowed_bonds = allowed_bonds
        
        self.bond_map = {
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE
        }

    def generate_extensions(self, parent_smarts):
        """
        Generates all valid 1-step chemical extensions of the parent subgraph.
        """
        parent_mol = Chem.MolFromSmarts(parent_smarts)
        if parent_mol is None:
            return []

        # Ensure we can compute properties on the subgraph
        try:
            parent_mol.UpdatePropertyCache(strict=False)
        except:
            pass

        candidates = []
        
        # 1. Atom Additions (Grow outward)
        candidates.extend(self._generate_atom_additions(parent_mol))
        
        # 2. Bond Modifications (Ring closures & Bond order increases)
        candidates.extend(self._generate_bond_modifications(parent_mol))

        # 3. Canonicalization & Deduplication
        unique_smarts = set()
        final_extensions = []

        for mol in candidates:
            try:
                # Use isomericSmiles=True in MolToSmarts to get distinct patterns
                canon_smarts = Chem.MolToSmarts(mol, isomericSmiles=True)
                if canon_smarts not in unique_smarts:
                    # Filter out invalid patterns immediately
                    if Chem.MolFromSmarts(canon_smarts) is not None:
                        unique_smarts.add(canon_smarts)
                        final_extensions.append(canon_smarts)
            except:
                continue

        return final_extensions

    def _generate_atom_additions(self, parent_mol):
        new_mols = []
        num_atoms = parent_mol.GetNumAtoms()

        for i in range(num_atoms):
            for atomic_num in self.allowed_atoms:
                for bond_order_int in self.allowed_bonds:
                    rw_mol = Chem.RWMol(parent_mol)
                    new_idx = rw_mol.AddAtom(Chem.Atom(atomic_num))
                    rw_mol.AddBond(i, new_idx, self.bond_map[bond_order_int])
                    new_mols.append(rw_mol)
        return new_mols

    def _generate_bond_modifications(self, parent_mol):
        new_mols = []
        num_atoms = parent_mol.GetNumAtoms()
        
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                bond = parent_mol.GetBondBetweenAtoms(i, j)
                
                if bond:
                    # Increase bond order
                    current_order = bond.GetBondTypeAsDouble()
                    if current_order < 3.0:
                        new_order_int = int(current_order) + 1
                        if new_order_int in self.bond_map:
                            rw_mol = Chem.RWMol(parent_mol)
                            rw_mol.RemoveBond(i, j)
                            rw_mol.AddBond(i, j, self.bond_map[new_order_int])
                            new_mols.append(rw_mol)
                else:
                    # Ring Closure
                    for bond_order_int in self.allowed_bonds:
                        # Mostly relevant for single/double bonds in rings
                        if bond_order_int <= 2: 
                            rw_mol = Chem.RWMol(parent_mol)
                            rw_mol.AddBond(i, j, self.bond_map[bond_order_int])
                            new_mols.append(rw_mol)
        return new_mols


class SIDTNode:
    """
    A node in the Subgraph Isomorphic Decision Tree.
    """
    def __init__(self, smarts="*", depth=0, value=0.0, stdev=np.nan, n_samples=0, mse=0.0):
        self.smarts = smarts              # Subgraph Pattern
        self.mol_pattern = Chem.MolFromSmarts(smarts)
        self.depth = depth
        self.value = value                # Predicted value (mean)
        self.stdev = stdev  # standard deviation of target values, i.e. uncertainty estimate
        self.n_samples = n_samples
        self.mse = mse
        
        self.left = None                  # YES Branch (Matches Extension)
        self.right = None                 # NO Branch (Does not match Extension)
        self.is_leaf = True


class RDSIDTRegressor:
    """
    Regressor that builds a decision tree where splits are defined by 
    chemical substructure extensions.
    """
    def __init__(self, allowed_atoms=None, max_depth=5, min_samples_split=2, min_impurity_decrease=1e-7, verbose=True):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.extender = SIDTExtender(allowed_atoms=allowed_atoms)
        self.verbose = verbose
        self.root = None
        self.pbar = None

    def __str__(self):
        return f"RDSIDTRegressor(max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, min_impurity_decrease={self.min_impurity_decrease})"

    def fit(self, smiles_list, y):
        if self.verbose:
            print("Starting SIDT training...")
            self.pbar = tqdm(total=len(smiles_list) // self.min_samples_split, desc="Building SIDT", unit="nodes")
        
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        
        # Filter out invalid molecules
        valid_idxs = [i for i, m in enumerate(mols) if m is not None]
        mols = [mols[i] for i in valid_idxs]
        y = np.array(y)[valid_idxs]
        
        # Initialize Root
        self.root = SIDTNode(
            smarts="*", 
            depth=0, 
            value=np.mean(y), 
            stdev=np.std(y),
            n_samples=len(y),
            mse=self._calculate_mse(y)
        )
        if self.verbose:
            self.pbar.update(1)
        self._split_node(self.root, mols, y)
        if self.verbose:
            self.pbar.close()
            print("Training complete.")
        return self

    def predict(self, smiles_list):
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        return np.array([self._predict_single(mol, self.root) for mol in mols])

    def _predict_single(self, mol, node):
        if mol is None: return node.value # Fallback
        
        if node.is_leaf:
            return (node.value, node.stdev)  # maybe just return the whole node?
        
        # In SIDT, the split question is: "Does it match the child's extended subgraph?"
        # The 'left' child holds the more specific (extended) subgraph.
        if mol.HasSubstructMatch(node.left.mol_pattern):
            return self._predict_single(mol, node.left)
        else:
            return self._predict_single(mol, node.right)

    def _split_node(self, node, mols, y):
        # Stop Criteria
        if (node.depth >= self.max_depth or 
            len(y) < self.min_samples_split or 
            node.mse == 0):
            return

        best_score = -float('inf')
        best_extension = None
        best_splits = None

        # Generate Extensions (The "mining" step)
        candidate_extensions = self.extender.generate_extensions(node.smarts)
        
        # Iterate over all chemically valid extensions
        for ext_smarts in candidate_extensions:
            ext_pat = Chem.MolFromSmarts(ext_smarts)
            if ext_pat is None: continue

            # Create Split Mask
            # Important: RDKit HasSubstructMatch checks if mol contains pattern
            matches = []
            for m in mols:
                matches.append(m.HasSubstructMatch(ext_pat))
            matches = np.array(matches)

            # If split is not useful (all yes or all no), skip
            if np.all(matches) or not np.any(matches):
                continue

            y_left = y[matches]
            y_right = y[~matches]

            # Calculate Variance Reduction
            curr_mse = node.mse
            w_left = len(y_left) / len(y)
            w_right = len(y_right) / len(y)
            
            mse_left = self._calculate_mse(y_left)
            mse_right = self._calculate_mse(y_right)
            
            weighted_mse = (w_left * mse_left) + (w_right * mse_right)
            improvement = curr_mse - weighted_mse

            if improvement > best_score:
                best_score = improvement
                best_extension = ext_smarts
                # Store the data splits to avoid recomputing
                mols_left = [m for i, m in enumerate(mols) if matches[i]]
                mols_right = [m for i, m in enumerate(mols) if not matches[i]]
                best_splits = (y_left, y_right, mols_left, mols_right)

        # Apply Split if valid
        if best_score > self.min_impurity_decrease and best_extension is not None:
            y_left, y_right, mols_left, mols_right = best_splits
            
            # Create Left Child (The Extension Match)
            node.left = SIDTNode(
                smarts=best_extension,
                depth=node.depth + 1,
                value=np.mean(y_left),
                stdev=np.std(y_left),
                n_samples=len(y_left),
                mse=self._calculate_mse(y_left),  # could store the above 3, just recalculate instead
            )
            
            # Create Right Child (The Remainder)
            # Inherits parent's SMARTS because it failed the check for the extension
            node.right = SIDTNode(
                smarts=node.smarts,
                depth=node.depth + 1,
                value=np.mean(y_right),
                stdev=np.std(y_right),
                n_samples=len(y_right),
                mse=self._calculate_mse(y_right)
            )
            # if pbar would increase above its total, increase the total
            if self.verbose:
                if self.pbar.n + 2 > self.pbar.total:
                    self.pbar.total += self.pbar.n
                self.pbar.update(2)
            
            node.is_leaf = False
            
            # Recurse
            self._split_node(node.left, mols_left, y_left)
            self._split_node(node.right, mols_right, y_right)

    def _calculate_mse(self, y):
        if len(y) == 0: return 0.0
        return mean_squared_error(y, [np.mean(y)] * len(y))

    def print_tree(self, node=None, indent=""):
        if node is None: node = self.root
        
        print(f"{indent}Structure: {node.smarts} | Value: {node.value:.2f} | Stdev: {node.stdev:.2f} | N: {node.n_samples}")
        
        if not node.is_leaf:
            print(f"{indent}  [Yes] Contains {node.left.smarts} ?")
            self.print_tree(node.left, indent + "    ")
            print(f"{indent}  [No]  (Remains {node.right.smarts})")
            self.print_tree(node.right, indent + "    ")


# small sklearn-compatible wrapper for integration with sklearn hpopt
class SIDTRegressorWrapper:
    def __init__(self, allowed_atoms=None, max_depth=5, min_samples_split=2, min_impurity_decrease=1e-7):
        self.allowed_atoms = allowed_atoms
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.model = RDSIDTRegressor(allowed_atoms=allowed_atoms, max_depth=max_depth, min_samples_split=min_samples_split, min_impurity_decrease=min_impurity_decrease, verbose=False)

    def get_params(self, deep=True):
        return {
            "allowed_atoms": self.allowed_atoms,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_impurity_decrease": self.min_impurity_decrease
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        self.model = RDSIDTRegressor(allowed_atoms=self.allowed_atoms, max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_impurity_decrease=self.min_impurity_decrease, verbose=False)
        return self

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return np.array([x[0] for x in self.model.predict(X)])  # return just the mean predictions for sklearn compatibility


if __name__ == "__main__":
    JMOL_TO_KCALMOL = 1 / 4184
    from rdkit.rdBase import BlockLogs

    bl = BlockLogs()

    import json

    with open("QM9_noncyclic_CHON_Hf298_Jmol_train.json") as f:
        train_data = json.load(f)
    
    with open("QM9_noncyclic_CHON_Hf298_Jmol_val.json") as f:
        val_data = json.load(f)

    train_smiles = [x[0] for x in train_data]
    train_y = [x[1]*JMOL_TO_KCALMOL for x in train_data]

    val_smiles = [x[0] for x in val_data]
    val_y = [x[1]*JMOL_TO_KCALMOL for x in val_data]
    atoms = set()
    for s in train_smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            for atom in mol.GetAtoms():
                atoms.add(atom.GetAtomicNum())
    print("Unique atomic numbers in training set:", atoms)

    # hyperparameter tuning
    from sklearn.model_selection import GridSearchCV, PredefinedSplit

    ps = PredefinedSplit(test_fold=[-1] * len(train_smiles) + [0] * len(val_smiles))

    param_grid = {
        "allowed_atoms": [atoms],
        "max_depth": [4, 8, 16, 32],
        "min_samples_split": [len(train_smiles) // 1000, len(train_smiles) // 500],
        "min_impurity_decrease": [1e-7, 1e-6, 1e-5],
    }

    wrapper = SIDTRegressorWrapper()
    grid_search = GridSearchCV(wrapper, param_grid, cv=ps, n_jobs=-1, refit=False, scoring="neg_mean_squared_error", verbose=2)
    grid_search.fit(train_smiles + val_smiles, train_y + val_y)

    print("Refitting model on best parameters:", grid_search.best_params_)

    model = RDSIDTRegressor(**grid_search.best_params_.values().mapping)
    model.fit(train_smiles + val_smiles, train_y + val_y)

    test_data = json.load(open("QM9_noncyclic_CHON_Hf298_Jmol_test.json"))
    smiles = [x[0] for x in test_data]
    y = [x[1]*JMOL_TO_KCALMOL for x in test_data]
    preds = model.predict(smiles)
    y_pred = [p[0] for p in preds]
    y_stdev = [p[1] for p in preds]

    from parity import parity_plot_with_intervals

    parity_plot_with_intervals(
        y,
        y_pred,
        y_err=y_stdev,
        title=str(model),
        outfile="sidt_parity.png",
        point_kwargs={"alpha": 0.3},
        errorbar_kwargs={"alpha": 0.1},
    )   
