#
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.svm import SVC
import pickle

dataset = open( "data_cdk.txt", "r" )
# remove header
dataset = [ line.rstrip().split("\t") for line in dataset ][1:]
mols = [ Chem.MolFromSmiles( line[1] ) for line in dataset ]
Y = []
# label active / non active I labeled compund that has pIC50 > 7.0 activity.
for line in dataset:
    if float( line[2] ) > 7.0:
        Y.append( 1 )
    else:
        Y.append( -1 )
Y = np.asarray( Y )
# calc fingerprint
fps = [ AllChem.GetMorganFingerprintAsBitVect( mol,2 ) for mol in mols ]

X = []
for fp in fps:
    arr = np.zeros( (1,) )
    DataStructs.ConvertToNumpyArray( fp, arr )
    X.append( arr )


cls = SVC( probability=True, C=100 )
cls.fit( X, Y )

f = open( "svcmodel.pkl", "wb" )
pickle.dump( cls, f )
f.close()
