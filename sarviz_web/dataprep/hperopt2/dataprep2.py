#
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import pickle
from hyperopt import hp, tpe, fmin

param_space_svc = {
                    "C" : hp.loguniform( "C", np.log(1), np.log(1000) ),
                    "gamma" : hp.loguniform( "gamma", np.log(0.0001), np.log(0.1)),

}


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

trainX, testX, trainY, testY = train_test_split( X, Y, test_size = 0.1 ,random_state = 40 )

def eval_func( args ):
    args["probability"] = True
    clf = SVC( **args )
    clf.fit( trainX, trainY )
    predicty = clf.predict( testX )
    score = accuracy_score( testY, predicty )
    print( args, score )
    return -score

best = fmin( eval_func, param_space_svc, algo = tpe.suggest, max_evals = 100 )
print("result")
best["probability"] = True
print(best)
cls = SVC(**best)


f = open( "svcmodel.pkl", "wb" )
pickle.dump( cls, f )
f.close()
