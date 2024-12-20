import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings("ignore")

# prepara o dicionário de path dos dados
pathDict = {}
pathDict[0] = 'data/C. elegans/'
pathDict[1] = 'data/D. melanogaster/'
pathDict[2] = 'data/H. sapiens/'

for index in range(0,3):
    resultList = []
    for k in range(4,5,1):
        vocab_size = 4 ** k
        print('Verificando eficácia do k = ' + str(k))
        data = np.loadtxt(pathDict[index] + 'freq-' + str(vocab_size) + '-bpe.csv', dtype=np.float64, delimiter=',')
        np.random.shuffle(data)
        num_vectors, vector_size = data.shape
        vector_size = vector_size - 1
        x = data[:,:vector_size]
        print(*x)
        y = data[:,vector_size:vector_size+1]
        print(*y)

        std_X = MinMaxScaler().fit_transform(x)
        print("shape: ", std_X.shape)

        final_test_acc = []
        final_test_Sn = []
        final_test_Sp =[]
        final_mcc = []
        final_auc = []
        tprs = []
        fprs = []

        n_splits=10
        sKF = StratifiedKFold(n_splits=n_splits, shuffle=False)
        i = 0

        for train_index, test_index in sKF.split(std_X,y):
            i +=1
            y_score=[]
            x_train = std_X[train_index]
            y_train = y[train_index]
            x_test = std_X[test_index]
            y_test = y[test_index]
            clf = LinearSVC(penalty='l2', loss='hinge', dual='auto', C=1.3)
            clf.fit(x_train, y_train.ravel())
            y_pre = clf.predict(x_test)  
            y_score = clf.decision_function(x_test)
            fpr, tpr, thresholds = roc_curve(y_test,y_score)
            tprs.append(tpr)
            fprs.append(fpr)
            roc_auc = auc(fpr,tpr)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pre).ravel()
            test_acc = (tn + tp)/(tn + fp + fn + tp ) 
            test_Sn = tp/(fn+tp)
            test_Sp = tn/(fp+tn)
            mcc = (tp*tn-fp*fn)/pow(((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)),0.5) 
            final_test_acc.append(test_acc)
            final_test_Sn.append(test_Sn)
            final_test_Sp.append(test_Sp)
            final_mcc.append(mcc)
            final_auc.append(roc_auc)
            print('train_Accuracy: {:.3f}'.format(clf.score(x_train, y_train)))
            print('test_Accuracy:%0.4f, test_Sn:%0.f, test_Sp:%0.4f, mcc:%0.4f, roc_auc:%0.4f' 
                                                   % (test_acc, test_Sn, test_Sp, mcc, roc_auc))
            print("confusion matrix:\n"+str (confusion_matrix(y_test, y_pre)))
            print('---------------------------------------------------')
               
        Final_test_acc = (sum(final_test_acc) / len(final_test_acc)) 
        Final_test_Sn = (sum(final_test_Sn) / len(final_test_Sn)) 
        Final_test_Sp = (sum(final_test_Sp) / len(final_test_Sp)) 
        Final_mcc =  (sum(final_mcc) / len(final_mcc)) 
        Final_auc = (sum(final_auc)/len(final_auc))  
        print('Final_test_Accuracy:%0.4f' % (Final_test_acc))
        print('Final_test_Sn:%0.4f' % (Final_test_Sn))
        print('Final_test_Sp:%0.4f' % (Final_test_Sp))
        print('Final_mcc:%0.4f' % (Final_mcc))
        print('Final_AUC:%0.4f' % (Final_auc))
        results = []
        results.append(Final_test_acc)
        results.append(Final_test_Sn)
        results.append(Final_test_Sp)
        results.append(Final_mcc)
        results.append(Final_auc)
        resultList.append(results)

        print('###########################################################')

    print('Imprimindo estatísticas para frequências de bpe.')
    # printar a lista em ordem
    resultFile = open(pathDict[index] + 'freq-results-bpe.txt','w')
    for i in range(len(resultList)):
        print(*resultList[i], file=resultFile)
    resultFile.close()