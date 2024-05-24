import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from statsmodels.stats import inter_rater as irr
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support



import util
from generate_graph import GenerateGraph
from regularization import Regularization

    


def holdout_data(sample_size, data):
    #train = data.sample(frac=sample_size, random_state=200)
    #test = data.drop(train.index).reset_index(drop=True)
    #train = train.reset_index(drop=True)
    
    train, test = train_test_split(data, train_size=sample_size, stratify=data['toxic'])
    
    return train, test

def run_experiment(n, reg_size, reg_method, classifier, train_data, test_data):
    
    
    kappa = []
    fscore = []
    precision = []
    recall = []
    
    for i in range(n):
        
        #train_data, test_data = holdout_data(sample_size,data)
        
        #Generate Graph
        graph, node_sentence = GenerateGraph(train_data, test_data).generate_graph_toxicity()
        
        train_labels = util.get_labels(train_data)
        dev_labels = util.get_labels(test_data)
        reg = Regularization().regularizer(graph, node_sentence, train_labels, 'features', pre_annotated=reg_size , method=reg_method)
        train = pd.read_csv('features/train.csv')
        dev = pd.read_csv('features/test.csv')
        
        clf = classifier

        clf.fit(train[['toxic', 'nontoxic']], np.ravel(train[['label']]))
        print('Training: ', clf.score(dev[['toxic', 'nontoxic']], dev_labels))
        y_pred = clf.predict(dev[['toxic', 'nontoxic']])
        
        print(classification_report(dev_labels, y_pred))
        
        
        toxic = test_data['toxic'].to_list()
        tn, fp, fn, tp = confusion_matrix(toxic, y_pred).ravel()
        
        k = np.around(cohen_kappa_score(toxic, y_pred),3)
        print('Kappa score:', k)
        #k = 2 * (tp*tn - fn*fp) / ((tp+fp) * (fp+tn) + (tp+fn) * (fn+tn))
        #print(k)
        kappa.append(k)
        
        p, r, f, s = precision_recall_fscore_support(toxic, y_pred, average='binary')
        fscore.append(f)
        precision.append(p)
        recall.append(r)
        
    return kappa, fscore, precision, recall    

if __name__ == '__main__':
    
    train_corpus = 'data/Toxic-BR_Curado.csv'
    test_corpus = 'data/ToLD-BR.csv'
    
    train_data = pd.read_csv(train_corpus, encoding='utf8', index_col=0, sep=';')
    test_data = pd.read_csv(test_corpus, encoding='utf8', sep=',')
    
    #train_size=0.8
    n = 10
    reg_size = 0.05
    reg_method = 'gfhf'
    
    #clf = MLPClassifier(random_state=1, max_iter=300)
    #clf = SGDClassifier(max_iter=1000, tol=1e-3)
    clf = HistGradientBoostingClassifier()
    
    #Pre-processing
    train_data = util.preprocessing(train_data)
    test_data = util.preprocessing(test_data)

    kappa_score, f_score, precision_score, recall_score = run_experiment(n, reg_size, reg_method, clf, train_data, test_data)    
    
    print('Mean Cohen Kappa: ' + str(np.mean(kappa_score)))
    print('Standard Deviation: ' + str(np.std(kappa_score)))
    
    print('Mean Fscore: ' + str(np.mean(f_score)))
    print('Standard Fscore: ' + str(np.std(f_score)))
    
    print('Mean Precision: ' + str(np.mean(precision_score)))
    print('Standard Precision: ' + str(np.std(precision_score)))
    
    print('Mean Cohen Recall: ' + str(np.mean(recall_score)))
    print('Standard Recall: ' + str(np.std(recall_score)))