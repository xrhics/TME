# python
# coding: utf-8
# author: xrhics

import result
import sys
from scipy import sparse
import os
import math
from treelib import Tree
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import f1_score


#dim=[100,200,300,400,500,600,700,800,900,1000]
dim=[500]
poi2id={}
category2id={}
poi2category={}
mymrr=[]
loss=[]
micro=[]
macro=[]
myset="\\tky"
if myset=="\\tky":
   n_pois=12605
   n_categories=103
else:
    n_pois=15632
    n_categories=138
DATA_DIR = 'E:\jupyter notebook\\tree-guided'+myset
#sampleRate = [10,20,30,40,50,60,70,80]
sampleRate = [10,20,30,40]
batch_size=1000
window_size=5
iteration_num = 10
l1 = [0.1]
#l2 = [1,0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001,0.00000001]
l2 = 0.000001


def indexFile(file,DATA_DIR,batch_size, window_size):
    saveid = 0
    count=0
    MLrows = []
    MLcols = []
    MCrows = []
    MCcols = []
    for l in open(file):
        trajectory = l.strip().split(',')
        for tuple in trajectory[0:]:
            poi=tuple.split('#')[0]
            if poi not in poi2id:
                poi2id[poi] = len(poi2id)
            # if  len(tuple.split('#'))<2:
            #     print(tuple)
            #     print(trajectory)
            category = tuple.split('#')[1]
            if category!='NULL' and category!='NUL':
                if category not in category2id:
                    category2id[category] = len(category2id)
        #find the target and the context
        poiids = []
        categoryids=[]
        for tuple in trajectory[0:]:
            poiids.append(poi2id[tuple.split('#')[0]])
            if  tuple.split('#')[1]=='NULL' or tuple.split('#')[1]=='NUL':
                categoryids.append(-1)
            else:
                categoryids.append(category2id[tuple.split('#')[1]])
        count=count+1
        # n_pois = len(poi2id)
        # n_categories = len(category2id)
        for ind_focus, poiid_focus in enumerate(poiids):
            #find the low bound and high bound of the context window
            ind_lo = max(0, ind_focus-window_size)
            ind_hi = min(len(poiids), ind_focus+window_size+1)

            # ind_c: the index of the context, ind_focus: index of target
            for ind_c in range(ind_lo, ind_hi):
                if ind_c == ind_focus:
                    continue
                '''diagonals are zeros or not'''
                if poiid_focus == poiids[ind_c]:
                    continue
                # the array of target poi
                MLrows.append(poiid_focus)
                # the array of corresponding context poi
                MLcols.append(poiids[ind_c])

                if categoryids[ind_c]!=-1:
                    # the array of target poi
                    MCrows.append(poiid_focus)
                    # the array of corresponding context category
                    MCcols.append(categoryids[ind_c])

        if count%batch_size == 0 and count != 0:
            np.save(os.path.join(DATA_DIR, 'intermediate\MLcoo_%d_%d.npy' % (saveid, count)), np.concatenate([np.array(MLrows)[:, None], np.array(MLcols)[:, None]], axis=1))
            np.save(os.path.join(DATA_DIR, 'intermediate\MCcoo_%d_%d.npy' % (saveid, count)), np.concatenate([np.array(MCrows)[:, None], np.array(MCcols)[:, None]], axis=1))
            saveid = saveid + batch_size
            MLrows = []
            MLcols = []
            MCrows = []
            MCcols = []
    np.save(os.path.join(DATA_DIR, 'intermediate\MLcoo_%d_%d.npy' % (saveid, count)), np.concatenate([np.array(MLrows)[:, None], np.array(MLcols)[:, None]], axis=1))
    np.save(os.path.join(DATA_DIR, 'intermediate\MCcoo_%d_%d.npy' % (saveid, count)), np.concatenate([np.array(MCrows)[:, None], np.array(MCcols)[:, None]], axis=1))
    return count


def write_poi2id(res_pt):
    print('write:', res_pt)
    wf = open(res_pt, 'w')
    for poi, poiid in sorted(poi2id.items(), key=lambda d:d[1]):
        wf.writelines('%d\t%s' % (poiid, poi))
        wf.writelines('\n')
    wf.close()

def write_category2id(res_pt):
    print('write:', res_pt)
    wf = open(res_pt, 'w')
    for category, categoryid in sorted(category2id.items(), key=lambda d:d[1]):
        wf.writelines('%d\t%s' % (categoryid, category))
        wf.writelines('\n')
    wf.close()
    

def _matrixw_batchML(lo, hi, matW):
    coords = np.load(os.path.join(DATA_DIR, 'intermediate\MLcoo_%d_%d.npy' % (lo, hi)))
    rows = coords[:, 0]
    cols = coords[:, 1]
    tmp = sparse.coo_matrix((np.ones_like(rows), (rows, cols)), shape=(n_pois, n_pois), dtype='float32').tocsr()
    matW = matW + tmp
    sys.stdout.flush()
    return matW

def _matrixw_batchMC(lo, hi, matW):
    coords = np.load(os.path.join(DATA_DIR, 'intermediate\MCcoo_%d_%d.npy' % (lo, hi)))
    rows = coords[:, 0]
    cols = coords[:, 1]
    tmp = sparse.coo_matrix((np.ones_like(rows), (rows, cols)), shape=(n_pois, n_categories), dtype='float32').tocsr()
    matW = matW + tmp
    sys.stdout.flush()
    return matW


def save_file(DATA_DIR,sampleRate,batch_size):
    trajectory_pt = DATA_DIR+'\\trajectoryTraining'+str(sampleRate)+'.txt'
    poi_pt = DATA_DIR+'\\tempData\poiIndex'+str(sampleRate)+'.csv'
    category_pt = DATA_DIR + '\\tempData\categoryIndex'+str(sampleRate)+'.csv'
    n_users =indexFile(trajectory_pt,DATA_DIR,1000,5)
    n_pois = len(poi2id)
    n_categories = len(category2id)
    print('n(categories)=', n_categories, 'n(poi)=', n_pois)
    write_poi2id(poi_pt)
    write_category2id(category_pt)

    start_idx = list(range(0, n_users, batch_size))
    end_idx = start_idx[1:] + [n_users]

    matrixML = sparse.csr_matrix((n_pois, n_pois), dtype='float32')
    matrixMC = sparse.csr_matrix((n_pois, n_categories), dtype='float32')

    for lo, hi in zip(start_idx, end_idx):
        matrixML = _matrixw_batchML(lo, hi, matrixML)
        matrixMC = _matrixw_batchMC(lo, hi, matrixMC)


    np.save(os.path.join(DATA_DIR, 'matrix\coordinate_co_binary_dataML'+str(sampleRate)+'.npy'), matrixML.data)
    np.save(os.path.join(DATA_DIR, 'matrix\coordinate_co_binary_indicesML'+str(sampleRate)+'.npy'), matrixML.indices)
    np.save(os.path.join(DATA_DIR, 'matrix\coordinate_co_binary_indptrML'+str(sampleRate)+'.npy'), matrixML.indptr)
    np.save(os.path.join(DATA_DIR, 'matrix\coordinate_co_binary_dataMC'+str(sampleRate)+'.npy'), matrixMC.data)
    np.save(os.path.join(DATA_DIR, 'matrix\coordinate_co_binary_indicesMC'+str(sampleRate)+'.npy'), matrixMC.indices)
    np.save(os.path.join(DATA_DIR, 'matrix\coordinate_co_binary_indptrMC'+str(sampleRate)+'.npy'), matrixMC.indptr)


def get_tree_group(category2id):
    tree_loc=Tree()
    data_loc = pd.read_csv("E:\jupyter notebook\\tree-guided\category0.csv")
    id2tag={}
    id2tag["root"]="root"
    for i in range(len(data_loc)):
        id2tag[data_loc.loc[i,'id']]=data_loc.loc[i,'name']
    tree_loc.create_node("root","root")
    for m in range(len(data_loc)):
        a=",".join(str(i) for i in data_loc.loc[m])
        arr=a.split(",")
        _id=arr[1]
        _name=arr[2]
        _parent=id2tag[(arr[3].split("$"))[0]]
        tree_loc.create_node(_name,_name,parent=_parent)
    tree_group=[]
    for key in category2id:
        sub_tree=tree_loc.subtree(key).all_nodes()
        a=[]
        for node in sub_tree:
            if node.tag in category2id:
                a.append(category2id[node.tag])
            else:
                if len(a)==0:
                    break;
        tree_group.append(a)
    return tree_group



def PMI(matrixML):
    count_row = np.asarray(matrixML.sum(axis=1)).ravel()
    count_column = np.asarray(matrixML.sum(axis=0)).ravel()
    n_pairs = matrixML.data.sum()
    n_row = matrixML.shape[0]
    # constructing the SPPMI matrix
    MII = matrixML.copy()
    for i in range(n_row):
        lo, hi, d, idx = get_row(MII, i)
        MII.data[lo:hi] = np.log(d * n_pairs / (count_row[i] * count_column[idx]))

    MII.data[MII.data < 0] = 0
    MII.eliminate_zeros()
    k_ns = 1
    MII_ns = MII.copy()
    if k_ns > 1:
        offset = np.log(k_ns)
    else:
        offset = 0.
    MII_ns.data -= offset
    MII_ns.data[MII_ns.data < 0] = 0
    MII_ns.eliminate_zeros()
    return MII_ns
def get_row(Y, i):
    lo, hi = Y.indptr[i], Y.indptr[i + 1]
    return lo, hi, Y.data[lo:hi], Y.indices[lo:hi]


def get_ml(m):
    #load matrix poi-poi
    data1 = np.load(os.path.join(DATA_DIR, 'matrix\coordinate_co_binary_dataML'+str(sampleRate[m])+'.npy'))
    indices1 = np.load(os.path.join(DATA_DIR, 'matrix\coordinate_co_binary_indicesML'+str(sampleRate[m])+'.npy'))
    indptr1 = np.load(os.path.join(DATA_DIR, 'matrix\coordinate_co_binary_indptrML'+str(sampleRate[m])+'.npy'))
    matrixML = sparse.csr_matrix((data1, indices1, indptr1), shape=(n_pois, n_pois))
    #see the sparseness
    ML_ns = PMI(matrixML)
    np.save(os.path.join(DATA_DIR, 'matrix\ML'+str(sampleRate[m])+'.npy'),ML_ns)
    return ML_ns
def get_mc(m):
    # load matrix poi-category
    data2 = np.load(os.path.join(DATA_DIR, 'matrix\coordinate_co_binary_dataMC'+str(sampleRate[m])+'.npy'))
    indices2 = np.load(os.path.join(DATA_DIR, 'matrix\coordinate_co_binary_indicesMC'+str(sampleRate[m])+'.npy'))
    indptr2 = np.load(os.path.join(DATA_DIR, 'matrix\coordinate_co_binary_indptrMC'+str(sampleRate[m])+'.npy'))
    matrixMC = sparse.csr_matrix((data2, indices2, indptr2), shape=(n_pois, n_categories))
        #see the sparseness
    MC_ns = PMI(matrixMC)
    np.save(os.path.join(DATA_DIR, 'matrix\MC'+str(sampleRate[m])+'.npy'),MC_ns)
    return MC_ns

def get_S(MC_ns):
    x,y=MC_ns.shape
    S = np.zeros((y,y),dtype=np.float32) 
    for i in range(y):
        for j in range(y):
            S[i][j]= np.dot(MC_ns[i],MC_ns[j])/(np.linalg.norm(MC_ns[i])*np.linalg.norm(MC_ns[j]))
    return S

def get_u(Gv):
    y=n_categories
    u = np.zeros((y,y),dtype=np.float32) 
    for i in range(y):
        for j in range(y):
            if i in Gv[j]:
                u[i][j]= 1/math.sqrt(len(Gv[j]))
    return u

def get_S(C):
    x,y=C.shape
    S = np.zeros((x,x),dtype=np.float32)
    for i in range(x):
        for j in range(x):
            S[i][j]= np.dot(C[i],C[j])/(np.linalg.norm(C[i])*np.linalg.norm(C[j]))
    return S

def get_u(Gv):
    y=n_categories
    u = np.zeros((y,y),dtype=np.float32)
    for i in range(y):
        for j in range(y):
            if i in Gv[j]:
                u[i][j]= 1/math.sqrt(len(Gv[j]))
    return u

def get_group_weight(MC_ns,Gv):
    S=get_S(MC_ns)
    u=get_u(Gv)
    result=[]
    for i in range(len(u)):
        result.append(np.dot(np.dot(u[i].T,S),u[i]))
    result=np.array(result)
    return result

def queryTop1(queryEmbedding,categoryVectorList,trueCategory,true_y,test_y):    ##choose top1 category
    rank=0
    ranklist={}
    num1=len(categoryVectorList)
    a=np.array(queryEmbedding)
    aa=np.dot(a,a.T)
    for i in range(num1):
        categoryVector=categoryVectorList[i]
        b=np.array(categoryVector)
        ab=np.dot(a,b.T)
        bb=np.dot(b,b.T)
        cosSim=0
        if aa!=0 and bb!=0:
            cosSim = ab / (sqrt(aa) * sqrt(bb))
        ranklist.update({i:cosSim})
    ranklist=sorted(ranklist.items(),key = lambda x:x[1],reverse = True)
    test_y.append(ranklist[0][0])
    i=0
    t=int(trueCategory)
    true_y.append(t)
    for key in ranklist:
        if int(key[0])==t:
            rank=i+1
            break
        i+=1
    score=1/rank
    return score


def evaluate(categoryembedding,poiembedding,testPath,poiNameIndexPath,categoryNameIndexPath):
    poiNameIndexMap={}
    categoryNameIndexMap={}
    testMap={}
    categoryVectorList={}
    poiVectorList={}
    test_y = []
    true_y=[]
    data=pd.read_csv(poiNameIndexPath,header=None)
    num=len(data)
    for i in range(num):
        a=data.loc[i,0].split("\t")
        poiNameIndexMap.update({a[1]:a[0]})
    data=pd.read_csv(categoryNameIndexPath,header=None)
    num=len(data)
    for i in range(num):
        a=data.loc[i,0].split("\t")
        categoryNameIndexMap.update({a[1]:a[0]})
    with open(testPath, 'r') as file_to_read:
        while 1:
            lines = file_to_read.readline()
            if not lines:
                break
            a=lines.split("#")
            testMap.update({a[0]:a[1].split("\n")[0]})
    for i in range(len(categoryembedding)):
        categoryVectorList.update({i:list(categoryembedding[i])})
    for i in range(len(poiembedding)):
        poiVectorList.update({i:list(poiembedding[i])})
    mrr=0
    count=0
    for key in testMap:
        poiIndex=int(poiNameIndexMap[key])
        trueCategoryIndex=categoryNameIndexMap[testMap[key]]
        poiVector=poiVectorList[poiIndex]
        score=queryTop1(poiVector, categoryVectorList, trueCategoryIndex,true_y,test_y)
        mrr += score
        count=count+1
    mrr = mrr / count
    print(mrr)
    mymrr.append(mrr)
    f1_micro = f1_score(true_y, test_y,  average='micro')
    print(f1_micro)
    f1_macro = f1_score(true_y, test_y, average='macro')
    print(f1_macro)
    micro.append(f1_micro)
    macro.append(f1_macro)
    return true_y,test_y


def getLt(Ml,Mc,Lc,C,l1):
    part=np.dot(Ml,Lc)+np.dot(Mc,C)
    view_dim=C.shape[1]
    inv_part=np.dot(Lc.T,Lc)+np.dot(C.T,C)+l1*np.eye(view_dim,dtype=np.float32)
    Lt= np.dot(part,np.linalg.inv(inv_part))    
    return Lt


def getLc(Ml,Y,Lt,C,l1):
    view_dim=C.shape[1]
    part=np.dot(Ml.T,Lt)+np.dot(Y,C)
    inv_part=np.dot(Lt.T,Lt)+np.dot(C.T,C)+l1*np.eye(view_dim,dtype=np.float32)
    Lc= np.dot(part,np.linalg.inv(inv_part))    
    return Lc


def getD(C,ev, Gv):
    task_num,d = C.shape
    nodes_num = ev.__len__()
    D = np.zeros((d,d),dtype= np.float32)
    Q = np.zeros((d,nodes_num),dtype= np.float32)
    q_total = 0
    for v in range(nodes_num):
        c_v =  C[Gv[v],:].T
        for k in range(d):
            Q[k,v] = ev[v] * np.linalg.norm(c_v[k],2)
            q_total += Q[k,v]
    Q = Q / q_total
    for k in range(d):
        for v in range(nodes_num):
            if Q[k,v] !=0:
                for t in range(task_num):
                    if t in Gv[v]:
                        D[k,k] += (ev[v] * ev[v]) / Q[k,v]
    
    return D,q_total


def getC(Mc,Y,Lc,Lt,Q,l1,l2):      
    view_dim=Q.shape[0]
    part=np.dot(Mc.T,Lt)+np.dot(Y.T,Lc)
    inv_part = np.dot(Lt.T,Lt)+np.dot(Lc.T,Lc)+l1*np.eye(view_dim,dtype=np.float32)+l2 * Q
    C = np.dot(part,np.linalg.inv(inv_part))
    return C

def our_optimize(Ml,Mc,i,Y, d, Gv,  iteration_num, l1,l2):
    DEFAULT_TOLERANCE = 0.5
    
    ##Lt,Lc,C  initialization
    poi_num=Mc.shape[0]
    task_num=n_categories
    
    Lt=np.random.rand(poi_num,d)
    Lc=np.random.rand(poi_num,d)
    C=np.random.rand(task_num,d)
    obj_ex = 0
    for iter in range(iteration_num):    #Iterate 10 times
        print("------第"+str(iter+1)+"次迭代------")
        ev = get_group_weight(C,Gv)     
        Lt=getLt(Ml,Mc,Lc,C,l1)
        Lc=getLc(Ml,Y,Lt,C,l1)
        D, q_total = getD(C,ev, Gv)
        C=getC(Mc,Y,Lc,Lt,D,l1,l2)
        true_y,test_y=evaluate(C,Lc,DATA_DIR+"\\testPOI"+str(sampleRate[i])+".txt",DATA_DIR+"\\tempData\poiIndex"+str(sampleRate[i])+".csv",DATA_DIR+"\\tempData\categoryIndex"+str(sampleRate[i])+".csv")
        # object function
        f1=Ml-np.dot(Lt,Lc.T)
        first_term =np.sqrt(np.sum(np.multiply(f1,f1)))
    
        f2=Mc-np.dot(Lt,C.T)
        second_term =np.sqrt(np.sum(np.multiply(f2, f2)))
       
        f3=Y-np.dot(Lc,C.T)
        third_term =np.sqrt(np.sum(np.multiply(f3, f3)))
       
        fourth_term = l2 * q_total
        fifth_term = l1*(np.linalg.norm(Lt,2)+np.linalg.norm(Lc,2)+np.linalg.norm(C,2))
        
        # object value
        obj_value = first_term + second_term + third_term + fourth_term + fifth_term  #计算T值
        loss.append(obj_value)
        obj_error = np.abs(obj_value - obj_ex)   #更新迭代的值差
        
        print('%d ===== %f ===== %f' %(iter, obj_value, obj_error))
        if obj_error < DEFAULT_TOLERANCE:
            print('training done')
            break
        obj_ex = obj_value
    
    return Lt, Lc, C,true_y,test_y


def get_train_set(m):
    train_set_x=[]
    train_set_y=[]
    file=DATA_DIR+"\\trajectoryTraining"+str(sampleRate[m])+".txt"
    for l in open(file):
        trajectory = l.strip().split(',')
        for tuple in trajectory[0:]:
            train_set_x.append(poi2id[tuple.split('#')[0]])
            if tuple.split('#')[1]=='NULL':
                train_set_y.append(-1)
            else:
                train_set_y.append(category2id[tuple.split('#')[1]])
    return train_set_x,train_set_y

def our_model():
    for i in range(len(sampleRate)):
        file=DATA_DIR+"/trajectoryTraining"+str(sampleRate[i])+".txt"
        indexFile(file,DATA_DIR,1000,window_size)
        save_file(DATA_DIR,sampleRate[i],batch_size)  #Data initialization and save
        Gv = get_tree_group(category2id)
        train_set_X,train_set_Y=get_train_set(i)
        for j in range(len(train_set_X)):
            poi2category[train_set_X[j]]=train_set_Y[j]
        Mc =  get_mc(i)
        Ml = get_ml(i)
        Ml=Ml.toarray()
        Mc=Mc.toarray()
        task_indices = np.unique(train_set_Y)
        task_dim = len(task_indices)-1  #category number
        Y = np.zeros((len(poi2id),task_dim),dtype=np.float32) 
        for j in range(len(poi2id)):
            indices=poi2category[j]
            if indices==-1:
                continue
            Y[j,indices] = 1                           #Y initialization
        for j in l1:
            for d in dim:
                print(str(j)+"---"+str(d))
                Lt,Lc,C,true_y,test_y = our_optimize(Ml,Mc,i,Y, d,Gv, iteration_num, j,l2)
                np.savetxt(DATA_DIR+'\matrix\Lt'+str(sampleRate[i])+'-'+str(d)+'-'+str(j)+'-'+str(l2)+'.txt', Lt)
                np.savetxt( DATA_DIR+'\matrix\Lc'+str(sampleRate[i])+'-'+str(d)+'-'+str(j)+'-'+str(l2)+'.txt', Lc)
                np.savetxt( DATA_DIR+'\matrix\C'+str(sampleRate[i])+'-'+str(d)+'-'+str(j)+'-'+str(l2)+'.txt', C)
                


if __name__ == "__main__":
    our_model()
    print(mymrr)
