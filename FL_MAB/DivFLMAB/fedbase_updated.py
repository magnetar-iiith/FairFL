import numpy as np
import tensorflow as tf
from tqdm import tqdm
from numpy import linalg as LA
from flearn.models.client import Client
from flearn.utils.model_utils import Metrics
from flearn.utils.tf_utils import process_grad

from sklearn.metrics import pairwise_distances

class BaseFedarated(object):
    def __init__(self, params, learner, dataset):
        # transfer parameters to self
        for key, val in params.items(): setattr(self, key, val);

        # create worker nodes
        tf.reset_default_graph()
        self.client_model = learner(*params['model_params'], self.inner_opt, self.seed)
        self.clients = self.setup_clients(dataset, self.client_model)
        print('{} Clients in Total'.format(len(self.clients)))
        self.latest_model = self.client_model.get_params()

        # initialize system metrics
        self.metrics = Metrics(self.clients, params)
        
        self.norm_diff = np.zeros((len(self.clients), len(self.clients)))
        self.norm_diff2 = np.zeros((len(self.clients), len(self.clients))) 

    def __del__(self):
        self.client_model.close()

    def setup_clients(self, dataset, model=None):
        '''instantiates clients based on given train and test data directories

        Return:
            list of Clients
        '''
        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]
        all_clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
        return all_clients


    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []

        for c in self.clients:
            ct, cl, ns = c.train_error_and_loss() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
            
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]

        return ids, groups, num_samples, tot_correct, losses


    def show_grads(self):  
        '''
        Return:
            gradients on all workers and the global gradient
        '''

        model_len = process_grad(self.latest_model).size
        global_grads = np.zeros(model_len)  

        cc = 0
        samples=[]

        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            num_samples, client_grads = c.get_grads(model_len) 
            #num_samples, client_grads = c.get_grads(self.latest_model) 
            samples.append(num_samples)
            # serial_cl_grads = process_grad(client_grads)
            if cc == 0:
                intermediate_grads = np.zeros([len(self.clients) + 1, len(client_grads)])
            # print(client_grads)
            
            # serial_cl_grads = client_grads
            global_grads = np.add(global_grads, client_grads * num_samples)
            intermediate_grads[cc] = client_grads
            # print('serial_cl_grads shape', serial_cl_grads.shape)
            cc += 1

        global_grads = global_grads * 1.0 / np.sum(np.asarray(samples)) 
        intermediate_grads[-1] = global_grads

        return intermediate_grads
 
  
    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        return ids, groups, num_samples, tot_correct

    def save(self):
        pass
    
    #def select_cl_submod(self, round, num_clients, N_i, True ):
    def select_cl_submod(self, round, num_clients, N_i,C_i,sum_grad_sum, stochastic_greedy ):
        print ("entered in cl_submod")
        
        # Get per-client gradients as well as global gradient 
        #all_grads = self.show_grads()

        # if round == 0 or self.m_interval == 1:  # at the first iteration or when m=1, collect gradients from all clients
        #     self.all_grads = np.asarray(self.show_grads()[:-1])
            
        #print("New shape", np.shape(self.all_grads))
        # for i in range(len(self.clients)):
        #     if all_grads[i].shape[0] > 8:
        #         print('all_grads[i]', all_grads[i].shape)

        # cc = 0
        # # Get per-client model parameters (most recent available at client)
        # for c in self.clients:
        #     #client_params = c.get_params()
        #     #cl_params = np.append(client_params[0].flatten(), client_params[1])
        #     # cl_params = c.updatevec
        #     if cc == 0:
        #         all_cl_params = np.zeros([len(self.clients), len(c.updatevec)])
        #     all_cl_params[cc] = c.updatevec
        #     cc += 1
        
        # Calculate gradient normed difference
        #m_interval = 1 # parameter m in the paper; (integer >=1)
        # if round % self.m_interval == 0:
        # for i in range(len(self.clients)):
        #     for j in range(len(self.clients)):
        #         # print(all_grads[i], all_grads[j])
        #         self.norm_diff[i,j] = np.linalg.norm(all_grads[i]-all_grads[j])
        #         self.norm_diff2[i,j] = np.linalg.norm(all_cl_params[i]-all_cl_params[j])

        # if round <= 1 or self.sim_metric == "grad":
        #if round % self.m_interval == 0: # Update similarity matrix only after m_interval epochs
            #self.norm_diff = pairwise_distances(all_grads[:-1], metric="euclidean")
        # self.norm_diff = pairwise_distances(self.all_grads, metric="euclidean")
        # np.fill_diagonal(self.norm_diff, self.norm_diff.max())
        # np.fill_diagonal(self.norm_diff, self.norm_diff.min(1) * 0.5)
        # np.fill_diagonal(self.norm_diff, 0)
        # else:
        #     self.norm_diff = pairwise_distances(all_cl_params, metric="euclidean")
        #     np.fill_diagonal(self.norm_diff, 0.1)

        if stochastic_greedy:
            print ("entered")
            SUi,N_i,C_i,sum_grad_sum = self.stochastic_greedy(num_clients,0.1,N_i,C_i,sum_grad_sum,round )
            
        else:
            SUi = self.lazy_greedy(num_clients)
        # print('Set Diff:', SUi0.difference(SUi), SUi.difference(SUi0))
                
        indices = np.array(list(SUi))
        selected_clients = np.asarray(self.clients)[indices]
        
        return indices, selected_clients,self.all_grads, N_i,C_i,sum_grad_sum
        #return indices,N_i, selected_clients, self.all_grads
        #return indices, selected_clients, self.all_grads,N_i
        
        
#############################################################################################################################################
    def stochastic_greedy(self, num_clients, subsample,N_i,C_i,sum_grad_sum, round ):
        
        

        V_set = set(range(len(self.clients)))
        
        
        SUi = set()
        prev_selected = []

        V_set = list(V_set)

        val = []
        for idx in range (len(self.clients)):
            if N_i[idx] == 0:
                val.append(0)
#            val.append (N_i[idx])
            
            else:
                value = np.sqrt( (2* np.log(round) )  / (N_i[idx]) )
                val.append (value)

        val=np.array(val)
        #print('Val is of the client to subtract is:',val)
        
        
        print ("sum_grad_sum", sum_grad_sum)
        print ("(self.all_grads[:, V_set].sum(0)", self.all_grads[:, V_set].sum(0))
        
        sum_grad_sum = sum_grad_sum + (self.all_grads[:, V_set].sum(0))
        
        
        
        #result_sum = self.all_grads[:, V_set].sum(0)
        print("chaceking shape", self.all_grads[:, V_set].shape) 
        print ("checking shape of sum_grad", )
        if round == 0:
            SUi = V_set
            
            for idx in range (len(self.clients)):
                N_i[ idx ] = N_i[ idx] + 1
                
            
            
        
        else:
            
            for ni in range(num_clients):  
                
                if ni == 0 :
                    
                    marg_util = ((self.all_grads[:, V_set].sum(0)) + sum_grad_sum)/round
                    print ("marginal utility ", marg_util)
                    marg_util_normalized = marg_util - val[V_set]
                    print ("marginal utility normalized", marg_util_normalized)
                    i = marg_util_normalized.argmin()
                    print ("selected index ", i)
                    client_min = marg_util_normalized [V_set[i]]
                    prev_selected.append(client_min)
                    
                else:
                    print ("we are selecting clients", ni+1)
                    ans =  (prev_selected)
                    marg_util = ((self.all_grads[:,V_set].sum(0)) + sum_grad_sum [V_set] )/round
                    marg_util_normalized = (marg_util-val[V_set])
                    temp =[]
                    tempj=[]
                    for i in range(len(ans)):
                        tem =[]
                        for j in range(len(marg_util_normalized)):
                            tem.append(ans[i]-marg_util_normalized[j])

                        temp.append(np.min(tem))
                        tempj.append(np.argmin(tem))
                    client_min_R = np.array(temp)
                    
                    print ("checking values", client_min_R)
                    
                    #client_min_R = temp.tolist()

                    
                    marg_util_normalized = client_min_R
                    print ("to check with",marg_util_normalized )
                    
                    
                    i = marg_util_normalized.argmin()
                    client_min = tempj[i] #client_min_R[i]
                    prev_selected.append(client_min)
                    
                    print ("run",client_min)
                
                #for client_idx, util in zip(V_set, marg_util_normalized):
                    #normalized_marg_util_dict[client_idx] = util

                    
                    
                    
                N_i[ V_set[i] ] = N_i[ V_set[i]   ] + 1
                SUi.add(V_set[i])      
                V_set.remove(V_set[i])
            
            print('Number of times client get selected is as as follows ',N_i)
            
            #sum_grad_sum = sum_grad_sum_1 

        return SUi, N_i, C_i, sum_grad_sum
    
        
        
            
            
#####################################################################################################################################################     
    
        

    def greedy(self, num_clients):
        # initialize the ground set and the selected set
        print('entered  to greedy ')
        V_set = set(range(len(self.clients)))
        SUi = set()
        for ni in range(num_clients):
            R_set = list(V_set)
            if ni == 0:
                marg_util = self.norm_diff[:, R_set].sum(0)
                i = marg_util.argmin()
                client_min = self.norm_diff[:, R_set[i]]
            else:
                client_min_R = np.minimum(client_min[:,None], self.norm_diff[:,R_set])
                marg_util = client_min_R.sum(0)
                i = marg_util.argmin()
                client_min = client_min_R[:, i]
            # print(R_set[i], marg_util[i])
            SUi.add(R_set[i])
            V_set.remove(R_set[i])
        return SUi

    def lazy_greedy(self, num_clients):
        # initialize the ground set and the selected set
        V_set = set(range(len(self.clients)))
        SUi = set()

        S_util = 0
        marg_util = self.norm_diff.sum(0)
        i = marg_util.argmin()
        L_s0 = 2. * marg_util.max()
        marg_util = L_s0 - marg_util
        client_min = self.norm_diff[:,i]
        # print(i)
        SUi.add(i)
        V_set.remove(i)
        S_util = marg_util[i]
        marg_util[i] = -1.
        
        while len(SUi) < num_clients:
            argsort_V = np.argsort(marg_util)[len(SUi):]
            for ni in range(len(argsort_V)):
                i = argsort_V[-ni-1]
                SUi.add(i)
                client_min_i = np.minimum(client_min, self.norm_diff[:,i])
                SUi_util = L_s0 - client_min_i.sum()

                marg_util[i] = SUi_util - S_util
                if ni > 0:
                    if marg_util[i] < marg_util[pre_i]:
                        if ni == len(argsort_V) - 1 or marg_util[pre_i] >= marg_util[argsort_V[-ni-2]]:
                            S_util += marg_util[pre_i]
                            # print(pre_i, L_s0 - S_util)
                            SUi.remove(i)
                            SUi.add(pre_i)
                            V_set.remove(pre_i)
                            marg_util[pre_i] = -1.
                            client_min = client_min_pre_i.copy()
                            break
                        else:
                            SUi.remove(i)
                    else:
                        if ni == len(argsort_V) - 1 or marg_util[i] >= marg_util[argsort_V[-ni-2]]:
                            S_util = SUi_util
                            # print(i, L_s0 - S_util)
                            V_set.remove(i)
                            marg_util[i] = -1.
                            client_min = client_min_i.copy()
                            break
                        else:
                            pre_i = i
                            SUi.remove(i)
                            client_min_pre_i = client_min_i.copy()
                else:
                    if marg_util[i] >= marg_util[argsort_V[-ni-2]]:
                        S_util = SUi_util
                        # print(i, L_s0 - S_util)
                        V_set.remove(i)
                        marg_util[i] = -1.
                        client_min = client_min_i.copy()
                        break
                    else:
                        pre_i = i
                        SUi.remove(i)
                        client_min_pre_i = client_min_i.copy()
        return SUi

    def select_clients(self, round, num_clients=20):
        '''selects num_clients clients weighted by number of samples from possible_clients
        
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''

        num_clients = min(num_clients, len(self.clients))
        np.random.seed(round)  # make sure for each comparison, we are selecting the same clients each round
        indices = np.random.choice(range(len(self.clients)), num_clients, replace=False)
        return indices, np.asarray(self.clients)[indices]

    def aggregate(self, wsolns):
        total_weight = 0.0
        base = [0]*len(wsolns[0][1])

        for (w, soln) in wsolns:  # w is the number of local samples
            total_weight += w
            for i, v in enumerate(soln):
                base[i] += w*v.astype(np.float64)
    
        averaged_soln = [v / total_weight for v in base]

        return averaged_soln

    def aggregate_simple(self, wsolns):
        total_weight = 0.0
        base = [0]*len(wsolns[0][1])

        for (w, soln) in wsolns:  # w is the number of local samples
            total_weight += 1
            for i, v in enumerate(soln):
                base[i] += v.astype(np.float64)
    
        averaged_soln = [v / total_weight for v in base]

        return averaged_soln
    
    def aggregate_submod(self, wsolns, gammas):
        total_weight = 0.0
        total_gamma = 0.0
        base = [0]*len(wsolns[0][1])
        
        gammas = list(gammas)
        for (wsols, gamma) in zip(wsolns, gammas):
            total_weight += wsols[0]
            for i, v in enumerate(wsols[1]):
                base[i] += gamma*wsols[0]*v.astype(np.float64)
            total_gamma +=gamma
    
        averaged_soln = [v / (total_weight*total_gamma) for v in base]

        return averaged_soln

