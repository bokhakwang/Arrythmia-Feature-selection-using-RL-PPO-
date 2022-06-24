import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from environment import Environment
from os import listdir, makedirs
import numpy as np
import random
import Testing
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import collections
from imblearn.under_sampling import ClusterCentroids ,TomekLinks,OneSidedSelection
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

from collections import OrderedDict
#Hyperparameters
learning_rate = 0.0001
gamma         = 1.0
lmbda         = 0.95
eps_clip      = 0.1 # cliping 할때 쓰인다
K_epoch       = 1 #20 step을 3번 반복
T_horizon     = 20
######## z-score Centering ##########
def Zscore(x):
    x = (x - x.mean())/x.std()
    return x
#####################################

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(21, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(256, 128)
        self.fc_pi = nn.Linear(128, 24)
        self.fc_v = nn.Linear(128, 1)
        # self.fc_pi1 = nn.Linear(18,100)
        # self.fc_pi2 = nn.Linear(100, 21)

        # self.fc_v1 = nn.Linear(18,50)
        # self.fc_v2 = nn.Linear(50, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc4(x))
        x = self.fc_pi(x)
        # x = F.relu(self.fc_pi1(x))
        # x = self.fc_pi2(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc5(x))
        v = self.fc_v(x)
        # x = F.relu(self.fc_v1(x))
        # v = self.fc_v2(x)
        return v

       
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)# s는 numpy array
            a_lst.append([a])# dimension이 맞춰주기 위해 괄호 안에 넣어서 append를 해줌.
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])#실제 a가 할 확율
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float).cuda(), torch.tensor(a_lst).cuda(), \
                                          torch.tensor(r_lst).cuda(), torch.tensor(s_prime_lst, dtype=torch.float).cuda(), \
                                          torch.tensor(done_lst, dtype=torch.float).cuda(), torch.tensor(prob_a_lst).cuda()
                                          
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self, step_on_off):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.cpu()
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:#recursive
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            #advantage_lst: advandtage 거꾸로 쌓인 버전
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).cuda()

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))
            #prob_a :oldpolicy의 확율
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())
            #value loss는 minimize, policy loss는 maximize
            #detach(): td_target이 만들어 지기까지 필요했던 앞의 그래프들를 다 떼어버린다는뜻
            #즉, gradient의 flow가 발생하지 않음!
            #떼어내지 않으면 target도 loss가 줄이는 방향으로 update가됨
            self.optimizer.zero_grad()
            #theta_prime = self.state_dict() - 0.001 * torch.gradient(loss)
            loss.mean().backward()
            if step_on_off:
                self.optimizer.step()

        return loss
      
             
def dataload(path_):
    x = np.ndarray((1,21))
    y = np.array([])

    path = path_
    subject_list = [f for f in listdir(path)]
    
    for subject in subject_list:
        train_path = path + subject
        npz = np.load(train_path)
        print(subject+" load")
        x = np.concatenate((x, npz['x']),axis=0)
        y = np.concatenate((y, npz['y']),axis=0)
    
    x = np.delete(x,[0],axis=0)
    x = Zscore(x)
    return x,y

def main():
    SEED = 111222
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(0)
    #x,y = dataload("./mit_TDfeature_npz/training/")
    #x, y = dataload("./mit_4class_npz/training/")
    #x, y = dataload("./incart_TDfeature_npz/training/")
    # npz = np.load("./mit_4class_npz/undersampled/undersampled.npz")
    npz = np.load("./incart_TDfeature_npz/undersampled/undersampled.npz")
    x = npz['x']
    y = npz['y']
    name_list= ['fc1.bias','fc1.weight', 'fc2.bias', 'fc2.weight','fc3.bias','fc3.weight',
                'fc4.bias','fc4.weight','fc5.bias','fc5.weight','fc_pi.bias','fc_pi.weight','fc_v.bias','fc_v.weight']
    #name_list = ['fc_pi1.bias', 'fc_pi1.weight', 'fc_pi2.bias', 'fc_pi2.weight','fc_v1.bias', 'fc_v1.weight','fc_v2.bias', 'fc_v2.weight']
    np.set_printoptions(precision=6, suppress=True)
    # cc = ClusterCentroids()
    # x, y = cc.fit_resample(x, y)
    # save_dict = {
    #      "x": x,
    #      "y": y,
    # }
    # makedirs("./incart_TDFeature_npz/undersampled", exist_ok=True)
    # np.savez("./mit_TDFeature_npz/undersampled/undersampled.npz", **save_dict)
    c = collections.Counter(y)
    print(sorted(collections.Counter(y).items()))
    ratio = (len(y)-c[0])/c[0]
    env = Environment(x,y,ratio)
    model = PPO()
    model = torch.load("./ppo_model_incart2.pt")
    model = model.cuda()
    acc_lst = []
    for epoch in range(100):
        #T_horizon     = 64 #몇 Time step동안 data를 모을지
        env.Datasuffle()
        y = env.get_y()
        y = y[:len(y) - (len(y)%20)]
        prediction = []
        attempt = 0.0
        attempt_class = [0.000001, 0.000001, 0.000001]
        correct_class = [0.0, 0.0, 0.0]
        correct = 0.0
        for n_epi in range(int(len(y)/20)):
            model_tempt = model.cuda()
            model_tempt.load_state_dict(model.state_dict())#state_dict 는 model의 weight 정보를 dictionary 형태로 담고있다.
            lst = []
            #env.flush_selected_feature()
            for T in range(20): #batch
                s = env.reset() # 첫 state 받기
                done = False
                while not done:  # task
                    for t in range(T_horizon):  # 실제로 T step 만큼만 data를 모으고 학습함
                        prob = model.pi(torch.from_numpy(s).float().cuda())
                        m = Categorical(prob)
                        a = m.sample().item()
                        s_prime, r, done, info = env.step(a,attempt_class, correct_class)
                        model.put_data((s, a, r, s_prime, prob[a].item(), done))#prob[a].item(): 나중에 ratio계산할때 쓰임
                        s = s_prime
                        if done:
                            prediction.append(a)
                            attempt += 1
                            if env.get_label() ==0:
                                attempt_class[0] += 1
                            if env.get_label() ==1:
                                attempt_class[1] += 1
                            if env.get_label() ==2:
                                attempt_class[2] += 1
                            # if env.get_label() == 3:
                            #     attempt_class[3] += 1
                            if a == env.get_label():
                                if a == 0:
                                    correct_class[0]+=1
                                elif a==1:
                                    correct_class[1]+=1
                                elif a==2:
                                    correct_class[2]+=1
                                # elif a==3:
                                #     correct_class[3]+=1
                            if r>=0:
                                correct += 1
                            break
                    model.train_net(True)
                s = env.reset2()
                done = False
                while not done:
                    prob = model.pi(torch.from_numpy(s).float().cuda())
                    m = Categorical(prob)
                    a = m.sample().item()
                    s_prime, r, done, info = env.step(a,attempt_class,correct_class)
                    model.put_data((s, a, r, s_prime, prob[a].item(), done))
                    s = s_prime

                model.train_net(False)
                lst.append(model.state_dict())
                model.load_state_dict(model_tempt.state_dict())

                if T==19 and n_epi%10==0:
                    print("#epoch: {}, # of episode :{},ACC: {:.5f}".format(epoch+1,n_epi,correct/attempt))
                    print("N: {:.5f}, S :{:.5f}, V: {:.5f}".format(correct_class[0] / attempt_class[0] ,
                                                                    correct_class[1] / attempt_class[1] ,
                                                                    correct_class[2] / attempt_class[2] ) )

            weights = [0 for i in range(14)]
            
            for i in range(0,20):
                for j in range(14):
                     weights[j] += weights[j] + lst[i][name_list[j]]
            for j in range(14):
                model.state_dict()[name_list[j]] =  model.state_dict()[name_list[j]] - 0.005 * weights[j]

          
        attempt_ = 0
        precision = precision_score(y, prediction, average=None)
        recall = recall_score(y, prediction, average=None)
        accuracy = accuracy_score(y,prediction)
        N_F1_score = 2 * (precision[0]*recall[0])/(precision[0]+recall[0])
        S_F1_score = 2 * (precision[1]*recall[1])/(precision[1]+recall[1])
        V_F1_score = 2 * (precision[2]*recall[2])/(precision[2]+recall[2])
        #F_F1_score = 2 * (precision[3] * recall[3]) / (precision[3] + recall[3])
        print("N F1_score:", N_F1_score)
        print("S F1_score:", S_F1_score)
        print("V F1_score:", V_F1_score)
        #print("V F1_score:", F_F1_score)
        print("Accuracy:",accuracy)
        print("MF1: ", ((N_F1_score + S_F1_score + V_F1_score) / 3) * 100)
        #np.save("./train_accuracy2.npy",np.array(acc_lst),)
        #print("F precision/recall:",precision[3], "/", recall[3])
        env.PrintFeature()
        env.Flush()
        torch.save(model, "./ppo_model_incart3.pt")
        if accuracy >0.79:
            val_acc = Testing.test("./ppo_model_incart3.pt")
            acc_lst.append(val_acc)

    np.save("./train_accuracy.npy", np.array(acc_lst))

if __name__ == '__main__':
    main()

