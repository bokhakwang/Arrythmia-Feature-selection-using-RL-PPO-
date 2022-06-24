import numpy as np
from sklearn import utils
import collections
class Environment:
    def __init__(self, data_x, data_y, ratio):
        self.data_x = data_x
        self.data_y = data_y.astype('int32')
        self.data_len = len(data_y)

        self.mask = np.ones( 21 )
        self.x  = np.zeros( 21 )
        self.y   = np.zeros( 1 )
        self.selected_feature_N  = np.zeros( 21 )
        self.selected_feature_S  = np.zeros( 21 )
        self.selected_feature_V  = np.zeros( 21 )
        self.selected_feature_F = np.zeros(21)
        self.temp_selected_feature  = np.zeros( 21 )
        self.dataset_idx = 0
        self. major_ratio = ratio
        self. score = 0

    def reset(self):# 에피소드 마다
        self.mask = np.ones( 21 )
        self.temp_selected_feature  = np.zeros( 21 )
        self.x, self.y = self._generate_sample()
        return self._get_state()

    def flush_selected_feature(self):
        self.temp_selected_feature = np.ones(21)

    def reset2(self):
        self.mask = np.ones( 21 )
        self.temp_selected_feature = np.zeros(21)
        self.x, self.y = self._generate_sample2()
        return self._get_state()        

        
    def step(self, action, attempt_class, correct_class):
        if action >= 3:
            self.mask[action - 3] += 0.01
            self.temp_selected_feature[action-3] += 1
            r = -0.05
            done = False

        if action < 3:  # 피쳐 고르는 것보다 분류하는 것이 더 좋을 경우, 앞에 있는 숫자가 클래시피케이션
            if action == self.y: #맞출때
                if action == 0:
                    r = (1 - (correct_class[0]/attempt_class[0]))
                    self.selected_feature_N = self.selected_feature_N + self.temp_selected_feature
                elif action ==1:
                    r = (1 - (correct_class[1]/attempt_class[1]))
                    self.selected_feature_S = self.selected_feature_S + self.temp_selected_feature
                elif action ==2:
                    r = (1 - (correct_class[2]/attempt_class[2]))
                    self.selected_feature_V = self.selected_feature_V + self.temp_selected_feature

                #r = 0
            else: #못맞출
                if self.y == 0:
                    r = ((correct_class[0]/attempt_class[0]) - 1)

                elif self.y ==1:
                    r = ((correct_class[1] / attempt_class[1]) - 1)

                elif self.y ==2:
                    r = ((correct_class[2] / attempt_class[2]) - 1)

                #r = -1
#			r[i] = REWARD_CORRECT if (action[i] >= self.y[i]-5) or (action[i] <= self.y[i]+5) else REWARD_INCORRECT
            #self._reset
            done = True

        s_ = self._get_state()

        return (s_, r,  done, {})


    def _generate_sample(self):
        #idx = np.random.randint(0, self.data_len)
        self.dataset_idx += 1
        x = self.data_x[self.dataset_idx-1]
        y = self.data_y[self.dataset_idx-1]
        
        return (x, y)
    def _generate_sample2(self):
        #idx = np.random.randint(0, self.data_len)
        x = self.data_x[self.dataset_idx-1]
        y = self.data_y[self.dataset_idx-1]
        return (x, y)

    def _get_state(self):
        x_ = self.x * self.mask
        #x_ = np.concatenate((x_, self.mask), 0)
        return x_
    def PrintFeature(self):
        print(self.selected_feature_N)
        print(self.selected_feature_S)
        print(self.selected_feature_V)
        #print(self.selected_feature_F)

    def Datasuffle(self):
        self.data_x, self.data_y = utils.shuffle(self.data_x, self.data_y)
        self.dataset_idx = 0

    def Flush(self):
        self.selected_feature_N = np.zeros( 21 )
        self.selected_feature_S = np.zeros( 21 )
        self.selected_feature_V = np.zeros( 21 )

        #self.selected_feature_F = np.zeros(21)
    def get_label(self):
        return self.data_y[self.dataset_idx-1]
    def get_y (self):
        return self.data_y