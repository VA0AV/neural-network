import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class SNN:
    def __init__(self,n_hidden):
        """
        1.双层标准神经网络实现逻辑回归
        2.n_hidden是隐藏层神经元个数，需要手动调节
        3.n_output是输出层神经元个数，因为要实现逻辑回归功能，所以默认为1
        4.默认初始的if_have_trained(是否已经进行训练)为False
        5.默认初始的if_standardize(是否进行标准化)为False
        6.默认初始的if_normalize(是否进行归一化)为False
        7.默认初始的if_L2_regularize(是否进行L2正则化)为False
        8..默认初始的if_need_ReLU(是否需要在隐藏层使用ReLU激活函数)为False
        """
        self.n_hidden=n_hidden
        self.n_output=1

        self.if_have_trained=False
        self.if_standardize=False
        self.if_normalize=False
        self.if_L2_regularize=False
        self.if_need_ReLU=False

        self.true_percentage_list=[]

    def sigmoid(self,z):
        """
        1.sigmoid激活函数,将z值映射到(0,1)
        """
        a=1/(1+np.exp(-z))

        return a

    def ReLU(self,z):
        """
        1.ReLU激活函数
        """
        a=np.maximum(0,z)

        return a

    def initialize(self):
        """
        1.权重(w)和偏置(b)初始化
        2.这时候有隐藏层了，而且隐藏层神经元一般不止一个，b1最好是一个多维的列向量
          而输出层还是只有一个神经元，b2依然直接设置为0
        3.返回一个存储参数的parameters字典
        """
        w1=np.random.randn(self.n_input,self.n_hidden)*0.01
        b1=np.zeros((self.n_hidden,1))
        w2=np.random.randn(self.n_hidden,self.n_output)*0.01
        b2=np.zeros((self.n_output,1))

        parameters={'w1':w1,
                    'b1':b1,
                    'w2':w2,
                    'b2':b2}

        return parameters

    def standardize(self,X_train):
        """
        1.对数据进行标准化(无量纲化)，提高计算速度(实践发现可以略微提高准确率)
        2.将训练集特征值的平均值和标准差设置为类属性
          以便用相同的参数对测试集进行标准化
        """
        self.mean=np.mean(X_train)
        self.std=np.std(X_train)

        X_train=(X_train-np.mean(X_train))/np.std(X_train)
        
        return X_train

    def standardize_predict(self,X_test):
        """
        1.测试集专用的标准化方法
        2.对测试集特征值进行标准化
        """
        X_test=(X_test-self.mean)/self.std

        return X_test

    def normalize(self,X_train):
        """
        1.对训练集特征值进行归一化(无量纲化)，提高计算速度(实践发现可以略微提高准确率)
        2.将训练集特征值的最大值和最小值设置为类属性
          以便用相同的参数对测试集进行归一化
        """
        self.min=np.min(X_train)
        self.max=np.max(X_train)

        X_train=(X_train-np.min(X_train))/(np.max(X_train)-np.min(X_train))

        return X_train

    def normalize_predict(self,X_test):
        """
        1.测试集专用的归一化方法
        2.对测试集特征值进行归一化
        """
        X_test=(X_test-self.min)/(self.max-self.min)

        return X_test

    def forward_propogation(self,parameters,X_train):
        """
        1.向前传播，返回经过sigmoid或ReLU激活函数处理后的z(即a)
        2.记得先把参数从字典中取出
        3.把a1,a2存入cache字典中，返回
        """
        w1=parameters['w1']
        b1=parameters['b1']
        w2=parameters['w2']
        b2=parameters['b2']

        z1=np.dot(w1.T,X_train)+b1
        if self.if_need_ReLU==False:
            a1=self.sigmoid(z1)
        elif self.if_need_ReLU==True:
            a1=self.ReLU(z1)
        z2=np.dot(w2.T,a1)+b2
        a2=self.sigmoid(z2)

        cache={'a1':a1,
               'a2':a2}

        return cache

    def forward_propogation_predict(self,X_test):
        """
        1.predict方法专用的向前传播方法
        2.所用到的参数都是已经训练好的self.parameters_have_trained
        """
        w1=self.parameters_have_trained['w1']
        b1=self.parameters_have_trained['b1']
        w2=self.parameters_have_trained['w2']
        b2=self.parameters_have_trained['b2']

        z1=np.dot(w1.T,X_test)+b1
        if self.if_need_ReLU==False:
            a1=self.sigmoid(z1)
        elif self.if_need_ReLU==True:
            a1=self.ReLU(z1)
        z2=np.dot(w2.T,a1)+b2
        a2=self.sigmoid(z2)

        cache={'a1':a1,
               'a2':a2}

        return cache

    def back_propogation(self,cache,parameters,X_train,Y_train,lambd):
        """
        1.记得从字典中取出参数
        2.反向传播，返回损失和一个记录导数的gradients(梯度)字典
        3.dw和db都代表w和b对损失(cost)的偏导数
        """
        a1=cache['a1']
        a2=cache['a2']
        w1=parameters['w1']
        w2=parameters['w2']

        if self.if_L2_regularize==False:
            cost=(-1/self.m_train)*np.sum(Y_train*np.log(a2)+(1-Y_train)*np.log(1-a2))
            
            dz2=a2-Y_train
            dw2=(1/self.m_train)*np.dot(a1,dz2.T)
            db2=(1/self.m_train)*np.sum(dz2,axis=1,keepdims=True)


            if self.if_need_ReLU==False:
                dz1=np.multiply(np.dot(w2,dz2),a1-np.power(a1,2))
            elif self.if_need_ReLU==True:
                da=np.where(np.array(a1)<=0,0,1)
                dz1=np.multiply(np.dot(w2,dz2),da)
            dw1=(1/self.m_train)*np.dot(X_train,dz1.T)
            db1=(1/self.m_train)*np.sum(dz1,axis=1,keepdims=True)

        elif self.if_L2_regularize==True:
            cost=(-1/self.m_train)*np.sum(Y_train*np.log(a2)+(1-Y_train)*np.log(1-a2))+(lambd/2*self.m_train)*(np.sum(np.square(w1))+np.sum(np.square(w2)))

            dz2=a2-Y_train
            dw2=(1/self.m_train)*np.dot(a1,dz2.T)+(lambd/self.m_train)*w2
            db2=(1/self.m_train)*np.sum(dz2,axis=1,keepdims=True)


            if self.if_need_ReLU==False:
                dz1=np.multiply(np.dot(w2,dz2),a1-np.power(a1,2))
            elif self.if_need_ReLU==True:
                da=np.where(np.array(a1)<=0,0,1)
                dz1=np.multiply(np.dot(w2,dz2),da)
            dw1=(1/self.m_train)*np.dot(X_train,dz1.T)+(lambd/self.m_train)*w1
            db1=(1/self.m_train)*np.sum(dz1,axis=1,keepdims=True)

        gradients={'dw2':dw2,
                   'db2':db2,
                   'dw1':dw1,
                   'db1':db1}

        return cost,gradients

    def update_parameters(self,parameters,gradients,alpha):
        """
        1.梯度下降法更新函数返回更新过的w1,w2,b1,b2
        """
        dw2=gradients['dw2']
        db2=gradients['db2']
        dw1=gradients['dw1']
        db1=gradients['db1']

        w2=parameters['w2']
        b2=parameters['b2']
        w1=parameters['w1']
        b1=parameters['b1']

        w2=w2-alpha*dw2
        b2=b2-alpha*db2
        w1=w1-alpha*dw1
        b1=b1-alpha*db1

        #参数重新存一遍
        parameters={'w1':w1,
                    'b1':b1,
                    'w2':w2,
                    'b2':b2}

        return parameters

    def train(self,X_train,Y_train,times=2000,alpha=0.01,lambd=0.01,transform='s',regularize=False,need_ReLU=True):
        """
        1.train方法集成了实现逻辑回归所需要的必要方法
          实际运用时直接调用train方法
          p.s. train方法将训练过的参数字典,损失列表，训练次数列表设为了类属性,后两者是为了方便画表格
          p.s.一旦调用train方法，if_have_trained变为True
        2.m_train为训练集样本个数
        3.n_input为输入层神经元个数，也是训练集特征个数
        """
        self.if_have_trained=True

        m_train=X_train.shape[1]
        self.m_train=m_train

        n_input=X_train.shape[0]
        self.n_input=n_input

        #判断接下来的训练和预测中是否使用标准化|归一化
        if transform=='s':
            self.if_standardize=True
        elif transform=='n':
            self.if_normalize=True
        elif transform==False:
            pass
        
        #判断是否使用正则化
        if regularize=='L2':
            self.if_L2_regularize=True
        else:
            pass
        #判断是否在隐藏层使用ReLU激活函数
        if need_ReLU==True:
            self.if_need_ReLU=True
        
        #记录损失和训练次数
        costs=[]
        iter_times=[]
        #初始化
        parameters=self.initialize()
        #标准化|归一化|或不做处理
        if self.if_standardize==True:
            X_train=self.standardize(X_train)
        elif self.if_normalize==True:
            X_train=self.normalize(X_train)

        for i in range(times):
            cache=self.forward_propogation(parameters,X_train)
            cost,gradients=self.back_propogation(cache,parameters,X_train,Y_train,lambd)
            parameters=self.update_parameters(parameters,gradients,alpha)

            if (i+1)%100==0:
                cost=round(cost,4)
                print(f'循环{i+1}次，损失为{cost}')
                costs.append(cost)
                iter_times.append(i+1)

        self.parameters_have_trained=parameters
        self.costs=costs
        self.iter_times=iter_times

    def predict(self,X_test,Y_test,threshold=0.5):
        """
        1.进行模型评估，输出正确率
          p.s.要在训练后才能调用predict方法
        2.将X_test,a和true_false_matrix设为类属性，以便展示预测错误的图片
        3.将最初的X_test设置为类属性，是为了不管是否进行标准化|归一化(改变原始数据的操作)
          show_false_example方法都能正确画出预测错误的样本的图像
        """
        if self.if_have_trained==True:
            self.previous_X_test=X_test
            if self.if_standardize==True:
                X_test=self.standardize_predict(X_test)
            elif self.if_normalize==True:
                X_test=self.normalize_predict(X_test)

            cache=self.forward_propogation_predict(X_test)
            a2=cache['a2']
            Y_predict=np.where(np.array(a2)>=threshold,1,0)

            true_false_matrix=(Y_predict==Y_test)
            true_percentage=(np.sum(true_false_matrix==True)/true_false_matrix.size)*100
            true_percentage=round(true_percentage,3)
            print('正确率为: ',true_percentage,"%",sep='')

            self.true_percentage_list.append(true_percentage)

            self.a2=a2
            self.true_false_matrix=true_false_matrix

    def plot(self):
        """
        绘制'训练次数和损失的关系'图像
        """
        #选择字体，使之可以显示中文
        plt.rcParams['font.sans-serif']=['SimHei']

        if self.if_have_trained==True:
            plt.figure(figsize=(10,6))
            plt.title('训练次数和损失的关系')
            plt.plot(self.iter_times,self.costs)
            plt.yticks(self.costs[::2])

            plt.xlabel('训练次数')
            plt.ylabel('损失')

            plt.show()

    def split_train_test(self,path,percentage=0.2):
        """
        1.载入并划分数据集(csv类型)
        2.percentage表示测试集占比
        """
        data=np.loadtxt(path,delimiter=',')
        np.random.shuffle(data)
        data=data.T
        target=data[0].reshape(1,-1)
        feature=data[1:]

        #print(target,target.shape,sep='\n')
        #print(feature,feature.shape,sep='\n')

        split_threshold=int(percentage*data.shape[1])

        X_train=feature[:,split_threshold:]
        X_test=feature[:,:split_threshold]
        Y_train=target[0,split_threshold:]
        Y_test=target[0,:split_threshold]

        return X_train,X_test,Y_train,Y_test

    def show_false_example(self):
        """
        1.展示预测错误的图像和对应的a的值
        """
        position=np.where(self.true_false_matrix==False)[1]
        for i in position:
            print('a2的值为:',self.a2[0,i])
            image_vector=self.previous_X_test.T[i]
            image=Image.new('L',(64,64),255)

            x,y=0,0
            for ii in image_vector:
                if ii==1:
                    ii=255
                image.putpixel((x,y),int(ii))
                if x==63:
                    x=0
                    y+=1
                else:
                    x+=1
        
            image.show()

    def show_average_true_percentage(self,path,percentage=0.2,try_times=10,times=2000,alpha=0.01,lambd=0.01,transform='s',regularize=False,need_ReLU=True):
        """
        1.用于多次训练模型，求取预测的平均准确率和方差
        """
        for i in range(try_times):
            print(f'第{i+1}次训练')
            X_train,X_test,Y_train,Y_test=self.split_train_test(path,percentage)
            self.train(X_train,Y_train,times,alpha,lambd,transform,regularize,need_ReLU)
            self.predict(X_test,Y_test,threshold=0.5)

        print(f'{try_times}次训练后，预测的平均准确率为:',np.mean(self.true_percentage_list),sep='')
        print(f'{try_times}次训练后，预测的准确率的方差为:',round(np.std(self.true_percentage_list)**2,3))

    def save_model(self,name,model):
        """
        1.导入joblib，用于保存训练所得的模型
        """
        import joblib
        if self.if_have_trained==True:
            joblib.dump(filename=name,value=model)
        else:
            print('尚未进行训练')

    def load_model(self,name):
        """
        1.用于加载保存好的模型
        """
        import joblib
        self.if_have_trained=True

        snn=joblib.load(filename=name)

        return snn