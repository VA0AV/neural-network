import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class LogisticRegretion:
    def __init__(self):
        """
        1.默认输出层的神经元为1个
        2.默认初始的if_have_trained(是否已经进行训练)为False
        3.默认初始的if_standardize(是否进行标准化)为False
        4.默认初始的if_normalize(是否进行归一化)为False
        """
        self.n_output=1
        self.if_have_trained=False
        self.if_standardize=False
        self.if_normalize=False

    def sigmoid(self,z):
        """
        1.sigmoid激活函数,将z值映射到(0,1)
        """
        a=1/(1+np.exp(-z))

        return a

    def initialize(self):
        """
        1.权重(w)和偏置(b)初始化
          单纯的逻辑回归相当于一个感知机，没有隐藏层，且输出层只有一个神经元
          偏置设为一个实数就行
        """
        w=np.random.randn(self.dim,1)*0.01
        b=0

        return w,b

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

    def forward_propogation(self,w,b,X_train):
        """
        1.向前传播，返回经过sigmoid激活函数处理后的z(即a)
        """
        z=np.dot(w.T,X_train)+b
        a=self.sigmoid(z)

        return a

    def forward_propogation_predict(self,X_test):
        """
        1.predict方法专用的向前传播方法
        2.所用到的w,b都是已经训练好的self.w_have_trained，self.b_have_trained
        """
        z=np.dot(self.w_have_trained.T,X_test)+self.b_have_trained
        a=self.sigmoid(z)

        return a

    def back_propogation(self,a,X_train,Y_train,m_train):
        """
        1.反向传播，返回损失和一个记录导数的gradients(梯度)字典
        2.dw和db都代表w和b对损失(cost)的偏导数
        """
        cost=(-1/m_train)*np.sum(Y_train*np.log(a)+(1-Y_train)*np.log(1-a))
        #print((-1/m_train))
        #print(np.log(a))
        #print((1-Y_train))
        #print(np.log(1-a))
        dw=(1/m_train)*np.dot(X_train,(a-Y_train).T)
        db=(1/m_train)*np.sum(a-Y_train)

        gradients={
            'dw':dw,
            'db':db
        }

        return cost,gradients

    def update_parameters(self,w,b,gradients,alpha):
        """
        1.梯度下降法更新函数返回更新过的w,b
        """
        dw=gradients['dw']
        db=gradients['db']

        w=w-alpha*dw
        b=b-alpha*db

        return w,b

    def train(self,X_train,Y_train,times=2000,alpha=0.01,transform='s'):
        """
        1.train方法集成了实现逻辑回归所需要的必要方法
          实际运用时直接调用train方法
          p.s. train方法将训练过的w,b,损失列表，训练次数列表设为了类属性,后两者是为了方便画表格
          p.s.一旦调用train方法，if_have_trained变为True
        2.m_train为训练集样本个数
        3.dim为训练集特征个数
        """
        self.if_have_trained=True
        #判断接下来的训练和预测中是否使用标准化|归一化
        if transform=='s':
            self.if_standardize=True
        elif transform=='n':
            self.if_normalize=True
        elif transform==False:
            pass

        m_train=X_train.shape[1]

        dim=X_train.shape[0]
        self.dim=dim
        #记录损失和训练次数
        costs=[]
        iter_times=[]
        #初始化
        w,b=self.initialize()
        #标准化|归一化|或不做处理
        if self.if_standardize==True:
            X_train=self.standardize(X_train)
        elif self.if_normalize==True:
            X_train=self.normalize(X_train)

        for i in range(times):
            a=self.forward_propogation(w,b,X_train)
            cost,gradients=self.back_propogation(a,X_train,Y_train,m_train)
            w,b=self.update_parameters(w,b,gradients,alpha)

            if (i+1)%100==0:
                cost=round(cost,4)
                print(f'循环{i+1}次，损失为{cost}')
                costs.append(cost)
                iter_times.append(i+1)

        self.w_have_trained=w
        self.b_have_trained=b
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

            a=self.forward_propogation_predict(X_test)
            Y_predict=np.where(np.array(a)>=threshold,1,0)

            true_false_matrix=(Y_predict==Y_test)
            true_percentage=(np.sum(true_false_matrix==True)/true_false_matrix.size)*100
            true_percentage=round(true_percentage,3)
            print('正确率为: ',true_percentage,"%",sep='')

            self.a=a
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
            print('a的值为:',self.a[0,i])
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
