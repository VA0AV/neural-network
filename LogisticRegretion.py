import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class LogisticRegretion:
    def __init__(self):
        """
        默认输出层的神经元为1个
        默认初始的if_have_trained(是否已经进行训练)
        """
        self.n_output=1
        self.if_have_trained=False

    def sigmoid(self,z):
        """
        sigmoid激活函数
        """
        a=1/(1+np.exp(-z))

        return a

    def initialize(self):
        """
        权重和偏置初始化
        """
        w=np.random.randn(self.dim,1)*0.01
        b=0

        return w,b

    def forward_propogation(self,w,b,X_train):
        """
        向前传播，返回经过sigmoid激活函数处理后的z(即a)
        """
        z=np.dot(w.T,X_train)+b
        a=self.sigmoid(z)

        return a

    def forward_propogation_predict(self,X_test):
        """
        predict方法专用的向前传播方法
        所用到的w,b都是已经训练好的self.w_have_trained，self.b_have_trained
        """
        z=np.dot(self.w_have_trained.T,X_test)+self.b_have_trained
        a=self.sigmoid(z)

        return a

    def back_propogation(self,a,X_train,Y_train,m_train):
        """
        反向传播，返回损失和一个记录导数的gradients(梯度)字典
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
        梯度下降法更新函数
        返回更新过的w,b
        """
        dw=gradients['dw']
        db=gradients['db']

        w=w-alpha*dw
        b=b-alpha*db

        return w,b

    def train(self,X_train,Y_train,times=2000,alpha=0.01):
        """
        train方法集成了实现逻辑回归所需要的必要方法
        实际运用时直接调用train方法
        p.s. train方法将训练过的w,b,损失列表，训练次数列表设为了类属性
        """
        m_train=X_train.shape[1]

        dim=X_train.shape[0]
        self.dim=dim

        costs=[]
        iter_times=[]

        w,b=self.initialize()
        for i in range(times):
            a=self.forward_propogation(w,b,X_train)
            cost,gradients=self.back_propogation(a,X_train,Y_train,m_train)
            w,b=self.update_parameters(w,b,gradients,alpha)

            if (i+1)%100==0:
                cost=round(cost,3)
                print(f'循环{i+1}次，损失为{cost}')
                costs.append(cost)
                iter_times.append(i+1)

        self.w_have_trained=w
        self.b_have_trained=b
        self.costs=costs
        self.iter_times=iter_times

        self.if_have_trained=True

    def predict(self,X_test,Y_test,threshold=0.5):
        """
        进行模型评估，输出正确率
        要在训练后才能调用predict方法
        将X_test,a和true_false_matrix设为类属性，以便展示预测错误的图片
        """
        if self.if_have_trained==True:
            a=self.forward_propogation_predict(X_test)
            Y_predict=np.where(np.array(a)>=threshold,1,0)

            true_false_matrix=(Y_predict==Y_test)
            true_percentage=(np.sum(true_false_matrix==True)/true_false_matrix.size)*100
            true_percentage=round(true_percentage,3)
            print('正确率为: ',true_percentage,"%",sep='')

            self.X_test=X_test
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
        载入并划分数据集(csv类型)
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
        position=np.where(self.true_false_matrix==False)[1]
        for i in position:
            print('a的值为:',self.a[0,i])
            image_vector=self.X_test.T[i]
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