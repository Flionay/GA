#!/usr/bin/env python
# coding: utf-8

# ## 遗传算法详解---附加一个计算例子的代码实现

# 比如我们要解下面这个方程的最大值 $$y=x*sin(10\pi x)+2$$然后画出函数图像如下：
# [借鉴博客](http://czrzchao.com/simpleGaByPython)
# [大白话GA](https://blog.csdn.net/hiudawn/article/details/80144221)

# In[62]:


import matplotlib.pyplot as plt
import numpy as np
import random
import math
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[63]:


x = np.arange(-1,2,0.01)
y = x*np.sin(10*np.pi*x)+2
plt.plot(x,y)
plt.show


# 1. 首先我们生成一堆个体，构成一个种群，分布在函数的不同区域。

# In[64]:


def geneEncoding(pop_size, chrom_length):
    '''
    输入 ： 种群大小  染色体长度
    输出 ： 染色体种群
    '''
    pop = [[]]
    for i in range(pop_size):
        temp = []
        for j in range(chrom_length):
            temp.append(random.randint(0, 1))
        pop.append(temp)

    return pop[1:]


# In[65]:


gene = geneEncoding(5,3)
gene


# In[66]:


def main_fun(x):
    y = x*np.sin(10*np.multiply(np.pi,x))+2
    return y


# In[67]:


main_fun([1,0.0555555,0.05])


# In[68]:


def plot_currnt_individual(X, Y):
    X1 = np.arange(-1,2,0.01)
    Y1 = main_fun(x)
    plt.plot(X1, Y1)
    plt.scatter(X, Y, c='r', s=5)
    plt.show()


# 这里的二进制计算得到的十进制，就是最简单的进制转换了，比如一个二进制为10101，转化为十进制就是1×24+0×23+1×22+0×21+1×20=21
# 然后将这个数字再次映射到x的允许区间内。比如另 P=21 则$$x=X_{min}+\frac{p*(X_{max} - X_{min})}{2^{染色体长度}-1})$$这样就将染色体解码为十进制的X区间内的数。

# In[69]:


def decodechrom(pop, chromosome_length, upper_limit,lower_limit):
    '''
    解码染色体  就是将二进制转换为10进制
    输入：种群 染色体长度 x的最大值(右端点)
    输出：十进制的染色体代表的数
    '''
    X = []
#     if len(pop) != 1:

    for ele in pop:

        temp = 0
        # 二进制变成实数，种群中的每个个体对应一个数字
        for i, coff in enumerate(ele):
            # 就是把二进制转化为十进制的
            temp += coff * (2 ** i)

        # 这个是把前面得到的那个十进制的数，再次缩放为另一个实数
        # 注意这个实数范围更广泛，可以是小数了，而前面二进制解码后只能是十进制的数
        # 参考https://blog.csdn.net/robert_chen1988/article/details/79159244
        X.append(lower_limit + (temp * (upper_limit-lower_limit)) / (2 ** chromosome_length - 1))
#     else:
#         temp = 0
#         for i, coff in enumerate(pop):
#                 # 就是把二进制转化为十进制的
#                 temp += coff * (2 ** i)

#             # 这个是把前面得到的那个十进制的数，再次缩放为另一个实数
#             # 注意这个实数范围更广泛，可以是小数了，而前面二进制解码后只能是十进制的数
#             # 参考https://blog.csdn.net/robert_chen1988/article/details/79159244
#         X.append(temp * upper_limit / (2 ** chromosome_length - 1))

    return X


# In[70]:


len([[1,0,1]])


# In[71]:

#
# test = decodechrom([[1,0,0]],3,2,-1)
# test


# 2. 计算每个染色体所所对应的目标函数值

# In[72]:


def calobjValue(pop, chrom_length, max_value,min_value):
    '''
    输入：种群  染色体长度  最大的X
    输出： 目标函数值
    '''
    temp1 = []
    obj_value = []
    temp1 = decodechrom(pop, chrom_length,max_value,min_value)   # 解码染色体
    obj_value = main_fun(temp1)
    return obj_value


# In[73]:


calobjValue(gene,3,2,-1)


# 3. 需要一个淘汰的操作，目的是为了去除掉一些坏的基因，比如说这个函数值全是大于0的，如果产生负值，就需要淘汰掉这个基因。

# In[74]:


# 淘汰（去除负值）
def calfitValue(obj_value):
    '''
    输入： 目标函数值
    输出： 淘汰后的目标函数值
    '''
    fit_value = []
    # 去掉小于0的值，更改c_min会改变淘汰的下限
    # 比如设成10可以加快收敛
    # 但是如果设置过大，有可能影响了全局最优的搜索
    c_min = 0
    for value in obj_value:
        if value > c_min:
            temp = value
        else:
            temp = 0.
        fit_value.append(temp)
    # fit_value保存的是活下来的值
    return fit_value


# #### 选择

# In[75]:


# 选择

def cum_sum(fit_value):
    # 输入[1, 2, 3, 4, 5]，返回[1,3,6,10,15]，matlab的一个函数
    # 这个地方遇坑，局部变量如果赋值给引用变量，在函数周期结束后，引用变量也将失去这个值
    temp = fit_value[:]
    for i in range(len(temp)):
        fit_value[i] = (sum(temp[:i + 1]))
        
    return temp

# 找出最优解和最优解的基因编码
def find_best(pop, fit_value):
    # 用来存最优基因编码
    best_individual = []
    # 先假设第一个基因的适应度最好
    best_fit = fit_value[0]
    for i in range(1, len(pop)):
        if (fit_value[i] > best_fit):
            best_fit = fit_value[i]
            best_individual = pop[i]
    # best_fit是最大的适应值
    # best_individual是对应的基因序列
    return [best_individual], best_fit

def selection1(pop, fit_value):
    # https://blog.csdn.net/pymqq/article/details/51375522

    p_fit_value = []
    # 适应度总和
    total_fit = sum(fit_value)
    # 归一化，使概率总和为1
    assert(fit_value!=0)
    for i in range(len(fit_value)):
        p_fit_value.append(fit_value[i] / total_fit)
    # 概率求和排序

    # https://www.cnblogs.com/LoganChen/p/7509702.html
    p_fit_value = cum_sum(p_fit_value) # 计算累计概率
    pop_len = len(pop)
    # 类似搞一个转盘吧下面这个的意思
    # ms = sorted([random.random() for i in range(pop_len)])
    # fitin = 0
    # newin = 0
    newpop = []
   
    # 转轮盘选择法 
    for i in range(pop_len):
        ms=random.random()
        j=0
        if p_fit_value[i]<0.5:

            j+=1
        newpop.append(pop[j])
    pop = newpop
    
    # while newin < pop_len and fitin < pop_len:
    #     # 如果这个概率大于随机出来的那个概率，就选这个
    #     if (ms[newin] < p_fit_value[fitin]):
    #         newpop.append(pop[fitin])
    #         newin = newin + 1
    #     else:
    #         fitin = fitin + 1
    # # 这里注意一下，因为random.random()不会大于1，所以保证这里的newpop规格会和以前的一样
    # # 而且这个pop里面会有不少重复的个体，保证种群数量一样
    #
    # # 之前是看另一个人的程序，感觉他这里有点bug，要适当修改
    # pop = newpop[:]
    
    return pop

def selection(pop, fit_value):
    s=sum(fit_value)
    temp=[k*1.0/s for k in fit_value]
    temp2=[]

    s2=0
    for k in temp:
        s2=s2+k
        temp2.append(s2)

    temp3=[]
    for _ in range(len(pop)):
        r=random.random()
        for i in range(len(temp2)):
            if r<=temp2[i]:
                temp3.append(i)
                break

    temp4=[]
    temp5=[]
    for i in temp3:
        temp4.append(pop[i])
        temp5.append(fit_value[i])
    pop[:]=temp4
    fit_value[:]=temp5
    return pop



# 交叉
def crossover(pop, pc):
    '''
    输入：种群 杂交概率
    输出：杂交后的种群
    
    
    '''
    pop_len = len(pop)
    for i in range(pop_len - 1):
        if(random.random() < pc):
            cpoint = random.randint(0,len(pop[0]))
            temp1 = []
            temp2 = []
            temp1.extend(pop[i][0:cpoint])
            temp1.extend(pop[i+1][cpoint:len(pop[i])])
            temp2.extend(pop[i+1][0:cpoint])
            temp2.extend(pop[i][cpoint:len(pop[i])])
            pop[i] = temp1
            pop[i+1] = temp2
    return pop
    


# In[78]:


# 基因突变
def mutation(pop, pm):
    
    '''
    输入：种群 变异概率
    输出：基因突变后的种群
    
    '''
    px = len(pop)
    py = len(pop[0])
    # 每条染色体随便选一个杂交
    for i in range(px):
        if (random.random() < pm):
            mpoint = random.randint(0, py - 1)
            if (pop[i][mpoint] == 1):
                pop[i][mpoint] = 0
            else:
                pop[i][mpoint] = 1
    return pop


# In[79]:


def plot_iter_curve(iter, results):
    X = [i for i in range(iter)]
    Y = [results[i][1] for i in range(iter)]
    plt.figure(figsize=(8,6),dpi=100)
    plt.plot(X, Y)
    plt.show()


# In[86]:


def main():
    pop_size = 800  # 种群数量
    upper_limit = 2  # 基因中允许出现的最大值
    lower_limit = -1
    chromosome_length = 10  # 染色体长度
    iters = 1600
    pc = 0.6  # 杂交概率
    pm = 0.01  # 变异概率
    results = []  # 存储每一代的最优解，N个二元组

    pop = geneEncoding(pop_size, chromosome_length)
    best_X = []
    best_Y = []
    for i in range(iters):
        obj_value = calobjValue(pop, chromosome_length, upper_limit, lower_limit)  # 个体评价，有负值
        fit_value = calfitValue(obj_value)  # 个体适应度，不好的归0，可以理解为去掉上面的负值
        best_individual, best_fit = find_best(pop, fit_value)  # 第一个是最优基因序列, 第二个是对应的最佳个体适度

        # 下面这句就是存放每次迭代的最优x值是最佳y值
        results.append([decodechrom(best_individual, chromosome_length, upper_limit, lower_limit), best_fit])
        # print(results)
        # 查看一下种群分布

        if i % 80 == 0:
            plot_currnt_individual(decodechrom(pop, chromosome_length, upper_limit, lower_limit), obj_value)

        pop = selection(pop, fit_value)  # 选择
        pop = crossover(pop, pc)  # 染色体交叉（最优个体之间进行0、1互换）
        pop = mutation(pop, pm)  # 染色体变异（其实就是随机进行0、1取反）
        # 最优解的变化
        # if i % 20 == 0:
        #     best_X.append(results[-1][0])
        #     best_Y.append(results[-1][1])
            # print("x = %f, y = %f" % (results[-1][0], results[-1][1]))
            # 看种群点的选择
    plt.scatter(results[-1][0], results[-1][1], s=10, c='r')
    x = np.arange(-1, 2, 0.01)
    y = x * np.sin(10 * np.pi * x) + 2
    plt.plot(x, y)
    plt.show
    # 看迭代曲线
    plot_iter_curve(iters, results)

if __name__=='__main__':

    main()





