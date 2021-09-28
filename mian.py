#####################################################################################
# author:tony jiao
# name:fem-2d
# finished 4/2019 at HNUT
# Copyright:MIT
######################################################################################


from matplotlib.tri.triinterpolate import LinearTriInterpolator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Polygon
import matplotlib.tri as tri
import pandas as pd
from solver import KMat, gauss, Strain, Stress

############################################ 前处理部分#################################################

#节点位移
nodes = np.array( [ [ 0 , 0 ] , [ 1 , 0 ] , [ 1 , 1 ] , [ 0 , 1 ] ])
#单元编号
elements= [ [3 , 1 , 0] , [ 2 , 1 , 3 ] ] 

############################自动划分矩形##########################
n=10
x=np.linspace(0,1,n)
y=np.linspace(0,1,n)
xx,yy=np.meshgrid(x,y)
# reshape meshgrid and stack them to get the right shape for
# delaunay ([x1,y1], [x2,y2], .....
nodes= np.dstack(np.meshgrid(x, y)).reshape(-1, 2)


#外力列阵
load =  [[99*2+1,-10000]] 
#位移列阵
constraint= [ [ 0 , 0 ] , [ 0 , 1 ] , [ 90 , 0 ] , [ 90, 1 ] ] 

F=np.zeros(2*len(nodes))
for i in range(len(load)):
    temp=load[i][0]
    F[temp]=load[i][1]

###############################################前处理可视化#################################################
#网格节点生成网格编号
x=nodes[:,0]
y=nodes[:,1]
N=len(x)
triObj = tri.Triangulation(x,y)	# 生成给定平面点组结构性网格的 Delauney 三角剖分.
elements=triObj.triangles #输出每个单元的顶点编号
elements=np.array(elements)

#网格结构可视化
fig1=plt.figure('mesh') #生成画图窗口
sub1=fig1.add_subplot(111,title='mesh') #调用子图
sub1.scatter(x,y,marker='o',color='b',s=40) #画节点
for i in range(N): #节点标号
    sub1.text(x[i]+0.002,y[i]+0.002,'{:d}'.format(i),fontsize=12,color='b')
for i in range(len(elements)): #网格标号
    centx=(x[elements[i][0]]+x[elements[i][1]]+x[elements[i][2]])/3
    centy=(y[elements[i][0]]+y[elements[i][1]]+y[elements[i][2]])/3
    sub1.text(centx,centy,'{:d}'.format(i),fontsize=12,color='k')
sub1.triplot(triObj,color='k')#画三角形网格

plt.get_current_fig_manager().window.state('zoomed')
plt.savefig("mesh")
plt.show()

####################施加载荷和约束###########################
#画载荷
fig2=plt.figure('load') #生成画图窗口
sub1=fig2.add_subplot(111,title='load') #调用子图
sub1.scatter(x,y,marker='o',color='b',s=40) #画节点
for i in range(N): #节点标号
    sub1.text(x[i],y[i],'{:d}'.format(i),fontsize=12,color='b')
for i in range(len(elements)): #网格标号
    centx=(x[elements[i][0]]+x[elements[i][1]]+x[elements[i][2]])/3
    centy=(y[elements[i][0]]+y[elements[i][1]]+y[elements[i][2]])/3
    sub1.text(centx,centy,'{:d}'.format(i),fontsize=12,color='k')
sub1.triplot(triObj,color='k')#画三角形网格
n=len(F)
for i in range(n):
    if F[i]!=0:
        f=str(F[i])
        x0=nodes[i//2][0]
        y0=nodes[i//2][1]
        bias=i%2
        dl=0.03*np.sign(F[i])
        if bias==0:
            plt.arrow(x0, y0, dl, 0, width=0.005,fc="r",ec='r')
            plt.text(x0,y0,f,fontsize=10,color='r')
        else:
            plt.arrow(x0, y0, 0, dl, width=0.01,fc="r",ec='r')
            plt.text(x0,y0,f,fontsize=10,color='r')
#画约束
n=len(constraint)
for i in range(n):
    temp=constraint[i][0]
    x0=nodes[temp][0]
    y0=nodes[temp][1]
    if  constraint[i][1]==0:
        sub1.scatter(x0-0.02,y0,marker='>',color='k',s=40) #画节点
    else:
        sub1.scatter(x0,y0+0.035,marker='v',color='k',s=40) #画节点


xlim=[min(x)-0.5*(max(x)-min(x)),max(x)+0.5*(max(x)-min(x))]
ylim=[min(y)-0.5*(max(y)-min(y)),max(y)+0.5*(max(y)-min(y))]
sub1.set_xlim(xlim)
sub1.set_ylim(ylim)

plt.get_current_fig_manager().window.state('zoomed')
plt.savefig("load")
plt.show()



##################################################求解部分#################################################
E0=200000000000#杨氏模量
v=0.25#泊松比
t=1#平板厚度
kt=KMat(elements, nodes, E0,v,t)#总刚矩阵
u=gauss(kt, constraint, F)#位移列阵
B=Strain(elements,nodes,u)#应变矩阵
S=Stress(B, E0, v)#应力矩阵




#############################################后处理可视化visualization########################################
########################应变云图###########
fig2=plt.figure('strain') #生成画图窗口
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)#调整子图间距
sub1=fig2.add_subplot(311,title='ε-x') #调用子图
#添加colorsbar
vmin=min(B.flatten())
vmax=max(B.flatten())
cmap = plt.get_cmap('jet')#将颜色映射到0-1区间
norm = colors.Normalize(vmin=vmin, vmax=vmax)#归一化
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm) #配置数值和颜色映射关系
plt.colorbar(sm) #生成调色板

sub2=fig2.add_subplot(312,title='ε-y') #调用子图
#添加colorsbar
vmin=min(B.flatten())
vmax=max(B.flatten())
cmap = plt.get_cmap('jet')#将颜色映射到0-1区间
norm = colors.Normalize(vmin=vmin, vmax=vmax)#归一化
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm) #配置数值和颜色映射关系
plt.colorbar(sm) #生成调色板

sub3=fig2.add_subplot(313,title='ε-xy') #调用子图
#添加colorsbar
vmin=min(B.flatten())
vmax=max(B.flatten())
cmap = plt.get_cmap('jet')#将颜色映射到0-1区间
norm = colors.Normalize(vmin=vmin, vmax=vmax)#归一化
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm) #配置数值和颜色映射关系
plt.colorbar(sm) #生成调色板

triangles=np.array(triObj.triangles)#列表数组转化为np数组
ntri=len(triangles)#获取单元编号
for i in range(ntri):
    vertices = np.zeros([3,2])
    for j in range(3):
        vertices[j,0]=x[triangles[i,j]]
        vertices[j,1]=y[triangles[i,j]]
    epsilonx=B[i][0]
    epsilony=B[i][1]
    epsilonxy=B[i][2]
    ploy1=Polygon(vertices,color=plt.cm.jet(plt.Normalize(vmin=vmin, vmax=vmax)(epsilonx)))
    ploy2=Polygon(vertices,color=plt.cm.jet(plt.Normalize(vmin=vmin, vmax=vmax)(epsilony)))
    ploy3=Polygon(vertices,color=plt.cm.jet(plt.Normalize(vmin=vmin, vmax=vmax)(epsilonxy)))
    sub1.add_patch(ploy1)
    sub2.add_patch(ploy2)
    sub3.add_patch(ploy3)
sub1.triplot(triObj,color='k')#画三角形网格
sub1.set_xlim([min(x)-0.5*(max(x)-min(x)),max(x)+0.5*(max(x)-min(x))])
sub1.set_ylim([min(y)-0.5*(max(y)-min(y)),max(y)+0.5*(max(y)-min(y))])
sub2.triplot(triObj,color='k')#画三角形网格
sub2.set_xlim([min(x)-0.5*(max(x)-min(x)),max(x)+0.5*(max(x)-min(x))])
sub2.set_ylim([min(y)-0.5*(max(y)-min(y)),max(y)+0.5*(max(y)-min(y))])
sub3.triplot(triObj,color='k')#画三角形网格
sub3.set_xlim([min(x)-0.5*(max(x)-min(x)),max(x)+0.5*(max(x)-min(x))])
sub3.set_ylim([min(y)-0.5*(max(y)-min(y)),max(y)+0.5*(max(y)-min(y))])


plt.get_current_fig_manager().window.state('zoomed')
plt.savefig("strain")
plt.show()





#####################################应力云图###########################
fig3=plt.figure('stress') #生成画图窗口
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)

sub1=fig3.add_subplot(311,title='σ-x') #调用子图
#添加colorsbar
vmin=min(S.flatten())
vmax=max(S.flatten())
cmap = plt.get_cmap('jet')#将颜色映射到0-1区间
norm = colors.Normalize(vmin=vmin, vmax=vmax)#归一化
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm) #配置数值和颜色映射关系
plt.colorbar(sm) #生成调色板

sub2=fig3.add_subplot(312,title='σ-y') #调用子图
#添加colorsbar
vmin=min(S.flatten())
vmax=max(S.flatten())
cmap = plt.get_cmap('jet')#将颜色映射到0-1区间
norm = colors.Normalize(vmin=vmin, vmax=vmax)#归一化
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm) #配置数值和颜色映射关系
plt.colorbar(sm) #生成调色板

sub3=fig3.add_subplot(313,title='σ-xy') #调用子图
#添加colorsbar
vmin=min(S.flatten())
vmax=max(S.flatten())
cmap = plt.get_cmap('jet')#将颜色映射到0-1区间
norm = colors.Normalize(vmin=vmin, vmax=vmax)#归一化
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm) #配置数值和颜色映射关系
plt.colorbar(sm) #生成调色板

triangles=np.array(triObj.triangles)#列表数组转化为np数组
ntri=len(triangles)#获取单元编号
for i in range(ntri):
    vertices = np.zeros([3,2])
    for j in range(3):
        vertices[j,0]=x[triangles[i,j]]
        vertices[j,1]=y[triangles[i,j]]
    sigmax=S[i][0]
    sigmay=S[i][1]
    sigmaxy=S[i][2]
    ploy1=Polygon(vertices,color=plt.cm.jet(plt.Normalize(vmin=vmin, vmax=vmax)(sigmax)))
    ploy2=Polygon(vertices,color=plt.cm.jet(plt.Normalize(vmin=vmin, vmax=vmax)(sigmay)))
    ploy3=Polygon(vertices,color=plt.cm.jet(plt.Normalize(vmin=vmin, vmax=vmax)(sigmaxy)))
    sub1.add_patch(ploy1)
    sub2.add_patch(ploy2)
    sub3.add_patch(ploy3)
sub1.triplot(triObj,color='k')#画三角形网格
sub1.set_xlim([min(x)-0.5*(max(x)-min(x)),max(x)+0.5*(max(x)-min(x))])
sub1.set_ylim([min(y)-0.5*(max(y)-min(y)),max(y)+0.5*(max(y)-min(y))])
sub2.triplot(triObj,color='k')#画三角形网格
sub2.set_xlim([min(x)-0.5*(max(x)-min(x)),max(x)+0.5*(max(x)-min(x))])
sub2.set_ylim([min(y)-0.5*(max(y)-min(y)),max(y)+0.5*(max(y)-min(y))])
sub3.triplot(triObj,color='k')#画三角形网格
sub3.set_xlim([min(x)-0.5*(max(x)-min(x)),max(x)+0.5*(max(x)-min(x))])
sub3.set_ylim([min(y)-0.5*(max(y)-min(y)),max(y)+0.5*(max(y)-min(y))])



plt.get_current_fig_manager().window.state('zoomed')
plt.savefig("stress")
plt.show()

################################################数据保存#######################################
##节点标号
list1=elements
name1=['单元节点i','单元节点j','单元节点m']
Elements=pd.DataFrame(columns=name1,data=list1)
Elements.to_csv('Elements.csv')



##节点坐标
list2=nodes
name2=['x','y']
Nodes=pd.DataFrame(columns=name2,data=list2)
Nodes.to_csv('Nodes.csv')


##单元应变
list3=B
name3=['epsilonx','epsilony','epsilonxy']
Epsilon=pd.DataFrame(columns=name3,data=list3)
Epsilon.to_csv('Epsilon.csv')


##单元应变
list4=S
name4=['sigmax','sigmay','sigmaxy']
Sigma=pd.DataFrame(columns=name4,data=list4)
Sigma.to_csv('Sigma.csv')