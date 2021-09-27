#author:tony jiao
#finished 4/2019 at HNUT


from matplotlib.tri.triinterpolate import LinearTriInterpolator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Polygon
import matplotlib.tri as tri
import sys
from solver import KMat, gauss, Strain, Stress

############################################ 前处理部分#################################################
#节点位移
nodes = np.array( [ [ 0 , 0 ] , [ 1 , 0 ] , [ 1 , 1 ] , [ 0 , 1 ] ])
#单元编号
elements= [ [3 , 1 , 0] , [ 2 , 1 , 3 ] ] 
#外力列阵
F =  [ 0 , 0 , -10000 , 0 ,-10000 , 0 ,0 , 0 ] 
#位移列阵
U = [ [ 1 , 0 ] , [ 2 , 0 ] , [ 7 , 0 ] , [ 8 , 0 ] ] 



###############################################前处理可视化#################################################
#非结构网格节点生成网格编号
x=nodes[:,0]
y=nodes[:,1]
N=len(x)
triObj = tri.Triangulation(x, y,elements)	# 生成给定平面点组结构性网格的 Delauney 三角剖分.
print(triObj.triangles) #输出每个单元的顶点编号
print(triObj.edges) #输出每个边的顶点编号
elements=np.array(elements)
#网格结构可视化
fig1=plt.figure('mesh') #生成画图窗口
sub1=fig1.add_subplot(111,title='mesh') #调用子图
#plt.plot(x,y,color='r',label='edge')
#plt.plot([x[0],x[-1]],[y[0],y[-1]],color='r',label='edge')

sub1.scatter(x,y,marker='o',color='b',s=50) #画节点
for i in range(N): #节点标号
    sub1.text(x[i]+0.02,y[i]+0.02,'{:d}'.format(i),fontsize=12,color='b')
for i in range(len(elements)): #网格标号
    centx=(x[elements[i][0]]+x[elements[i][1]]+x[elements[i][2]])/3
    centy=(y[elements[i][0]]+y[elements[i][1]]+y[elements[i][2]])/3
    sub1.text(centx,centy,'{:d}'.format(i),fontsize=12,color='r')
sub1.triplot(triObj,color='k')#画三角形网格

xlim=[min(x)-0.5*(max(x)-min(x)),max(x)+0.5*(max(x)-min(x))]
ylim=[min(y)-0.5*(max(y)-min(y)),max(y)+0.5*(max(y)-min(y))]
sub1.set_xlim(xlim)
sub1.set_ylim(ylim)
plt.savefig("mesh")
plt.show()

##################################################求解部分#################################################
E0=200000000000#杨氏模量
v=0.25#泊松比
t=1#平板厚度
kt=KMat(elements, nodes, E0,v,t)#总刚矩阵
u=gauss(kt, U, F)#位移列阵
B=Strain(elements,nodes,u)#应变矩阵
S=Stress(B, E0, v)#应力矩阵




#############################################后处理可视化visualization########################################
########################应变云图###########
fig2=plt.figure('strain') #生成画图窗口
sub1=fig2.add_subplot(311,title='ε-x') #调用子图
#添加colorsbar
N=3*len(B)
cmap = plt.get_cmap('jet', N)
norm = colors.Normalize(vmin=min(B.flatten()), vmax=max(B.flatten()))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm) 
plt.colorbar(sm) 

sub2=fig2.add_subplot(312,title='ε-y') #调用子图
#添加colorsbar
N=3*len(B)
cmap = plt.get_cmap('jet', N)
norm = colors.Normalize(vmin=min(B.flatten()), vmax=max(B.flatten()))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm) 
plt.colorbar(sm) 

sub3=fig2.add_subplot(313,title='ε-xy') #调用子图
#添加colorsbar
N=3*len(B)
cmap = plt.get_cmap('jet', N)
norm = colors.Normalize(vmin=min(B.flatten()), vmax=max(B.flatten()))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm) 
plt.colorbar(sm) 

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
    ploy1=Polygon(vertices,color=plt.cm.jet(epsilonx))
    ploy2=Polygon(vertices,color=plt.cm.jet(epsilony))
    ploy3=Polygon(vertices,color=plt.cm.jet(epsilonxy))
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
plt.savefig("strain")
plt.show()





#####################################应力云图###########################
fig3=plt.figure('stress') #生成画图窗口

sub1=fig3.add_subplot(311,title='σ-x') #调用子图
#添加colorsbar
N=3*len(S)
cmap = plt.get_cmap('jet', N)
norm = colors.Normalize(vmin=min(S.flatten()), vmax=max(S.flatten()))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm) 
plt.colorbar(sm) 

sub2=fig3.add_subplot(312,title='σ-y') #调用子图
N=3*len(S)
cmap = plt.get_cmap('jet', N)
norm = colors.Normalize(vmin=min(S.flatten()), vmax=max(S.flatten()))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm) 
plt.colorbar(sm) 

sub3=fig3.add_subplot(313,title='σ-xy') #调用子图
#添加colorsbar
N=3*len(S)
cmap = plt.get_cmap('jet', N)
norm = colors.Normalize(vmin=min(S.flatten()), vmax=max(S.flatten()))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm) 
plt.colorbar(sm) 

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
    ploy1=Polygon(vertices,color=plt.cm.jet(sigmax))
    ploy2=Polygon(vertices,color=plt.cm.jet(sigmay))
    ploy3=Polygon(vertices,color=plt.cm.jet(sigmaxy))
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

plt.savefig("stress")
plt.show()
