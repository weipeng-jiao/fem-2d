#author:tony jiao
#finished 4/2019 at HNUT


from matplotlib.tri.triinterpolate import LinearTriInterpolator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.tri as tri
import sys
from solver import KMat, gauss, Strain, Stress

##前处理部分

elements= [ [3 , 1 , 0] , [ 2 , 1 , 3 ] ] 
nodes = np.array( [ [ 0 , 0 ] , [ 1 , 0 ] , [ 1 , 1 ] , [ 0 , 1 ] ])
F =  [ 0 , 0 , -10000 , 0 ,-10000 , 0 ,0 , 0 ] 
U = [ [ 1 , 0 ] , [ 2 , 0 ] , [ 7 , 0 ] , [ 8 , 0 ] ] 

x=nodes[:,0]
y=nodes[:,1]
N=len(x)

#非结构网格节点生成网格编号
triObj = tri.Triangulation(x, y,elements)	# 生成给定平面点组凸包的 Delauney 三角剖分.
print(triObj.triangles) #输出每个单元的顶点编号
print(triObj.edges) #输出每个边的顶点编号
elements=np.array(elements)
#网格结构可视化
fig=plt.figure('网格结构图') #生成画图窗口
sub=fig.add_subplot(111) #调用子图
#plt.plot(x,y,color='r',label='edge')
#plt.plot([x[0],x[-1]],[y[0],y[-1]],color='r',label='edge')

sub.scatter(x,y,marker='o',color='b',s=50) #画节点
for i in range(N): #节点标号
    sub.text(x[i]+0.02,y[i]+0.02,'{:d}'.format(i),fontsize=12,color='b')
for i in range(len(elements)): #网格标号
    centx=(x[elements[i][0]]+x[elements[i][1]]+x[elements[i][2]])/3
    centy=(y[elements[i][0]]+y[elements[i][1]]+y[elements[i][2]])/3
    sub.text(centx,centy,'{:d}'.format(i),fontsize=12,color='r')
sub.triplot(triObj,color='k')#画三角形网格



#求解部分
E0=200000000000#杨氏模量
v=0.25#泊松比
t=1#平板厚度
kt=KMat(elements, nodes, E0,v,t)#总刚矩阵
u=gauss(kt, U, F)#位移列阵
B=Strain(elements,nodes,u)#应变矩阵
S=Stress(B, E0, v)#应力矩阵





#填充颜色
triangles=np.array(triObj.triangles)#列表数组转化为np数组
ntri=len(triangles)
for i in range(ntri):
    vertices = np.zeros([3,2])
    for j in range(3):
        vertices[j,0]=x[triangles[i,j]]
        vertices[j,1]=y[triangles[i,j]]
    ploy=Polygon(vertices,color=plt.cm.plasma(1))
    sub.add_patch(ploy)



sub.set_xlim([0,1])
sub.set_ylim([0,1])
plt.show()
