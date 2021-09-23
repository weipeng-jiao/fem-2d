#author:tony jiao
#finished


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.tri as tri
import sys

x = [0,0,1,2,2,3,3]
y = [0,2,3,2,1,1,0]



N=len(x)

#节点生成网格编号
triObj = tri.Triangulation(x, y)	# 生成给定平面点组凸包的 Delauney 三角剖分.
print(triObj.triangles) #输出每个单元的顶点编号
print(triObj.edges) #输出每个边的顶点编号
t=np.array(triObj.triangles)

#网格结构可视化
fig=plt.figure('网格结构图') #生成画图窗口
sub=fig.add_subplot(111) #调用子图
#plt.plot(x,y,color='r',label='edge')
#plt.plot([x[0],x[-1]],[y[0],y[-1]],color='r',label='edge')

sub.scatter(x,y,marker='o',color='b',s=50) #画节点
for i in range(N): #节点标号
    sub.text(x[i]+0.02,y[i]+0.02,'{:d}'.format(i),fontsize=12,color='b')
for i in range(len(t)): #网格标号
    centx=(x[t[i,0]]+x[t[i,1]]+x[t[i,2]])/3
    centy=(y[t[i,0]]+y[t[i,1]]+y[t[i,2]])/3
    sub.text(centx,centy,'{:d}'.format(i),fontsize=12,color='r')
sub.triplot(triObj,color='k')#画三角形网格
plt.show()#显示



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
