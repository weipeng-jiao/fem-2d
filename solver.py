#生成总刚矩阵
import numpy as np
def KMat(elements, nodes, E0, v, t):#单元数组 节点数组 弹体模量 泊松比 平面厚度
    n=2*len(nodes)
    Kt=np.zeros((n,n))
    i=0
    while i< n:
        #读取单元节点数信息
        a=elements[i][0]; b=elements[i][1]; c=elements[i][2];
        #读取节点坐标信息
        xi=nodes[a][0]; xj=nodes[b][0]; xm=nodes[c][0];
        yi=nodes[a][1]; yj=nodes[b][1]; ym=nodes[c][1];
        #获得A，b，c
        bi=yj-ym; bj=ym-yi; bm=yi-yj; 
        ci=-(xj-xm); cj=-(xm-xi); cm=-(xi-xj);
        A=(bi*cj-bj*ci)/2 #单元面积
        item=E0*t/(4*(1-v*v)*A) #矩阵前系数
        #生成单刚矩阵
        Ke = np.zeros((6 ,6))
        Ke[0][0]=(bi*bi+((1-v)/2)*ci*ci)*item; Ke[0][1]=(v*bi*ci+((1-v)/2)*ci*bi)*item; Ke[0][2]=(bi*bj+((1-v)/2)*ci*cj)*item; Ke[0][3]=(v*bi*cj+((1-v)/2)*ci*bj)*item; Ke[0][4]=(bi*bm+((1-v)/2)*ci*cm)*item; Ke[0][5]=(v*bi*cm+((1-v)/2)*ci*bm)*item;
        Ke[1][0]=(v*bi*ci+((1-v)/2)*ci*bi)*item; Ke[1][1]=(ci*ci+((1-v)/2)*bi*bi)*item; Ke[1][2]=(v*bj*ci+((1-v)/2)*cj*bi)*item; Ke[1][3]=(ci*cj+((1-v)/2)*bi*bj)*item; Ke[1][4]=(v*bm*ci+((1-v)/2)*cm*bi)*item; Ke[1][5]=(ci*cm+((1-v)/2)*bi*bm)*item;
        Ke[2][0]=(bj*bi+((1-v)/2)*cj*ci)*item; Ke[2][1]=(v*bj*ci+((1-v)/2)*cj*bi)*item; Ke[2][2]=(bj*bj+((1-v)/2)*cj*cj)*item; Ke[2][3]=(v*bj*cj+((1-v)/2)*cj*bj)*item; Ke[2][4]=(bj*bm+((1-v)/2)*cj*cm)*item; Ke[2][5]=(v*bj*cm+((1-v)/2)*cj*bm)*item;
        Ke[3][0]=(v*bi*cj+((1-v)/2)*ci*bj)*item; Ke[3][1]=(cj*ci+((1-v)/2)*bj*bi)*item; Ke[3][2]=(v*bj*cj+((1-v)/2)*cj*bj)*item; Ke[3][3]=(cj*cj+((1-v)/2)*bj*bj)*item; Ke[3][4]=(v*bm*cj+((1-v)/2)*cm*bj)*item; Ke[3][5]=(cj*cm+((1-v)/2)*bj*bm)*item;
        Ke[4][0]=(bm*bi+((1-v)/2)*cm*ci)*item; Ke[4][1]=(v*bm*ci+((1-v)/2)*cm*bi)*item; Ke[4][2]=(bm*bj+((1-v)/2)*cm*cj)*item; Ke[4][3]=(v*bm*cj+((1-v)/2)*cm*bj)*item; Ke[4][4]=(bm*bm+((1-v)/2)*cm*cm)*item; Ke[4][5]=(v*bm*cm+((1-v)/2)*cm*bm)*item;
        Ke[5][0]=(v*bi*cm+((1-v)/2)*ci*bm)*item; Ke[5][1]=(cm*ci+((1-v)/2)*bm*bi)*item; Ke[5][2]=(v*bj*cm+((1-v)/2)*cj*bm)*item; Ke[5][3]=(cm*cj+((1-v)/2)*bm*bj)*item; Ke[5][4]=(v*bm*cm+((1-v)/2)*cm*bm)*item; Ke[5][5]=(cm*cm+((1-v)/2)*bm*bm)*item;
        #总刚矩阵单元位置
        a1= 2*a; a2= 2*a+1; 
        b1= 2*b; b2= 2*b+1;
        c1= 2*c; c2= 2*c+1;
        #总刚矩阵组装
        Kt[a1][a1]= Kt[a1][a1]+Ke[0][0]; Kt[a1][a2]= Kt[a1][a2]+Ke[0][1]; Kt[a1][b1]= Kt[a1][b1]+Ke[0][2]; Kt[a1][b2]= Kt[a1][b2]+Ke[0][3]; Kt[a1][c1]= Kt[a1][c1]+Ke[0][4]; Kt[a1][c2]= Kt[a1][c2]+Ke[0][5];
        Kt[a2][a1]= Kt[a2][a1]+Ke[1][0]; Kt[a2][a2]= Kt[a2][a2]+Ke[1][1]; Kt[a2][b1]= Kt[a2][b1]+Ke[1][2]; Kt[a2][b2]= Kt[a2][b2]+Ke[1][3]; Kt[a2][c1]= Kt[a2][c1]+Ke[1][4]; Kt[a2][c2]= Kt[a2][c2]+Ke[1][5];
        Kt[b1][a1]= Kt[b1][a1]+Ke[2][0]; Kt[b1][a2]= Kt[b1][a2]+Ke[2][1]; Kt[b1][b1]= Kt[b1][b1]+Ke[2][2]; Kt[b1][b2]= Kt[b1][b2]+Ke[2][3]; Kt[b1][c1]= Kt[b1][c1]+Ke[2][4]; Kt[b1][c2]= Kt[b1][c2]+Ke[2][5];
        Kt[b2][a1]= Kt[b2][a1]+Ke[3][0]; Kt[b2][a2]= Kt[b2][a2]+Ke[3][1]; Kt[b2][b1]= Kt[b2][b1]+Ke[3][2]; Kt[b2][b2]= Kt[b2][b2]+Ke[3][3]; Kt[b2][c1]= Kt[b2][c1]+Ke[3][4]; Kt[b2][c2]= Kt[b2][c2]+Ke[3][5];
        Kt[c1][a1]= Kt[c1][a1]+Ke[4][0]; Kt[c1][a2]= Kt[c1][a2]+Ke[4][1]; Kt[c1][b1]= Kt[c1][b1]+Ke[4][2]; Kt[c1][b2]= Kt[c1][b2]+Ke[4][3]; Kt[c1][c1]= Kt[c1][c1]+Ke[4][4]; Kt[c1][c2]= Kt[c1][c2]+Ke[4][5];
        Kt[c2][a1]= Kt[c2][a1]+Ke[5][0]; Kt[c2][a2]= Kt[c2][a2]+Ke[5][1]; Kt[c2][b1]= Kt[c2][b1]+Ke[5][2]; Kt[c2][b2]= Kt[c2][b2]+Ke[5][3]; Kt[c2][c1]= Kt[c2][c1]+Ke[5][4]; Kt[c2][c2]= Kt[c2][c2]+Ke[5][5];
        i=i+1
    return Kt





def gauss(Kt, U, F): #总刚度矩阵 位移边界 外力列阵
    #置1法
    i=0
    while i<len(U):
        a=2*U[i][0]+U[i][1]
        j=0
        while j<len(Kt):
            Kt[a][j]=0
            Kt[j][a]=0
            j=j+1
        Kt[a][a]=1
        i=i+1
    #生成增广矩阵
    n=2*len(Kt)
    K=np.zeros((n,n+1))
    i=0
    while i<n:
        K[i][n]=F[i]
        j=0
        while j<n:
            K[i][j]=Kt[i][j]
            j=j+1
        i=i+1
    #高斯消元法
    i=0
    while i<n:
        k=i+1
        while k<n:
            temp =K[k][i]
            j=i
            while j<=n:
                temp =K[k][i]
                j=j+1
            k=k+1
        i=i+1
    X=np.zeros(n)
    i=n-1
    while i>=0:
        sum=0
        j=i+1
        while j<n:
            sum=sum+K[i][j]*X[j]
            j=j+1
        X[i]=(K[i][n]-sum)/K[i][i]
        i=i-1
    return X      

def Strain(elements,nodes,U):#获得应变信息
    n=len(elements)
    strain=np.zeros([n,3])
    
    for i in range(n):
        #读取单元节点数信息
        a=elements[i][0]; b=elements[i][1]; c=elements[i][2]
        #读取节点坐标信息
        xi=nodes[a][0]; xj=nodes[b][0]; xm=nodes[c][0]
        yi=nodes[a][1]; yj=nodes[b][1]; ym=nodes[c][1]
        #获得A，b，c
        bi=yj-ym; bj=ym-yi; bm=yi-yj
        ci=-(xj-xm); cj=-(xm-xi); cm=-(xi-xj)
        A=(bi*cj-bj*ci)/2 #单元面积
       
        B=np.zeros([3,6])
        B[0][0]=bi/(2*A); B[0][1]=0; B[0][2]=bj/(2*A); B[0][3]=0; B[0][4]=bm/(2*A);B[0][5]=0
        B[1][0]=0; B[1][1]=ci/(2*A); B[1][2]=0; B[1][3]=cj/(2*A); B[1][4]=0; B[1][5]=cm/(2*A)
        B[2][0]=ci/(2*A); B[2][1]=bi/(2*A);B[2][2]=cj/(2*A); B[2][3]=bj/(2*A); B[2][4]=cm/(2*A); B[2][5]=bm/(2*A)
        
        a1= 2*a-2; a2= 2*a-1
        b1= 2*b-2; b2= 2*b-1
        c1= 2*c-2; c2= 2*c-1
	    
        for j in range(3): 
	        strain[i][j]=B[j][0]*U[a1]+B[j][1]*U[a2]+B[j][2]*U[b1]+B[j][3]*U[b2]+B[j][4]*U[c1]+B[j][5]*U[c2]
	
        
	return strain
	
  	

def Stress(B,E0,v):
        item=E0/(1-v*v)
	    D=np.zeros([3,3])
	    #弹性矩阵各元素
	    D[0][0]=1*item;D[0][1]=v*item;D[0][2]=0
	    D[1][0]=v*item;D[1][1]=1*item;D[1][2]=0
	    D[2][0]=0;D[2][1]=0;D[2][2]=((1-v)/2)*item
	
        n1=len(B)
        n2=len(B[0])
        stress=np.zeros([n1,n2])
	 
        for i in range(n1):
            for j in range(n2):
		        stress[i][j]=D[j][0]*B[i][0]+D[j][1]*B[i][1]+D[j][2]*B[i][2]
		
	return stress