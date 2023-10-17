import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from scipy.integrate import simps

#Defining the space region
n=4001			#discretization
fin_pos=20
in_pos=0
x_step=1000
x,delt_x=np.linspace(-fin_pos,fin_pos,n,retstep=True)



#Defining the time region
final_time=5
delt_t=0.01


#defining the potential
v=0*x[1:-1]    #for free particle

#defining the parameters
alpha=(1j*delt_t)/(2*(delt_x)**2)
beta=1+2*alpha+(1j*delt_t*v)/2
gamma=1-2*alpha-(1j*delt_t*v)/2


#defining the matrices
u_1=np.eye(n-2,k=-1)*(-alpha)+np.diag(beta)-np.eye(n-2,k=1)*(alpha)
u_2=np.eye(n-2,k=-1)*(alpha)+np.diag(gamma)+np.eye(n-2,k=1)*(alpha)
			
#defining the evolution matrix
u=np.dot(np.linalg.inv(u_1),u_2)


#Initial form of the wavefuntion
k=2
gwp=np.exp(-x**2)*np.exp(1j*k*x)
norm=simps(np.abs(gwp)**2,x)
ngwp=np.exp(-x**2)*np.exp(1j*k*x)/np.sqrt(norm)


line,=plt.plot(x,np.abs(ngwp)**2)
plt.plot(x[1:-1],v)
plt.ylim(0,1)
plt.xlim(-fin_pos,fin_pos)
plt.grid('True')

t=0
time_step=0
nrm=[]
tplt=[]
#Evolution
while t<final_time:
	nrm.append(round(simps((np.abs(ngwp))**2,x),3))
	tplt.append(t)
	ngwp[1:-1]=np.dot(u,ngwp[1:-1])
	t+=delt_t
	line.set_ydata((np.abs(ngwp))**2)
	plt.pause(0.00001)
plt.title('Norm vs time')	
plt.xlabel('Time')
plt.ylabel('Norm')
plt.plot(tplt,nrm)
plt.show()





