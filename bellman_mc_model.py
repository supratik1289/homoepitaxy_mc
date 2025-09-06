#code implementing the homoepitaxy mc model from Bellman Konrad's PhD thesis

#necessary imports 
import numpy as np
from numpy import sinh,cosh,exp
import matplotlib.pyplot as plt
import time

#simulation parameters
l_0 = 55
k_m = 0.5
k_p = 1
C_s = 0.01
lambda_s = 100
lambda_p = k_p/C_s
lambda_m = k_m/C_s
Nterrace = 100
Natoms = 100000

#lateral flux calculation function obtained using 4.8 and 4.2
def J_s(x,l_0,lambda_p,lambda_m,lambda_s):
    a = 0.5*(l_0-2*x)/lambda_s
    b = 0.5*(l_0+2*x)/lambda_s
    c = lambda_s*lambda_p
    d = lambda_s*lambda_m
    e = lambda_p*lambda_m
    f = l_0/lambda_s
    g = lambda_s**2
    return -2*(c*lambda_m*cosh(a)-c*lambda_m*cosh(b)+d*lambda_s*sinh(a)-c*lambda_s*sinh(b))/(exp(f)*(d+e+g+c)-exp(-f)*(d-e+g-c))

#calculate the initial probabilities by using the fluxes as per 4.10
def calculate_prob(probs,flux_vals1,flux_vals2):
    tot_sum = np.sum(flux_vals1)+np.sum(flux_vals2)
    for i in range(Nterrace):
        probs[i] = (flux_vals1[i]+flux_vals2[(i+1)%Nterrace])/tot_sum
    return probs,tot_sum

#update function to update the terraces after one simulation step
def update(terrace_widths,flux_vals1,flux_vals2,tot_sum,probs):
    #based on the probabilities calculated using 4.10 choose the terrace to deposit adatom
    s = np.random.choice(100,p=probs)

    #change the width of the selected terrace and the terrace above it
    terrace_widths[(s+1)%Nterrace]+=l_0*Nterrace/100000
    terrace_widths[s]-=l_0*Nterrace/100000

    #calculate new flux values using 4.8 and 4.2
    flux_vals1[s] = J_s(terrace_widths[s],l_0,lambda_p,lambda_m,lambda_s)
    flux_vals2[s] = -J_s(-terrace_widths[s],l_0,lambda_p,lambda_m,lambda_s)
    flux_vals1[(s+1)%Nterrace] = J_s(terrace_widths[(s+1)%Nterrace],l_0,lambda_p,lambda_m,lambda_s)
    flux_vals2[(s+1)%Nterrace] = -J_s(-terrace_widths[(s+1)%Nterrace],l_0,lambda_p,lambda_m,lambda_s)

    #update the total sum and probabilities
    probs,tot_sum = calculate_prob(probs,flux_vals1,flux_vals2)
    
    #return every updated quantity
    return terrace_widths,flux_vals1,flux_vals2,probs,tot_sum

#run the simulation for number of adatoms given
def simulate(terrace_widths,probs,flux_vals1,flux_vals2,tot_sum,Natoms):
    for i in range(int(Natoms)):
        #print(i)
        terrace_widths,flux_vals1,flux_vals2,probs,tot_sum=update(terrace_widths,flux_vals1,flux_vals2,tot_sum,probs)
    return terrace_widths

#initialize
terrace_widths = np.array([l_0]*100,dtype=float)
flux_vals1 = J_s(terrace_widths,l_0,lambda_p,lambda_m,lambda_s)
flux_vals2 = -J_s(-terrace_widths,1_0,lambda_p,lambda_m,lambda_s)
probs = np.zeros(Nterrace)
probs,tot_sum = calculate_prob(probs,flux_vals1,flux_vals2)  

#simulate
start = time.time()
terrace_widths = simulate(terrace_widths,probs,flux_vals1,flux_vals2,tot_sum,Natoms)     
end = time.time()
print("Time taken to simulate:",end-start)

#plot the data
lengths = terrace_widths

x = np.concatenate(([0], np.cumsum(lengths)))
y = np.arange(len(lengths) + 1)

plt.figure(figsize=(10, 6))
plt.step(x, y, where='post', linewidth=2)
plt.ylim([0,25])
plt.xlim([0,1375])
plt.xlabel("Height (ML)")
plt.ylabel("Lateral Position")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

