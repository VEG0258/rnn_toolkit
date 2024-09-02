import numpy as np
import matplotlib.pyplot as plt

#Helping method1
def sample(a, temperature=1.0):
    if temperature == 1.0:
        print("default temperature = 1")
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

#Helping method2
def trans_mat(x, numstates, tau=1):
    if tau == 1:
        print("default tau = 1")
    space_size=numstates
    xm = x[0:-tau]
    xp = x[tau:]
    JP=np.histogram2d(xm,xp,bins=np.arange(space_size+1)+1)[0]
    sums = np.sum(JP, axis=1, keepdims=True)  # Sum over each row to normalize
    sums[sums == 0] = 1  # Avoid division by zero; replace 0 sums with 1
    T = JP / sums  # Normalize to get transition probabilities
    T[np.isnan(T)] = 1 / numstates  # Set NaN values if any (should be none with the fix) to 1/numstates
   
    return T.T

#Helping method3
def errory_plot(x,y,erry,style,style2,xlabel=None,ylabel=None,limits=None,label=None,title=None,scalex = 'linear',scaley='linear'): 
    plt.plot(x,y,style,label = label)
    plt.fill_between(x,y-erry,y+erry,color=style2)
    if limits:
        plt.axis(limits)
    plt.xscale(scalex)
    plt.yscale(scaley)
    plt.tick_params(labelsize=13)
    if title:
        plt.title(title,fontsize=20)
    if ylabel:
        plt.ylabel(ylabel,fontsize=20)
    if xlabel:
        plt.xlabel(xlabel,fontsize=20)

#Helping method4
def evals(T,indices=range(1,6,1)):
    ls = np.absolute(np.linalg.eigvals(T))
    ls=np.sort(ls)[::-1]
    return ls[indices]

#Helping method5
def eval_stat(data,tau,num=6,ndata=59):
    x = data
    T = trans_mat(x,tau=tau)
    ls = evals(T)
    sem=np.std(ls,axis=0)/np.sqrt(ndata)
    return ls,sem


def persisenceplot(dt, s, numstates, batch_size):

    mean_pers_rnn=np.zeros(numstates)
    c_rnn=np.zeros_like(mean_pers_rnn)
    mean_pers_data=np.zeros(numstates)
    c_data=np.zeros_like(mean_pers_data)

    #this loops breaks the data into behavioral bouts and takes the average bout length for each behavior
    for i in range(batch_size):
        rnn_bouts=np.split(s[i,:],np.where(np.diff(s[i,:])!=0)[0]+1)
        data_bouts=np.split(dt[i,:],np.where(np.diff(dt[i,:])!=0)[0]+1)
        
        for i,x in enumerate(rnn_bouts):
            mean_pers_rnn[int(x[0])]+=len(x)
            c_rnn[int(x[0])]+=1
            
        for i,x in enumerate(data_bouts):
            mean_pers_data[int(x[0])]+=len(x)
            c_data[int(x[0])]+=1


    mean_pers_data=mean_pers_data[c_data!=0]/c_data[c_data!=0]
    mean_pers_rnn[c_rnn==0]=1
    c_rnn[c_rnn==0]=1
    mean_pers_rnn=mean_pers_rnn[c_rnn!=0]/c_rnn[c_rnn!=0]

    #plotted histogram of the persistence times
    h1_dt,b_dt=np.histogram(mean_pers_data,bins=numstates,density=True)
    h1_rnn,b_rnn=np.histogram(mean_pers_rnn,bins=numstates,density=True)
    plt.plot((b_dt[1:]+b_dt[:-1])/2,h1_dt,label='data')
    plt.plot((b_rnn[1:]+b_rnn[:-1])/2,h1_rnn,label='rnn')

    plt.legend(loc="upper right")
    plt.xlabel('Persistence Times');
    plt.savefig(f'LSTM/plots/PersistenceTimes')
    plt.clf()
    print("persistence plot done")

    #plots mean persistence time of the rnn vs. the data
    srtidx=np.argsort(mean_pers_rnn)
    mean_pers_data=mean_pers_data[srtidx]
    mean_pers_rnn=mean_pers_rnn[srtidx]
    plt.plot(mean_pers_data,mean_pers_rnn,'o')
    plt.xlabel('$<\\tau_{Data}>$')
    plt.ylabel('$<\\tau_{RNN}>$')
    plt.title('Persistence Times')
    plt.savefig(f'LSTM/plots/meanpersistencetime')
    plt.clf()
    print("mean persistence plot done")

def transitionplot(dt, s, numstates):
    rnn_bouts=np.split(s.flatten(),np.where(np.diff(s.flatten())!=0)[0]+1)
    data_bouts=np.split(dt.flatten(),np.where(np.diff(dt.flatten())!=0)[0]+1)

    rnn_transitions=np.zeros(np.asarray(rnn_bouts, dtype=object).shape[0])
    data_transitions=np.zeros(np.asarray(data_bouts, dtype=object).shape[0])

    for i in range(rnn_transitions.shape[0]):
        rnn_transitions[i]=rnn_bouts[i][0]
        
    for i in range(data_transitions.shape[0]):
        data_transitions[i]=data_bouts[i][0]

    T_data=trans_mat(data_transitions, numstates)
    T_rnn=trans_mat(rnn_transitions, numstates)

    plt.plot(T_data.flatten(),T_rnn.flatten(),'o')

    #plots of the 1-step transition matrix for the data (left) and rnn (right) sequences
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15, 15))
    a1=ax2.imshow(T_data,cmap='cubehelix_r',vmin=0,vmax=.55)
    a2=ax1.imshow(T_rnn,cmap='cubehelix_r',vmin=0,vmax=.55)
    plt.savefig(f'LSTM/plots/transitionmatrix')
    print("transition matrix done")

def eigenvalueplot(dt, s, batch_size, numFlies):
    fig=plt.figure(figsize=(15, 5))

    rnn_transitions=[]
    data_transitions=[]
    
    #this for loop converts the time series to transitions
    for i in range(batch_size):
        rnn_bouts=np.split(s[i,:],np.where(np.diff(s[i,:])!=0)[0]+1)
        data_bouts=np.split(dt[i,:],np.where(np.diff(dt[i,:])!=0)[0]+1)
        
        rnn_transitions.append(np.zeros(np.asarray(rnn_bouts, dtype=object).shape[0]))
        for j,x in enumerate(rnn_bouts):
            rnn_transitions[i][j]=rnn_bouts[j][0]

        data_transitions.append(np.zeros(np.asarray(data_bouts, dtype=object).shape[0]))
        for j,x in enumerate(data_bouts):
            data_transitions[i][j]=data_bouts[j][0]

    #this computes the eigenvalues as a function of taus in the future for the rnn sequences
    taus=np.unique((10**np.arange(0,3.53,0.03)).astype(int))
    means=[]
    sems=[]
    for tau in taus:
        tmpm=np.zeros((s.shape[0],5))
        tmps=np.zeros_like(tmpm)
        for i in range(s.shape[0]):
            mean,sem=eval_stat(rnn_transitions[i][:],tau,num=5,ndata=s.shape[0])
            tmpm[i,:]=mean
            tmps[i,:]=sem          
        means.append(np.mean(tmpm,0))
        sems.append(np.mean(tmps,0))

    #plotting the eignevalues
    npmeans = np.array(means)
    npsems = np.array(sems)
    co = ['b','r','m','g','c']
    co2 = [[32/255,178/255,170/255],[233/255,150/255,122/255],[1,182/255,193/255],[50/255,205/255,50/255],[224/255,1,1]]
    labels = [r'$\mu=2$',r'$\mu=3$',r'$\mu=4$',r'$\mu=5$',r'$\mu=6$']
    fig.add_subplot(1,2,1)
    for i in range(npmeans.shape[1]):
        errory_plot(taus,npmeans[:,i],npsems[:,i],co[i]+'--',co2[i],r'$\tau$',r'$|\lambda|$',[min(taus), max(taus),0,1],label=None,title='RNN time Scales',scalex='log')
        
    #this computes the eigenvalues as a function of taus in the future for the data sequences    
    means=[]
    sems=[]
    for tau in taus:
        tmpm=np.zeros((batch_size,5))
        tmps=np.zeros_like(tmpm)
        for i in range(batch_size):
            mean,sem=eval_stat(data_transitions[i][:],tau,num=5,ndata=numFlies)
            tmpm[i,:]=mean
            tmps[i,:]=sem          
        means.append(np.mean(tmpm,0))
        sems.append(np.mean(tmps,0))

    #plotting the eignevalues
    npmeans = np.array(means)
    npsems = np.array(sems)
    co = ['b','r','m','g','c']
    co2 = [[32/255,178/255,170/255],[233/255,150/255,122/255],[1,182/255,193/255],[50/255,205/255,50/255],[224/255,1,1]]
    labels = [r'$\mu=2$',r'$\mu=3$',r'$\mu=4$',r'$\mu=5$',r'$\mu=6$']
    fig.add_subplot(1,2,2)
    for i in range(npmeans.shape[1]):
        errory_plot(taus,npmeans[:,i],npsems[:,i],co[i],co2[i],r'$\tau$',r'$|\lambda|$',[min(taus), max(taus),0,1],label=None,title='Data time Scales',scalex='log')

    fig.savefig(f'LSTM/plots/eigenvalues.png')
    print("eigenvalue plot done")
