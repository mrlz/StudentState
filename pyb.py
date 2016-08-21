import pybayes as pb
import numpy as np
import random
import math
from scipy.stats import dirichlet
import copy
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
from sklearn.metrics import mean_squared_error as MSE


class Multinomial(pb.Pdf):
    def __init__(self, n, P, K):
        self.n = n
        self.P = P
        self.K = K
        self._set_rv(K, None)
        P1 = np.array(self.P)[np.newaxis]
        self.m = np.dot(self.n,self.P)
        self.var = self.n*(np.diag(self.P) - np.dot(P1.T,P1))

    def mean(self, cond = None):
        """Return (conditional) mean value of the pdf.

        :rtype: :class:`numpy.ndarray`
        """
        return self.m

    def variance(self, cond = None):
        """Return (conditional) variance (diagonal elements of covariance).

        :rtype: :class:`numpy.ndarray`
        """
        return self.var

    def eval_log(self, x, cond = None):
        """Return logarithm of (conditional) likelihood function in point x.

        :param x: point which to evaluate the function in
        :type x: :class:`numpy.ndarray`
        :rtype: double
        """
        a = self.eval(x,cond)
        if ( a == 0.0 ):
            #print("ye")
            a = 0.000001
        #print("a = ".format(a))
        return np.log(a)

    def eval(self, x, cond = None):
        return multinomial_pdf(x, self.P)

    def sample(self, cond = None):
        """Return one random (conditional) sample from this distribution

        :rtype: :class:`numpy.ndarray`"""
        return np.random.multinomial(self.n,self.P)

def multinomial_pdf(x, P):
    ans = 0
    if (np.sum(x) == 1):
        first = math.gamma(np.sum(x)+1)
        bleh = 1.0
        prob = 1.0
        for i in range(x.shape[0]):
            bleh = bleh*math.gamma(x[i] + 1)
            prob = prob*np.power(P[i],x[i])
        ans = first/bleh*prob
    return ans


class Dirichlet(pb.Pdf):
    def __init__(self, alfa, K):
        self.alfa = alfa
        self.K = K
        self.dir = dirichlet(self.alfa)
        self._set_rv(K, None)

    def mean(self, cond = None):
        """Return (conditional) mean value of the pdf.

        :rtype: :class:`numpy.ndarray`
        """
        return self.dir.mean()

    def variance(self, cond = None):
        """Return (conditional) variance (diagonal elements of covariance).

        :rtype: :class:`numpy.ndarray`
        """
        return self.dir.var()

    def eval_log(self, x, cond = None):
        """Return logarithm of (conditional) likelihood function in point x.

        :param x: point which to evaluate the function in
        :type x: :class:`numpy.ndarray`
        :rtype: double
        """
        return self.dir.logpdf(x)

    def sample(self, cond = None):
        """Return one random (conditional) sample from this distribution

        :rtype: :class:`numpy.ndarray`"""
        #return np.random.dirichlet(self.alfa)
        return self.dir.rvs()

class ConditionalMultinomial(pb.CPdf):

    def __init__(self, n, K, rv = None, cond_rv = None):
        self.n = n
        self.K = K
        self._set_rvs(K,rv,K,cond_rv)

    def mean(self, x):
        """Return (conditional) mean value of the pdf.

        :rtype: :class:`numpy.ndarray`
        """
        return Multinomial(self.n,x,self.K).mean()

    def variance(self, x):
        """Return (conditional) variance (diagonal elements of covariance).

        :rtype: :class:`numpy.ndarray`
        """
        return Multinomial(self.n, x, self.K).variance()

    def eval_log(self, yt, xt):
        """Return logarithm of (conditional) likelihood function in point x.

        :param x: point which to evaluate the function in
        :type x: :class:`numpy.ndarray`
        :rtype: double
        """
        #print("evaluando muestra {0}\nprobabilidad{1}".format(yt,xt))
        return Multinomial(self.n, xt, self.K).eval_log(yt)

    def sample(self, x):
        """Return one random (conditional) sample from this distribution

        :rtype: :class:`numpy.ndarray`"""
        return Multinomial(self.n, x, self.K).sample()

class ConditionalDirichlet(pb.CPdf):
    def __init__(self, a, K, rv = None, cond_rv = None):
        self.K = K
        self.dir = dirichlet
        self.a = a
        self._set_rvs(K,rv,K,cond_rv)

    # def computeAlfa(self, w1, x):
    #     temp = self.a*(w1+(1-w1)*x)
    #     temp = temp + 0.000001
    #     return temp

    def computeAlfa(self, prev_mean_x,w1,x): ##Estimate generated based on mean previous estimate and particular previous vector
        temp = self.a*w1*x + (1-w1)*self.a*prev_mean_x
        temp = temp + 0.00000001
        return temp

    def mean(self, x, w):
        """Return (conditional) mean value of the pdf.

        :rtype: :class:`numpy.ndarray`
        """

        return self.dir.mean(self.computeAlfa(w,x))

    def variance(self, x, w):
        """Return (conditional) variance (diagonal elements of covariance).

        :rtype: :class:`numpy.ndarray`
        """
        return self.dir.var(self.computeAlfa(w,x))

    def eval_log(self, xt, xtp, w):
        """Return logarithm of (conditional) likelihood function in point x.

        :param x: point which to evaluate the function in
        :type x: :class:`numpy.ndarray`
        :rtype: double
        """
        return self.dir.logpdf(xt,self.computeAlfa(w,xtp))

    def sample(self, xpm, x, w):
        """Return one random (conditional) sample from this distribution

        :rtype: :class:`numpy.ndarray`"""
        valor = self.computeAlfa(xpm, w,x)
        par = self.dir.rvs(valor)
        par1 = np.random.dirichlet(valor)
        return par1


class conditionalFilter(pb.ParticleFilter):
    def bayes(self, previous_mean_x, yt, w):
        for i in range(self.emp.particles.shape[0]):
            xtp = copy.deepcopy(self.emp.particles[i])
            we = copy.deepcopy(self.emp.weights[i])
            self.emp.particles[i] = copy.deepcopy(self.p_xt_xtp.sample(previous_mean_x,xtp, w)) ##Estimate generated based on mean previous estimate and particular previous vector
            #self.emp.particles[i] = copy.deepcopy(self.p_xt_xtp.sample(xtp, we)) #Estimate generated based on previous weight (Observational model) and previous x (student state estimate). These values are used by the commented ComputeAlfa function on ConditionalDIrichlet.
            particle = copy.deepcopy(self.emp.particles[i])
            #self.emp.weights[i] = copy.deepcopy(self.particular_weight(yt, np.sum(alfa),yt.shape[0],alfa))
            self.emp.weights[i] = copy.deepcopy(math.exp(self.p_yt_xt.eval_log(yt, particle)))
        # assure that weights are normalised
        self.emp.normalise_weights()
        # resample
        self.emp.resample()
        return True


class Subject_Question:
    def __init__(self, theta, K, name, a,p, w):
        #Particles, Belief(x) = Dirichlet(theta), Process Model = P(X(t), X(t-1), w, u) = Dir(Alfa), Observation model = P(y(t), x(t)) = Mult(1,x(t))
        self.filter = conditionalFilter(p, Dirichlet(theta,K), ConditionalDirichlet(a, K), ConditionalMultinomial(1,K))
        self.name = name
        self.estimation = 0
        self.w = w

    def updateEstimation(self,yt):
        self.filter.bayes(self.filter.posterior().mean(), yt, self.w) ###new
        #self.filter.bayes(yt)
        self.estimation = self.filter.posterior().mean()

    def showEstimation(self):
        return self.estimation

class Student:
    def __init__(self, states, particles, w, a):
        self.state = copy.deepcopy(states)
        self.Question = []
        self.Question.append(Subject_Question(np.array([1.0,1.0,1.0,1.0]), 4, "Easy Question"  ,  a, particles,w))
        self.Question.append(Subject_Question(np.array([1.0,1.0,1.0,1.0]), 4, "Harder Question",  a, particles,w))
        self.Question.append(Subject_Question(np.array([1.0,1.0,1.0,1.0]), 4, "Medium Question", a, particles,w))
        self.Question.append(Subject_Question(np.array([1.0,1.0,1.0,1.0]), 4, "Dynamic Question"  ,  a, particles,w))
        self.Question.append(Subject_Question(np.array([1.0,1.0,1.0,1.0]), 4, "Dynamic Question"  ,  a, particles,w))

    def answer(self,q):
            return np.random.multinomial(1,self.state[q])


    def retrieveEstimation(self,q):
        return self.selectQuestion(q).showEstimation()

    def getState(self,q):
        return self.state[q]


    def updateState(self,q,t):
        if (q == 3):
            # print("updating")
            # print(self.state[3])
            self.state[q,0] = 0.7*math.sin(3.14*t)
            self.state[q,1] = 0.3*(100-t)/100
            self.state[q,2] = 0.3*(100- (100 // (t + 1)))
            self.state[q,3] = (self.state[q,0] + self.state[q,1] + self.state[q,2]) % 1
            self.state[q] = self.state[q]/np.sum(self.state[q])
        elif(q == 4):
            self.state[q,0] = 0.7*math.sin(0.0314*t)
            self.state[q,1] = 0.3*(100-t)/100
            self.state[q,2] = 0.3*(100- (100 // (t + 1)))/100
            self.state[q,3] = (self.state[q,0] + self.state[q,1] + self.state[q,2]) % 1
            self.state[q] = self.state[q]/np.sum(self.state[q])

            #self.state[3] = self.state[3]/np.linalg.norm(self.state[3])

    def selectQuestion(self,q):
        return self.Question[q]

    def updateEstimation(self,ans,q):
        Question = self.selectQuestion(q)
        Question.updateEstimation(ans)

def plot_estimate(y,x,i,goal,procedure):
    fig1 = plt.figure(figsize = (10,10))
    plt.xlabel('t', fontsize = 18)
    plt.ylabel('estimate'+str(i), fontsize = 18)
    plt.plot(x, y, color = 'red', label = 'State '+str(i))
    plt.plot(x, goal, color = 'blue', label = "Goal")
    plt.legend(loc = 'upper right')
    plt.savefig(procedure+"_s"+str(i)+'.png')
    plt.close(fig1)

def plot_mean1(goal,x,wmean,i, procedure,fmean, label1, label2, title):
    fig1 = plt.figure(figsize = (10,10))
    plt.title(title)
    plt.xlabel('t', fontsize = 18)
    plt.ylabel('Probability', fontsize = 18)
    plt.plot(x, wmean, color = 'red', label = label1+"|State "+str(i))
    plt.plot(x, goal, color = 'blue', label = "Goal")
    plt.plot(x, fmean, color = 'black', label = label2+"|State "+str(i))
    plt.legend(loc = 'upper right')
    plt.savefig(procedure+"_s"+str(i)+'.png')
    plt.close(fig1)

def plot_mean(goal,x,wmean,i, procedure,fmean, label1, label2):
    plot_mean1(goal,x,wmean,i, procedure,fmean, label1, label2,"")

def split_plot(v1,v2,v3,v4,a1,a2,a3,a4,goal, x, w_mean, fmean, label1, label2, procedure, title):
    plot_mean1(goal[:,0], x, w_mean[:,0],1,procedure, fmean[:,0], label1, label2, title+"\n"+v1+"      "+a1)
    plot_mean1(goal[:,1], x, w_mean[:,1],2,procedure, fmean[:,1], label1, label2, title+"\n"+v2+"      "+a2)
    plot_mean1(goal[:,2], x, w_mean[:,2],3,procedure, fmean[:,2], label1, label2, title+"\n"+v3+"      "+a3)
    plot_mean1(goal[:,3], x, w_mean[:,3],4,procedure, fmean[:,3], label1, label2, title+"\n"+v4+"      "+a4)

def plot_estimates(y,x,goal,w_mean,f_mean, ml5, ml10,thread):
    fig1 = plt.figure(figsize = (10,10))
    plt.xlabel('t', fontsize = 18)
    plt.ylabel('estimate', fontsize = 18)

    plt.plot(x,y[:,0], color = 'red', label = 'State 1')
    plt.plot(x,y[:,1], color = 'blue', label = 'State 2')
    plt.plot(x,y[:,2], color = 'black', label = 'State 3')
    plt.plot(x,y[:,3], color = 'magenta', label = 'State 4')

    plt.plot(x,goal[:,0], color = 'red', label = 'Goal 1', linestyle = '--')
    plt.plot(x,goal[:,1], color = 'blue', label = 'Goal 2',linestyle = '--')
    plt.plot(x,goal[:,2], color = 'black', label = 'Goal 3',linestyle = '--')
    plt.plot(x,goal[:,3], color = 'magenta', label = 'Goal 4',linestyle = '--')

    plt.legend(loc = 'upper right')
    plt.savefig('summary'+str(thread)+'.png')
    plt.close(fig1)

    addendum = str(thread)
    plot_estimate(y[:,0],x,1,goal[:,0],"final"+addendum)
    plot_estimate(y[:,1],x,2,goal[:,1],"final"+addendum)
    plot_estimate(y[:,2],x,3,goal[:,2],"final"+addendum)
    plot_estimate(y[:,3],x,4,goal[:,3],"final"+addendum)

    plot_mean(goal[:,0], x, w_mean[:,0],1,"Means"+addendum, f_mean[:,0], "Weighted Mean", "Mean")
    plot_mean(goal[:,1], x, w_mean[:,1],2,"Means"+addendum, f_mean[:,1], "Weighted Mean", "Mean")
    plot_mean(goal[:,2], x, w_mean[:,2],3,"Means"+addendum, f_mean[:,2], "Weighted Mean", "Mean")
    plot_mean(goal[:,3], x, w_mean[:,3],4,"Means"+addendum, f_mean[:,3], "Weighted Mean", "Mean")

    plot_mean(goal[:,0], x, ml5[:,0],1,"MLM"+addendum, ml10[:,0], "ML5", "ML10")
    plot_mean(goal[:,1], x, ml5[:,1],2,"MLM"+addendum, ml10[:,1], "ML5", "ML10")
    plot_mean(goal[:,2], x, ml5[:,2],3,"MLM"+addendum, ml10[:,2], "ML5", "ML10")
    plot_mean(goal[:,3], x, ml5[:,3],4,"MLM"+addendum, ml10[:,3], "ML5", "ML10")

    plot_mean(goal[:,0], x, w_mean[:,0],1,"WMvsML5"+addendum, ml5[:,0], "WM", "ML5")
    plot_mean(goal[:,1], x, w_mean[:,1],2,"WMvsML5"+addendum, ml5[:,1], "WM", "ML5")
    plot_mean(goal[:,2], x, w_mean[:,2],3,"WMvsML5"+addendum, ml5[:,2], "WM", "ML5")
    plot_mean(goal[:,3], x, w_mean[:,3],4,"WMvsML5"+addendum, ml5[:,3], "WM", "ML5")

    plot_mean(goal[:,0], x, w_mean[:,0],1,"WMvsML10"+addendum, ml10[:,0], "WM", "ML10")
    plot_mean(goal[:,1], x, w_mean[:,1],2,"WMvsML10"+addendum, ml10[:,1], "WM", "ML10")
    plot_mean(goal[:,2], x, w_mean[:,2],3,"WMvsML10"+addendum, ml10[:,2], "WM", "ML10")
    plot_mean(goal[:,3], x, w_mean[:,3],4,"WMvsML10"+addendum, ml10[:,3], "WM", "ML10")

def find_most_likely(Xs,Ys):
    #print("Xs = {0}".format(str(Xs)))
    estimations = np.zeros(Ys.shape[0])
    for i in range(Xs.shape[0]):
        prob = 1
        for y in Ys:
            #print("y = {0}\nXi = {1}".format(str(y), str(Xs[i])))
            prob = prob*multinomial_pdf(y,Xs[i])
        estimations[i] = prob
    return Xs[np.argmax(estimations, axis = 0)]

def testing_suite(iterations, thread, q, q_means, q_weighted, q_ml5, q_ml10, q_predictions, q_goal, plot, particles,w, a):
    intento = iterations
    wgts = [0.02, 0.03, 0.05, 0.08, 0.08, 0.09, 0.10, 0.10, 0.20, 0.25]
    Estudiante = Student(np.array([[0.8, 0.1, 0.09, 0.01], [0.3, 0.3, 0.2, 0.2], [0.5,0.05,0.05,0.4], [0.0,0.3,0.3,0.4], [0.0,0.3,0.3,0.4]]),particles,w, a)
    estimaciones = np.zeros((intento,4))
    question = q
    x = range(intento)
    y = np.zeros((intento,4))
    mean_y = np.zeros((intento,4))
    goal = np.zeros((intento,4))
    most_likely_last5 = np.zeros((intento,4))
    most_likely_last10 = np.zeros((intento,4))
    answers = np.zeros((intento,4))
    for intentos in range(intento):
        Estudiante.updateState(question,intentos)
        answers[intentos] = Estudiante.answer(question)
        Estudiante.updateEstimation(answers[intentos],question)
        estimaciones[intentos] = Estudiante.retrieveEstimation(question)
        y[intentos] = Estudiante.retrieveEstimation(question)
        goal[intentos] = Estudiante.getState(question)
        #print("iteration {0}".format(intentos))
        if (intentos < 10):
            mean_y[intentos] = np.average(y[0:intentos+1], axis = 0, weights = wgts[0: intentos + 1]) #weighted mean
            estimaciones[intentos] = np.average(y[0:intentos+1], axis = 0) #full mean
            if intentos < 5:
                most_likely_last5[intentos] = find_most_likely(y[0:intentos+1], answers[0:intentos+1]) #most likely to have produced last 5 data points
            else:
                most_likely_last5[intentos] = find_most_likely(y[5:intentos+1], answers[5:intentos+1])
            most_likely_last10[intentos] = find_most_likely(y[0:intentos+1], answers[0:intentos+1])
            #print y[0:intentos+1]
        else:
            mean_y[intentos] = np.average(y[0:intentos+1][-10:], axis = 0, weights = wgts)
            estimaciones[intentos] = np.average(y[0:intentos+1], axis = 0)
            most_likely_last5[intentos] = find_most_likely(y[0:intentos+1][-5:], answers[0:intentos+1][-5:]) #most likely to have produced last 5 data points
            most_likely_last10[intentos] = find_most_likely(y[0:intentos+1][-10:], answers[0:intentos+1][-10:]) #most likely to have produced last 10 data points
            #print y[0:intentos+1][-10:]

    mean = np.mean(y, axis = 0)
    if (plot == 1):
        plot_estimates(y,x,goal,mean_y, estimaciones, most_likely_last5, most_likely_last10,thread)
    q_means.put(estimaciones)
    q_weighted.put(mean_y)
    q_ml5.put(most_likely_last5)
    q_ml10.put(most_likely_last10)
    q_predictions.put(y)
    q_goal.put(goal)

jobs = []
question = 4 #indexed from 0 to 4
question_options = 4
iterations = 100
threads = 5
particles = 1000
w = 0.80
a = 25

means = np.zeros((threads,iterations,question_options))
weighted_means = np.zeros((threads,iterations,question_options))
ml5 = np.zeros((threads,iterations,question_options))
ml10 = np.zeros((threads,iterations,question_options))
predictions = np.zeros((threads,iterations,question_options))

Q_means = Queue()
Q_weighted = Queue()
Q_ml5 = Queue()
Q_ml10 = Queue()
Q_predictions = Queue()
Q_goal = Queue()
plot = 0 # Only plots of the means will be generated, if plot == 1 then plots for every iteration will be generated

for i in range(threads):
    p = Process(target = testing_suite, args = (iterations,i,question, Q_means, Q_weighted, Q_ml5, Q_ml10, Q_predictions, Q_goal, plot, particles,w,a))
    jobs.append(p)
    p.start()

for i in range(threads):
    means[i] = Q_means.get()
    weighted_means[i] = Q_weighted.get()
    ml5[i] = Q_ml5.get()
    ml10[i] = Q_ml10.get()
    predictions[i] = Q_predictions.get()

x = range(iterations)
goal = Q_goal.get()
total_means = np.average(means, axis = 0)
total_weighted = np.average(weighted_means, axis = 0)
total_ml5 = np.average(ml5, axis = 0)
total_ml10 = np.average(ml10, axis = 0)
total_predictions = np.average(predictions, axis = 0)

mean_mse = MSE(goal,total_means)
mean_mse0 = "Mean|State 1: "+ str(MSE(goal[:,0], total_means[:,0]))
mean_mse1 = "Mean|State 2: "+ str(MSE(goal[:,1], total_means[:,1]))
mean_mse2 = "Mean|State 3: "+ str(MSE(goal[:,2], total_means[:,2]))
mean_mse3 = "Mean|State 4: "+ str(MSE(goal[:,3], total_means[:,3]))

weighted_mse = MSE(goal,total_weighted)
weighted_mse0 = "Weighted Mean MSE|State 1: "+str(MSE(goal[:,0], total_weighted[:,0]))
weighted_mse1 = "Weighted Mean MSE|State 2: "+str(MSE(goal[:,1], total_weighted[:,1]))
weighted_mse2 = "Weighted Mean MSE|State 3: "+str(MSE(goal[:,2], total_weighted[:,2]))
weighted_mse3 = "Weighted Mean MSE|State 4: "+str(MSE(goal[:,3], total_weighted[:,3]))

ml5_mse = MSE(goal,total_ml5)
ml5_mse0 = "ML5|State 1: " + str(MSE(goal[:,0], total_ml5[:,0]))
ml5_mse1 = "ML5|State 2: " + str(MSE(goal[:,1], total_ml5[:,1]))
ml5_mse2 = "ML5|State 3: " + str(MSE(goal[:,2], total_ml5[:,2]))
ml5_mse3 = "ML5|State 4: " + str(MSE(goal[:,3], total_ml5[:,3]))

ml10_mse = MSE(goal,total_ml10)
ml10_mse0 = "ML10|State 1: " + str(MSE(goal[:,0], total_ml10[:,0]))
ml10_mse1 = "ML10|State 2: " + str(MSE(goal[:,1], total_ml10[:,1]))
ml10_mse2 = "ML10|State 3: " + str(MSE(goal[:,2], total_ml10[:,2]))
ml10_mse3 = "ML10|State 4: " + str(MSE(goal[:,3], total_ml10[:,3]))

predictions_mse = MSE(goal,total_predictions)
predictions_mse0 = "Raw Predictions|State 1: " + str(MSE(goal[:,0], total_predictions[:,0]))
predictions_mse1 = "Raw Predictions|State 2: " + str(MSE(goal[:,1], total_predictions[:,1]))
predictions_mse2 = "Raw Predictions|State 3: " + str(MSE(goal[:,2], total_predictions[:,2]))
predictions_mse3 = "Raw Predictions|State 4: " + str(MSE(goal[:,3], total_predictions[:,3]))

split_plot(weighted_mse0,weighted_mse1,weighted_mse2,weighted_mse3,mean_mse0,mean_mse1,mean_mse2,mean_mse3,goal,x,total_weighted,total_means, "Weighted Mean", "Mean", "Global Weighted Mean vs Full Mean", "Weighted Mean vs Mean\nWeighted Mean MSE: " + str(weighted_mse) + "      Mean MSE: " + str(mean_mse))
split_plot(weighted_mse0,weighted_mse1,weighted_mse2,weighted_mse3,ml5_mse0,ml5_mse1,ml5_mse2,ml5_mse3,goal,x,total_weighted, total_ml5, "Weighted Mean", "ML5", "Global Weighted Mean vs ML5", "Weighted Mean vs ML5\nWeighted Mean MSE: "+str(weighted_mse)+"      ML5 MSE: "+str(ml5_mse))
split_plot(weighted_mse0,weighted_mse1,weighted_mse2,weighted_mse3,ml10_mse0,ml10_mse1,ml10_mse2,ml10_mse3,goal,x,total_weighted, total_ml10, "Weighted Mean", "ML10", "Global Weighted Mean vs ML10", "Weighted Mean vs ML10\nWeighted Mean MSE: "+ str(weighted_mse)+"      ML10 MSE: "+str(ml10_mse))
split_plot(predictions_mse0,predictions_mse1,predictions_mse2,predictions_mse3,weighted_mse0,weighted_mse1,weighted_mse2,weighted_mse3,goal,x,total_predictions, total_weighted,"Predictions", "Weighted Mean","Global Predictions Mean vs Weighted Mean", "Raw Predictions vs Weighted Mean\nRaw Predictions MSE: " +str(predictions_mse)+"      Weighted Mean MSE: " + str(weighted_mse))
split_plot(ml5_mse0,ml5_mse1,ml5_mse2,ml5_mse3,ml10_mse0,ml10_mse1,ml10_mse2,ml10_mse3,goal,x,total_ml5,total_ml10,"ML5", "ML10","Global ML5 vs ML10", "ML5 vs ML10\nML5 MSE: " + str(ml5_mse) + "      ML10 MSE: " + str(ml10_mse))

for i in range(len(jobs)):
    print(i)
    jobs[i].join()
