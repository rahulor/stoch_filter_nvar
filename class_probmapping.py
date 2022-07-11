from scipy.special import  expit, logit # sigmoid, and inverse sigmoid function

class ProbabilityMapping():
    def __init__(self):
        self.a = 10.0
   
    def prob_to_y(self, p):
        y = logit(p)/self.a
        return(y)
    
    def y_to_prob(self, y):
        p = expit(self.a*y)
        return(p)
probmap = ProbabilityMapping()