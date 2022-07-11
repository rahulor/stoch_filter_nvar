#========================   nVAR configuration   ==============================
traintime = 100                      # [s] time to train for | can be 400 or 600
testtime  = 100                      # [s] time to test for  | can be 400 or 600

d = 1                               # input dimension | DO NOT CHANGE
k = 20                              # number of delay-taps | k=40: high accuracy, takes more RAM
L2penalty = 1e-3                    # ridge-parameter
nonlinearity = 3                    # 1 for linear, 2 for lin+squares, 3 for lin+squares+cubes
#------------------------------------------------------------------------------