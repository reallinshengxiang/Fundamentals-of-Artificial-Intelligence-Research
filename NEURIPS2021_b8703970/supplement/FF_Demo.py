'''
FF_Demo: A module file for creating, training, and testing recurrent neural networks using full-FORCE.
Created by Eli Pollock, Jazayeri Lab, MIT 12/13/2017
'''

import numpy as np
import numpy.random as npr
from scipy import sparse
import matplotlib.pyplot as plt


def create_parameters(dt=0.001):
    '''Use this to define hyperparameters for any RNN instantiation. You can create an "override" script to 
    edit any individual parameters, but simply calling this function suffices. Use the output dictionary to 
    instantiate an RNN class'''
    p = {'network_size': 300,              # Number of units in network
        'dt': dt,                         # Time step for RNN.
        'tau': 0.01,                     # Time constant of neurons
        'noise_std': 0,                    # Amount of noise on RNN neurons
        'g': 1,                         # Gain of the network (scaling for recurrent weights)
        'p': 1,                            # Controls the sparseness of the network. 1=>fully connected.
        'inp_scale': 1/2,                 # Scales the initialization of input weights
        'out_scale': 1/10,                 # Scales the initialization of output weights
        'bias_scale': 0,                # Scales the amount of bias on each unit
        'init_act_scale' : 1,            # Scales how activity is initialized
        ##### Training parameters for full-FORCE 
        'ff_steps_per_update': 2 ,  # Average number of steps per weight update
        'ff_alpha': 1,  # "Learning rate" parameter (should be between 1 and 100)
        'ff_num_batches': 10,
        'ff_trials_per_batch': 100,  # Number of inputs/targets to go through
        'ff_init_trials': 3,
        #### Testing parameters
        'test_trials': 10,
        'test_init_trials': 1,
    
    }
    return p



class RNN:
    '''
    Creates an RNN object. Relevant methods:
        __init___:             Creates attributes for hyperparameters, network parameters, and initial activity
        initialize_act:     Resets the activity
        run:                 Runs the network forward given some input
        train:                Uses one of several algorithms to train the network.
    '''

    def __init__(self, hyperparameters, num_inputs, num_outputs):
        '''Initialize the network
        Inputs:
            hyperparameters: should be output of create_parameters function, changed as needed
            num_inputs: number of inputs into the network
            num_outputs: number of outputs the network has
        Outputs:
            self.p: Assigns hyperparameters to an attribute
            self.rnn_par: Creates parameters for network that can be optimized (all weights and node biases)
            self.act: Initializes the activity of the network'''
        self.p = hyperparameters
        rnn_size = self.p['network_size']
        self.rnn_par = {'inp_weights': (npr.randn(num_inputs, rnn_size)) * self.p['inp_scale'],
                        'rec_weights': np.array(sparse.random(rnn_size, rnn_size, self.p['p'], 
                                                data_rvs = npr.randn).todense()) * self.p['g']/np.sqrt(rnn_size),
                        'out_weights': (npr.randn(rnn_size, num_outputs)) * self.p['out_scale'],
                        'bias': (npr.rand(1,rnn_size)-0.5)*2 * self.p['bias_scale']
                        }
        self.act = npr.randn(1,self.rnn_par['rec_weights'].shape[0])*self.p['init_act_scale']

    def performUpdate(self, currTimeStep):
        if ((currTimeStep==0) or (currTimeStep >= self.errorTrigger)):
            return True
        else:
            return False

    def initialize_act(self):
        '''Any time you want to reset the activity of the network to low random values'''
        self.act = npr.randn(1,self.rnn_par['rec_weights'].shape[0])*self.p['init_act_scale']

    def run(self,inputs, record_flag=0):
        '''Use this method to run the RNN on some inputs
        Inputs:
            inputs: An Nxm array, where m is the number of separate inputs and N is the number of time steps
            record_flag: If set to 0, the function only records the output node activity
                If set to 1, if records that and the activity of every hidden node over time
        Outputs:
            output_whole: An Nxk array, where k is the number of separate outputs and N is the number of time steps
            activity_whole: Either an empty array if record_flag=0, or an NxQ array, where Q is the number of hidden units

        '''
        p = self.p
        activity = self.act
        rnn_params = self.rnn_par
        def rnn_update(inp, activity):
            # updated to reflect tanh()+1 activation instead of tanh()
            dx = p['dt']/p['tau'] * (-activity + np.dot(np.tanh(activity),rnn_params['rec_weights']) + 
                                        np.dot(inp, rnn_params['inp_weights']) + rnn_params['bias'] + 
                                        npr.randn(1,activity.shape[1]) * p['noise_std'])
            return activity + dx

        def rnn_output(activity):
            return np.tanh( np.dot(np.tanh(activity), rnn_params['out_weights']) )

        activity_whole = []
        output_whole = []
        if record_flag==1:  # Record output and activity only if this is active
            activity_whole = np.zeros(((inputs.shape[0]),rnn_params['rec_weights'].shape[0]))
            t=0
         
        for inp in inputs:
            activity = rnn_update(inp, activity)
            output_whole.append(rnn_output(activity))
            if record_flag==1:
                activity_whole[t,:] = activity
                t+=1
        output_whole = np.reshape(output_whole, (inputs.shape[0],-1))
        self.act = activity
        
        return output_whole, activity_whole


        
        
    def train_rdm(self,inps_and_targs, monitor_training=0, errorTrigger = 10, **kwargs):
        '''Use this method to train the RNN using one of several training algorithms!
        Inputs:
            inps_and_targs: This should be a FUNCTION that randomly produces a training input and a target function
                            Those should have individual inputs/targets as columns and be the first two outputs 
                            of this function. Should also take a 'dt' argument.
            monitor_training: Collect useful statistics and show at the end
            **kwargs: use to pass things to the inps_and_targs function
        Outputs:
        Nothing explicitly, but the weights of self.rnn_par are optimized to map the inputs to the targets
        Use this to train the network according to the full-FORCE algorithm, described in DePasquale 2017
        This function uses a recursive least-squares algorithm to optimize the network.
        Note that after each batch, the function shows an example output as well as recurrent unit activity.
        Parameters: 
            In self.p, the parameters starting with ff_ control this function. 

        *****NOTE***** The function inps_and_targs must have a third output of "hints" for training.
        If you don't want to use hints, replace with a vector of zeros (Nx1)

        '''
        # First, initialize some parameters
        self.errorTrigger = errorTrigger
        p = self.p
        self.initialize_act()
        N = p['network_size']
        self.rnn_par['rec_weights'] = np.zeros((N,N))
        self.rnn_par['out_weights'] = np.zeros((self.rnn_par['out_weights'].shape))
        # create an activity tensor that will hold activities through time during trials
        activity_tensor = np.zeros((5000, 75, self.p["network_size"]))
        activity_targets = np.zeros((5000, 1))


        # Need to initialize a target-generating network, used for computing error:
        # First, take some example inputs, targets, and hints to get the right shape
        try:
            D_inputs, D_targs, D_hints = inps_and_targs(dt=p['dt'], **kwargs)[0:3]
            D_num_inputs = D_inputs.shape[1]
            D_num_targs = D_targs.shape[1]
            D_num_hints = D_hints.shape[1]
            D_num_total_inps = D_num_inputs + D_num_targs + D_num_hints
        except:
            raise ValueError('Check your inps_and_targs function. Must have a hints output as well!')

        # Then instantiate the network and pull out some relevant weights
        DRNN = RNN(hyperparameters = self.p, num_inputs = D_num_total_inps, num_outputs = 1)
        w_targ = np.transpose(DRNN.rnn_par['inp_weights'][D_num_inputs:(D_num_inputs+D_num_targs),:])
        w_hint = np.transpose(DRNN.rnn_par['inp_weights'][(D_num_inputs+D_num_targs):D_num_total_inps,:])
        Jd = np.transpose(DRNN.rnn_par['rec_weights'])

        ################### Monitor training with these variables:
        J_err_ratio = []
        J_err_mag = []
        J_norm = []

        self.valHist = []

        w_err_ratio = []
        w_err_mag = []
        w_norm = []
        ###################

        # Let the networks settle from the initial conditions
        print('Initializing',end="")
        for i in range(p['ff_init_trials']):
            print('.',end="")
            inp, targ, hints = inps_and_targs(dt=p['dt'], **kwargs)[0:3]
            D_total_inp = np.hstack((inp,targ,hints))
            DRNN.run(D_total_inp)
            self.run(inp)
        print('')

        # Now begin training
        print('Training network...')
        # Initialize the inverse correlation matrix
        P = np.eye(N)/p['ff_alpha']
        # initialize training counters
        trial_count = 0
        validation_acc = 0
        loss_hist = []
        W_in_hist = []
        W_rec_hist = []
        #while( validation_acc < 0.9):       
        validation_accuracy = 0.0
        validation_acc_hist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        ##################################################################
        # TRAINING LOOP 
        ##################################################################
        num_train_attempts = 1
        while(trial_count < 5000 and validation_accuracy < 0.9):
            self.initialize_act()
            # Create input, target, and hints. Combine for the driven network
            # want to create as a batch
            inp = np.zeros((75,100))   # for a batch of size 100
            targ = np.zeros((75, 100))
            hints = np.zeros((75, 100))
            inp, targ, hints = inps_and_targs(dt=p['dt'], **kwargs)[0:3]
            D_total_inp = np.hstack((inp,targ,hints))
            # For recording:
            dx = [] # Driven network activity
            x = []    # RNN activity
            z = []    # RNN output
            # update the current weight matrix histories
            # NOTE: THE WEIGHTS MAY BE UPDATED MULTIPLE TIMES PER TRIAL
            # WE ARE ONLY RECORDING THE WEIGHTS AT THE BEGINIGN OF EACH TRIAL
            # AND THEREFORE ARE NOT KEEPING A COMPLETE WEIGHT HISTORY
            W_in_hist.append(self.rnn_par['inp_weights'])
            W_rec_hist.append(self.rnn_par['rec_weights'])
            for t in range(len(inp)):
                # Run both RNNs forward and get the activity. Record activity for potential plotting
                dx_t = DRNN.run(D_total_inp[t:(t+1),:], record_flag=1)[1][:,0:5]
                z_t, x_t = self.run(inp[t:(t+1),:], record_flag=1)
 
                # update the activity tensors
                if t%100 == 0:
                    activity_tensor[trial_count, int(t/100), :] = x_t[0,:]
                    activity_targets[trial_count, :] = targ[-1]

                dx.append(np.squeeze(np.tanh(dx_t) + np.arange(5)*2))
                z.append(np.squeeze(z_t))
                #print('z', z[-1])
                x.append(np.squeeze(np.tanh(x_t[:,0:5]) + np.arange(5)*2))

                #if npr.rand() < (1/p['ff_steps_per_update']):
                if (self.performUpdate(t)):
                    # Extract relevant values
                    r = np.transpose(np.tanh(self.act))
                    rd = np.transpose(np.tanh(DRNN.act))
                    J = np.transpose(self.rnn_par['rec_weights'])
                    w = np.transpose(self.rnn_par['out_weights'])

                    # Now for the RLS algorithm:
                    # Compute errors
                    J_err = (np.dot(J,r) - np.dot(Jd,rd) 
                            -np.dot(w_targ,targ[t:(t+1),:].T) - np.dot(w_hint, hints[t:(t+1),:].T))
                    # loss for RDM task
                    if t == 0:
                        w_err = np.dot(w,r)     # target is zero
                    else:
                        RDM_targ = targ[-1]
                        w_err = np.dot(w,r) - RDM_targ # target is +/- 1
                    loss_hist.append(w_err)

                    # Compute the gain (k) and running estimate of the inverse correlation matrix
                    Pr = np.dot(P,r)
                    k = np.transpose(Pr)/(1 + np.dot(np.transpose(r), Pr))
                    #print("Pr shape", Pr.shape)
                    #print("r shape", r.shape)
                    #assert False
                    P = P - np.dot(Pr,k)

                    # Update weights
                    w = w - np.dot(w_err, k)
                    J = J - np.dot(J_err, k)
                    self.rnn_par['rec_weights'] = np.transpose(J)
                    self.rnn_par['out_weights'] = np.transpose(w)
                    
                    # compute the validation score
            val_err = self.ValidationScore(inps_and_targs)
            self.valHist.append(val_err)
            validation_accuracy_curr = 1 - val_err
            loss_hist.append(validation_accuracy_curr)
            validation_acc_hist[:9] = validation_acc_hist[1:]
            validation_acc_hist[-1] = validation_accuracy_curr
            validation_accuracy = np.min(validation_acc_hist)
            # increment the trial counter
            trial_count += 1
            if trial_count %10 == 0:
                #print("trial #:", trial_count, "validation score:", validation_accuracy, "                                        validation hist:", validation_acc_hist)
                print("trial #:", trial_count, "validation score:", validation_accuracy)
            if trial_count == 5000:  # reset the training
                return False

        ##################################################################
        # END TRAINING LOOP 
        ##################################################################
        print('Done training!', validation_accuracy)

        # clip the activity tensors
        self.activity_tensor = activity_tensor[:trial_count+1, :, :]
        self.activity_targets = activity_targets[:trial_count+1, :]
        # store the loss history as a NumPy array attribute
        self.losses = np.array(loss_hist)
        # store the weight histories as an attribute
        self.Wrec_hist = W_rec_hist
        self.Win_hist = W_in_hist
        
        return True
        

    
    def test(self, inps_and_targs, **kwargs):
        p = self.p
        '''
        Function that tests a trained network. Relevant parameters in p start with 'test'
        Inputs:
            Inps_and_targ: function used to generate time series (same as in train)
            **kwargs: arguments passed to inps_and_targs
        '''

        self.initialize_act()
        print('Initializing',end="")
        for i in range(p['test_init_trials']):
            print('.',end="")
            inp, targ = inps_and_targs(dt=p['dt'], **kwargs)[0:2]
            self.run(inp)
        print('')

        inp, targ = inps_and_targs(dt=p['dt'], **kwargs)[0:2]
        test_fig = plt.figure()
        ax = test_fig.add_subplot(1,1,1)
        tvec = np.arange(0,len(inp))*p['dt']
        line_inp = plt.Line2D(tvec, targ, linestyle='--', color='g')
        line_targ = plt.Line2D(tvec, targ, linestyle='--', color='r')
        line_out = plt.Line2D(tvec, targ, color='b')
        ax.add_line(line_inp)
        ax.add_line(line_targ)
        ax.add_line(line_out)
        ax.legend([line_inp, line_targ, line_out], ['Input','Target','Output'], loc=1)
        ax.set_title('RNN Testing: Wait')
        ax.set_xlim([0,p['dt']*len(inp)])
        ax.set_ylim([-1.2,1.2])
        ax.set_xlabel('Time (s)')
        test_fig.canvas.draw()

        E_out = 0 # Running squared error
        V_targ = 0  # Running variance of target
        print('Testing: %g trials' % p['test_trials'])
        for idx in range(p['test_trials']):
            inp, targ = inps_and_targs(dt=p['dt'], **kwargs)[0:2]

            tvec = np.arange(0,len(inp))*p['dt']
            ax.set_xlim([0,p['dt']*len(inp)])
            line_inp.set_xdata(tvec)
            line_inp.set_ydata(inp)
            line_targ.set_xdata(tvec)
            line_targ.set_ydata(targ)
            out = self.run(inp)[0]
            line_out.set_xdata(tvec)
            line_out.set_ydata(out)
            ax.set_title('RNN Testing, trial %g' % (idx+1))
            test_fig.canvas.draw()
            
            # E_out has been edited to only consider error at the final timestep
            E_curr = np.dot(np.transpose(out[-1]-targ[-1]), out[-1]-targ[-1])
            if np.abs(E_curr) > 0.25:
                E_curr = 1
            else:
                E_curr = 0
            E_out = E_out + E_curr
            
            V_targ = 1#V_targ + np.dot(np.transpose(targ), targ)
        print('')
        E_norm = E_out/p['test_trials']
        print('Normalized error: %g' % E_norm)
        return E_norm



    # def ValidationScore(self, inps_and_targs, **kwargs):
    #     p = self.p
    #     '''
    #     Function that tests a trained network. Relevant parameters in p start with 'test'
    #     Inputs:
    #         Inps_and_targ: function used to generate time series (same as in train)
    #         **kwargs: arguments passed to inps_and_targs
    #     '''

    #     self.initialize_act()
    #     for i in range(p['test_init_trials']):
    #         inp, targ = inps_and_targs(dt=p['dt'], **kwargs)[0:2]
    #         self.run(inp)

    #     E_out = 0 # Running squared error
    #     for idx in range(p['test_trials']):
    #         inp, targ = inps_and_targs(dt=p['dt'], **kwargs)[0:2]
    #         out = self.run(inp)[0]
            
    #         # E_out has been edited to only consider error at the final timestep
    #         E_curr = np.abs( out[-1] - targ[-1] )
    #         if E_curr > 1:
    #             E_out = E_out+1

    #     E_norm = E_out/p['test_trials']
    #     return E_norm
    
    def ValidationScore(self, inps_and_targs, **kwargs):
        p = self.p
        '''
        Function that tests a trained network. Relevant parameters in p start with 'test'
        Inputs:
            Inps_and_targ: function used to generate time series (same as in train)
            **kwargs: arguments passed to inps_and_targs
        '''

        self.initialize_act()
        inp, targ = fetchBatch(inps_and_targs, p['test_trials'])[0:2]     # serves as a test batch
        # targ is (timesteps, trials, 1)
        out, _ = self.run(inp)   # out is (timesteps, trials)

        E_out = 0 # Running squared error
        E_curr = np.abs( out[-1] - targ[-1,:,0] )        # vector of length p['test_trials']
        assert(E_curr.shape[0] == p['test_trials'])      # error has one entry for each test trial
        E_out=len(np.where(E_curr>1)[0])                 # use this line for context task
        #E_out=len(np.where(E_curr>0.5)[0])              # use this line for multisensory task
        E_norm = E_out/p['test_trials']
        return E_norm


def fetchBatch(inpts_and_targs, batchSize):
    '''
    generates a batch of data from the inpts_and_targs function

    Args:
        inpts_and_targs (TYPE): DESCRIPTION.

    Returns:
        batch

    '''
    sampleInput, sampleHint, sampleTarget = inpts_and_targs()
    inputDims = sampleInput.shape
    hintDims = sampleHint.shape
    targetDims = sampleTarget.shape
    inpts = np.zeros((inputDims[0], batchSize, inputDims[1]))   
    hints = np.zeros((hintDims[0], batchSize, hintDims[1]))   
    targs = np.zeros((hintDims[0], batchSize, targetDims[1]))   
    for i in range(batchSize):
        inpts[:,i,:], hints[:,i,:], targs[:,i,:] = inpts_and_targs()
        
    return inpts, hints, targs


if __name__=="__main__":
    var = 0.001
    task = Williams(N=750, mean=.1857, variance=var)
    inps_and_targs = task.GetInput
    
    # create the network and set hyper-parameters
    p = create_parameters(dt=0.003)
    p['g'] = 1
    p['network_size'] = 50
    p['tau'] = 0.03
    p['test_init_trials']=10
    p['test_trials'] = 2000
    p['ff_alpha'] = 1000
    p['ff_steps_per_update']=2
    rnn = RNN(p,1,1)
    
    # loop over 10 models to be trained
    #for model_num in range(2,3):
    # train the current model
    rnn.train_rdm(inps_and_targs, errorTrigger=50)