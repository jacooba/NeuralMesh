
import constants as c

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt


#Neural net: 
# recurrent network where values can only be transfered to adjacent neurons and energy is conserved
# since it can wrap arround, the mesh is really in the shape of a taurus
class NeuralMesh:
    def __init__(self, mesh_rows, mesh_cols, saveable=True):
        self.mesh_rows, self.mesh_cols = mesh_rows, mesh_cols
        self.saveable = saveable

        self.sess = tf.Session()
        self.defineGraph()
        if self.saveable:
            self.saver = tf.train.Saver()
            self.summary_writer = tf.summary.FileWriter(c.SUMMARY_DIR, self.sess.graph)

            checkPoint = tf.train.get_checkpoint_state(c.MODEL_DIR)
            modelExists = checkPoint and checkPoint.model_checkpoint_path
            if modelExists:
                self.saver.restore(self.sess, checkPoint.model_checkpoint_path)
            else:
                self.sess.run(tf.global_variables_initializer())
        else:
            self.sess.run(tf.global_variables_initializer())

    def defineGraph(self):
        self.glob_step = tf.Variable(0, name="global_step", trainable=False)

        if  c.NORMALIZE_WEIGHTS:
            self.initializer = tf.random_uniform_initializer()
        else:
            self.initializer = tf.random_normal_initializer(stddev=0.1)
        self.img_batch_in = tf.placeholder(tf.float32, shape=[c.BATCH_SZ, c.WINDOW_SZ, c.IMG_VEC_SZ]) #None is batch_sz
        self.labels = tf.placeholder(tf.int64, shape=[c.BATCH_SZ])

        #can use code bellow to get baseline =  shallow fully connected network:
        # img_batch_in = self.img_batch_in[:,0,:] #get first thing in windows
        # weights = tf.get_variable("W", dtype=tf.float32, shape=[c.IMG_VEC_SZ, self.mesh_rows*self.mesh_cols], initializer=self.initializer)
        # biases = tf.get_variable("B", dtype=tf.float32, shape=[self.mesh_rows*self.mesh_cols], initializer=self.initializer)
        # cur_state = tf.matmul(img_batch_in, weights) + biases
        # cur_state = tf.reshape(cur_state, [c.BATCH_SZ, self.mesh_rows,self.mesh_cols])
        # cur_state = tf.nn.relu(cur_state)



        #starting state for network
        #cur_state = tf.Variable(tf.zeros([self.mesh_rows, self.mesh_cols], dtype=tf.float32), trainable=c.TRAINABLE_INIT_STATE) #for trainable first state????
        cur_state = tf.get_variable("STATE", dtype=tf.float32, shape=[c.BATCH_SZ, self.mesh_rows, self.mesh_cols], initializer=tf.zeros_initializer(dtype=tf.float32))

        weights_in = tf.get_variable("W_IN", dtype=tf.float32, shape=[self.mesh_rows, self.mesh_cols, c.IMG_VEC_SZ], initializer=self.initializer)
        if c.BIAS_ON_INPUT:
            biases_in = tf.get_variable("B_IN", dtype=tf.float32, shape=[self.mesh_rows, self.mesh_cols], initializer=self.initializer)

        #-1 to 1 allows for inhibition and excitation effects (but does cuase E to tend to 0)
        W_D = tf.tanh(tf.get_variable("W_D", dtype=tf.float32, shape=[self.mesh_rows, self.mesh_cols], initializer=self.initializer)) #the weights fot fireing DOWN
        W_U = tf.tanh(tf.get_variable("W_U", dtype=tf.float32, shape=[self.mesh_rows, self.mesh_cols], initializer=self.initializer)) #the weights fot fireing UP
        W_R = tf.tanh(tf.get_variable("W_R", dtype=tf.float32, shape=[self.mesh_rows, self.mesh_cols], initializer=self.initializer)) #the weights fot fireing LEFT
        W_L = tf.tanh(tf.get_variable("W_L", dtype=tf.float32, shape=[self.mesh_rows, self.mesh_cols], initializer=self.initializer)) #the weights fot fireing RIGHT
        if c.NORMALIZE_WEIGHTS:
            W_SUM = self.safe_absolute_tensor(W_D) + self.safe_absolute_tensor(W_U) + self.safe_absolute_tensor(W_R) + self.safe_absolute_tensor(W_L)
            W_D, W_U, W_R, W_L = W_D/W_SUM, W_U/W_SUM, W_R/W_SUM, W_L/W_SUM

        D_SHIFT_MATRIX = tf.constant(np.roll(np.identity(self.mesh_rows), 1, axis=0), dtype=tf.float32) # D_SHIFT_MATRIX * [] => shift all rows down 1
        U_SHIFT_MATRIX = tf.constant(np.roll(np.identity(self.mesh_rows), -1, axis=0), dtype=tf.float32)
        R_SHIFT_MATRIX = tf.constant(np.roll(np.identity(self.mesh_cols), 1, axis=1), dtype=tf.float32) # [] * R_SHIFT_MATRIX => shift all cols right 1
        L_SHIFT_MATRIX = tf.constant(np.roll(np.identity(self.mesh_cols), -1, axis=1), dtype=tf.float32)

        #define recurrent network
        all_states_lists = [cur_state]
        for i in range(c.WINDOW_SZ):    
            curr_batch_input = self.img_batch_in[:,i,:] #get the input for this window
            #state_batch_reset = tf.nn.relu(cur_state) if c.MAX_TO_0_EACH_STEP else cur_state #reset the energies of the previous state. (introduce E b/c energy goes down when used for inhibition)
            state_batch_in = tf.tensordot(curr_batch_input, weights_in, axes=([1],[2])) #dense layer to state input
            if c.BIAS_ON_INPUT:
                state_batch_in += biases_in
            if c.CLIP_NEURONS_TO_NEG1_AND_1: #state_batch in never needs to be > 2 or <-2 in this case
                state_batch_in = 2.0*tf.nn.tanh(state_batch_in)

            
            cur_state = tf.nn.relu(cur_state) if c.MAX_TO_0_EACH_STEP else cur_state #reset energries if necessary
            cur_state += state_batch_in #new state given input E
            cur_state = self.clip_neurons(cur_state) if c.CLIP_NEURONS_TO_NEG1_AND_1 else cur_state #clip if necessary

            state_batch_to_fire = tf.nn.relu(cur_state) #only fire if E>0 (after new energy)
            if c.DEPLETE_TO_NEG_1: #give it the additional energey it deserves to deplete to -1
                state_batch_to_fire = safe_add_1_where_pos(state_batch_to_fire) #can use [-1,0] energy to fire as well (if >0)

            to_shift_D = tf.multiply(W_D, state_batch_to_fire)
            to_shift_U = tf.multiply(W_U, state_batch_to_fire)
            to_shift_R = tf.multiply(W_R, state_batch_to_fire)
            to_shift_L = tf.multiply(W_L, state_batch_to_fire)

            # NOTE: replaced lines bellow because broadcasting doesnt work right with matmul, only numpy matmul
            # shifted_inc_D = tf.matmul(D_SHIFT_MATRIX, to_shift_D)
            # shifted_inc_U = tf.matmul(U_SHIFT_MATRIX, to_shift_U)
            # shifted_inc_R = tf.matmul(to_shift_R, R_SHIFT_MATRIX)
            # shifted_inc_L = tf.matmul(to_shift_L, L_SHIFT_MATRIX)
            #shift energies in each direction
            shifted_inc_D = tf.transpose(tf.tensordot(D_SHIFT_MATRIX, to_shift_D, axes=([1],[1])), [1, 0, 2])
            shifted_inc_U = tf.transpose(tf.tensordot(U_SHIFT_MATRIX, to_shift_U, axes=([1],[1])), [1, 0, 2])
            shifted_inc_R = tf.tensordot(to_shift_R, R_SHIFT_MATRIX, axes=([2],[0]))
            shifted_inc_L = tf.tensordot(to_shift_L, L_SHIFT_MATRIX, axes=([2],[0]))

            #compute differentiable absolute value of energy to be removed
            abs_to_shift_D = self.safe_absolute_tensor(to_shift_D)
            abs_to_shift_U = self.safe_absolute_tensor(to_shift_U)
            abs_to_shift_R = self.safe_absolute_tensor(to_shift_R)
            abs_to_shift_L = self.safe_absolute_tensor(to_shift_L)

            state_increment = shifted_inc_D + shifted_inc_U + shifted_inc_R + shifted_inc_L #transfer energy to adjacent neurons
            state_decrement = abs_to_shift_D + abs_to_shift_U + abs_to_shift_R + abs_to_shift_L #decrement the energies from where they came

            # Compute output after energy transfer #
            if not c.NORMALIZE_WEIGHTS: #hack to not use more energy then we have if not normalizing (but also allows one neuron to fire fore more than one timestep)
                state_increment = state_increment * 0.25 #scale so that a stimulated neuron can fire repeatedly AND doesn NOT tranfer more E than it contains
                state_decrement = state_decrement * 0.25

                state_out = cur_state + state_increment - state_decrement #combine to determine the new state
            else:
                if c.DEPLETE_TO_NEG_1:
                    state_out = cur_state + state_increment - state_decrement
                else: #deplete to 0
                    #get rid of all positive energies used to fire then add shifted energies
                    state_out = tf.minimum(cur_state, 0.0) + state_increment
            # clip if necessary #
            state_out = self.clip_neurons(state_out) if c.CLIP_NEURONS_TO_NEG1_AND_1 else state_out 

            cur_state = state_out
            all_states_lists.append(state_out)

        #state was [batch_sz, rows, cols]
        #should npow be [batch_sz, window_sz, row, cols]
        self.all_states = tf.stack(all_states_lists, axis=1)
        if not c.CLIP_NEURONS_TO_NEG1_AND_1:
            self.all_states = tf.nn.sigmoid(self.all_states)


        #state -> logits
        weights_out = tf.get_variable("W_OUT", dtype=tf.float32, shape=[self.mesh_rows, self.mesh_cols, 10], initializer=self.initializer)
        biases_out = tf.get_variable("B_OUT", dtype=tf.float32, shape=[10], initializer=self.initializer)
        logits = tf.tensordot(cur_state, weights_out, axes=([1,2],[0,1])) #dot product => reduce to 10 numbers. broadcast to meet batch_sz.
        logits = logits + biases_out

        self.probabilities = tf.nn.softmax(logits)

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels))

        self.trainOp = tf.train.AdamOptimizer(learning_rate=c.LEARN_RATE).minimize(self.loss, global_step=self.glob_step)

        self.preds = tf.argmax(self.probabilities, axis=1)

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.preds, self.labels), tf.float32))

        if self.saveable:
            self.train_loss_summary = tf.summary.scalar('train_loss', self.loss)
            self.test_loss_summary = tf.summary.scalar('test_loss', self.loss)

    def safe_absolute_tensor(self, tensor):
        if c.ABS_APPROX:
            return tf.sqrt(tf.square(tensor)+c.EPS**2) - c.EPS 
        return tf.abs(tensor)

    #clips nuerons to -1 and 1
    def clip_neurons(self, tensor):
        return tf.minimum(tf.maximum(tensor, -1.0), 1.0)

    #takes in a tensor that has been just maxed with 0
    #adds 1 to positive values in a differentiable way. Note, doesnt quite add 1 to realy small values
    def safe_add_1_where_pos(self, tensor):
        return tensor + (tensor/(tensor+c.EPS))

    def save(self):
        if self.saveable:
            self.saver.save(self.sess, c.MODEL_PATH_PREFIX, global_step = self.glob_step)

    def train(self, trainImagesLabelsTup, valImagesLabelsTup=None):

        #prep data
        windowsArr, labelAnsArr = extractWindowsAndLabels(trainImagesLabelsTup)
        #prep val data
        if valImagesLabelsTup:
            valImages, valLabels = valImagesLabelsTup
            valImages, valLabels = np.repeat(valImages, int(c.BATCH_SZ/c.NUM_TEST_IMGS), axis=0), np.repeat(valLabels, int(c.BATCH_SZ/c.NUM_TEST_IMGS)) #hack to fit in batch size
            valWindows, valLabel = extractWindowsAndLabels(valImagesLabelsTup)

        #train
        for epoch in range(c.NUM_EPOCHS):
            step = epoch
            #catch up to saved state if resuming training. (dont do anything if step/epoch is too early)
            if step < tf.train.global_step(self.sess, self.glob_step):
                continue

            print("\nTrain Batch (Epoch):", step)

            feedDict = {self.img_batch_in: windowsArr, self.labels: labelAnsArr}
            if self.saveable:
                sessArgs = [self.accuracy, self.loss, self.train_loss_summary, self.trainOp]
                acc, lossReturned, summary, _ = self.sess.run(sessArgs, feed_dict=feedDict)
            else:
                sessArgs = [self.accuracy, self.loss, self.trainOp]
                acc, lossReturned, _ = self.sess.run(sessArgs, feed_dict=feedDict)

            print("loss -", lossReturned)
            print("accuracy -", acc)

            #save model/summary and validate
            if self.saveable:
                if (step%c.SAVE_FREQ == 0) or (step==c.NUM_EPOCHS-1):
                    self.save()
                if (step%c.SUMMARY_FREQ == 0) or (step==c.NUM_EPOCHS-1):
                    self.summary_writer.add_summary(summary, global_step=step)
                    if valImagesLabelsTup:
                        print("\n\nvalidating...")
                        feedDict = {self.img_batch_in: valWindows, self.labels: valLabel}
                        acc, valSummary = self.sess.run([self.accuracy, self.test_loss_summary], feed_dict=feedDict)
                        self.summary_writer.add_summary(valSummary, global_step=step)
                        print("val accuracy::", acc, "\n\n")



    def test(self, testImagesLabelsTup, display_energy_num=0):
        #prep data. repeat data enough to fill batch (hack)
        images, labels = testImagesLabelsTup
        images, labels = np.repeat(images, int(c.BATCH_SZ/c.NUM_TEST_IMGS), axis=0), np.repeat(labels, int(c.BATCH_SZ/c.NUM_TEST_IMGS))
        windowsArr, labelAnsArr = extractWindowsAndLabels(testImagesLabelsTup)

        feedDict = {self.img_batch_in: windowsArr, self.labels: labelAnsArr}

        if not display_energy_num:

            # RUN
            sessArgs = [self.accuracy, self.loss]
            acc, lossReturned = self.sess.run(sessArgs, feed_dict=feedDict)

        else: #display energy motion

            # RUN
            sessArgs = [self.accuracy, self.loss, self.all_states]
            acc, lossReturned, all_states = self.sess.run(sessArgs, feed_dict=feedDict)

            #[display_energy_num, window_sz, rows, cols]
            windows_of_interest = all_states[:display_energy_num,:,:,:]
            # assert windows_of_interest.shape == (display_energy_num, c.WINDOW_SZ, c.MESH_ROWS, c.MESH_COLS)
            n, w, _, _ = windows_of_interest.shape

            print("Energies for labels:", testImagesLabelsTup[1][:display_energy_num])
            figure, plots_array = plt.subplots(n, w)
            for window_num in range(n): 
                for state_num in range(w):
                    subplot = plots_array[window_num][state_num]
                    subplot.imshow(windows_of_interest[window_num][state_num][:,:], cmap='inferno')
                    subplot.axis('off')
                    # plt.subplot(n, w, window_num * w + state_num + 1)
                    # plt.imshow(windows_of_interest[window_num][state_num][:,:], cmap='inferno')
            plt.suptitle("Energies Over Time")
            # figure.subplots_adjust(hspace=0)
            # plt.xlabel("Time Step")
            # plt.ylabel("Window Number")
            plt.show()

        print("TEST loss -", lossReturned)
        print("TEST accuracy -", acc)
        return acc


#takes in a batch of images [[img],[img],...] and returns [[window],[window],...], where the window starts with the img
def windowifyImgBatch(images):
    windows = []
    for imgVec in images:
        vecToBroadcast = imgVec if c.USE_INPUT_AS_RESIDUAL else c.RESIDUAL_VEC
        window = [imgVec] + [vecToBroadcast for _ in range(c.WINDOW_SZ - 1)]
        windows.append(window)
    return windows

#unzips the tuples, makes windows, and converts to numpy array
def extractWindowsAndLabels(imagesLabelsTup):
    images = imagesLabelsTup[0]
    labelAns = imagesLabelsTup[1]
    windows = windowifyImgBatch(images)
    return np.array(windows), np.array(labelAns)

