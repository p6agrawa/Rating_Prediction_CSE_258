import data_loader
import tensorflow as tf
import random
import numpy as np

def init_weight(dim_in, dim_out, name=None, stddev=1.0):
    return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/math.sqrt(float(dim_in))), name=name)

def init_bias(dim_out, name=None):
    return tf.Variable(tf.zeros([dim_out]), name=name)

class Model:
    def __init__(self, options):
        data = data_loader.load_data()
        self.data = data
        self.options = options

        self.Wu = tf.Variable(
            tf.random_uniform(
                [data['num_users'] + 1, options['user_embedding_size']], -1.0, 1.0), 
            name = 'Wu')

        self.Bu = tf.Variable(
            tf.random_uniform(
                [data['num_users'] + 1], -1.0, 1.0), 
            name = 'Bu')

        self.Wb = tf.Variable(
            tf.random_uniform(
                [data['num_business'] + 1, options['business_embedding_size']], -1.0, 1.0), 
            name = 'Wb')

        self.Bb = tf.Variable(
            tf.random_uniform(
                [data['num_business'] + 1], -1.0, 1.0), 
            name = 'Bb')

        self.alpha = tf.Variable(
            tf.random_uniform(
                [1], -1.0, 1.0), 
            name = 'alpha')


    def build_model(self):
        user = tf.placeholder('int32',[ None, ], name = 'user')
        positive_business = tf.placeholder('int32',[ None, ], name = 'positive_business')
        rating = tf.placeholder('float32',[ None, ], name = 'rating')

        user_embedding = tf.nn.embedding_lookup( self.Wu, user)
        positive_embedding = tf.nn.embedding_lookup( self.Wb, positive_business)
        

        user_Bu = tf.nn.embedding_lookup(self.Bu, user)
        positive_Bb = tf.nn.embedding_lookup(self.Bb, positive_business)
        
        predicted_rating = tf.reduce_sum(user_embedding * positive_embedding, 1, name = "positive_logits") 
        predicted_rating += user_Bu + positive_Bb + self.alpha

        loss = tf.nn.l2_loss(predicted_rating - rating)

        reg_loss = tf.reduce_sum( (user_Bu * user_Bu) + (positive_Bb * positive_Bb) )
        reg_loss += tf.reduce_sum(positive_embedding * positive_embedding) + tf.reduce_sum(user_embedding * user_embedding)
        reg_loss = self.options['lambda'] * reg_loss

        
        loss += reg_loss

        return {
            'user' : user,
            'positive_business' : positive_business,
            'rating' : rating,
            'loss' : loss,
            'prediction' : predicted_rating
        }

    def evaluator(self):
        user = tf.placeholder('int32',[ None, ], name = 'user')
        positive_business = tf.placeholder('int32',[ None, ], name = 'positive_business')
        
        user_embedding = tf.nn.embedding_lookup( self.Wu, user)
        positive_embedding = tf.nn.embedding_lookup( self.Wb, positive_business)
        

        user_Bu = tf.nn.embedding_lookup(self.Bu, user)
        positive_Bb = tf.nn.embedding_lookup(self.Bb, positive_business)
        
        predicted_rating = tf.reduce_sum(user_embedding * positive_embedding, 1, name = "positive_logits") 
        predicted_rating += user_Bu + positive_Bb + self.alpha

        return {
            'user' : user,
            'positive_business' : positive_business,
            'prediction' : predicted_rating
        }

    def evaluate(self, sess, epoch, reuse = True):
        if reuse == True:
            tf.get_variable_scope().reuse_variables()

        model = self.evaluator()
        
        predictions = open("predictions_Rating{}.txt".format(epoch), 'w')
        for l in open("pairs_Rating.txt"):
            if l.startswith("userID"):
                predictions.write(l)
                continue
            u,i = l.strip().split('-')

            if u in self.data['u_hash_to_idx'] and i in self.data['b_hash_to_idx']:
                prediction,  = sess.run( [model['prediction']], feed_dict = {
                    model['user'] : [self.data['u_hash_to_idx'][u]],
                    model['positive_business'] : [self.data['b_hash_to_idx'][i]],
                    }
                )
            
                predictions.write(u + '-' + i + ",{}\n".format(prediction[0]))
            else:
                predictions.write(u + '-' + i + ",{}\n".format(4.184485))

        predictions.close()

    def train(self):
        
        data = []
        count = 0
        for user, businesses in self.data['user_business'].iteritems():
            # print "check", user, businesses
            user_idx = self.data['u_hash_to_idx'][user]
            positive_business = [ self.data['b_hash_to_idx'][val] for val in businesses]
            positive_b_rating = self.data['user_b_rating'][user]
            
            for i in range(len(positive_business)):
                data.append( [user_idx, positive_business[i], positive_b_rating[i]] )

            if count % 1000 == 0:
                print count, len(self.data['b_idx_to_hash'])

            count += 1

        random.shuffle(data)
        training_data = np.array( data[0: int(1.0 * len(data))] )
        # validation_data = np.array( data[int(1.0 * len(data)):] )

        model = self.build_model()
        train_op = tf.train.AdamOptimizer(self.options['learning_rate']).minimize(model['loss'])
        sess = tf.InteractiveSession()
        tf.initialize_all_variables().run()


        batch_size = self.options['batch_size']
        for epoch in range(self.options['num_epochs']):
            batch_no = 0
            while (batch_no+1) * batch_size < training_data.shape[0]:
                training_batch = training_data[batch_no * batch_size: (batch_no + 1) * batch_size ]
                _, loss, predictions = sess.run( [train_op, model['loss'], model['prediction']],
                    feed_dict = {
                        model['user'] : training_batch[:,0],
                        model['positive_business'] : training_batch[:,1],
                        model['rating'] : training_batch[:,2]
                    }
                    )

                if batch_no % 100 == 0:
                    print batch_no, epoch, loss
                    for i in range(10):
                        print "Actual vs Predicted", training_batch[i,2], predictions[i]
                
                batch_no += 1

            if epoch % 25 == 0:
                self.evaluate(sess, epoch)
                # if batch_no % 100 == 0:
                #     self.evaluate(sess, training_data, epoch)
                    

                

def main():
    md = Model({
        'user_embedding_size' : 32,
        'business_embedding_size' : 32,
        'learning_rate' : 0.001,
        'num_epochs' : 1000,
        'batch_size' : 1000,
        'lambda' : 0.5
        })
    md.train()

if __name__ == '__main__':
    main()