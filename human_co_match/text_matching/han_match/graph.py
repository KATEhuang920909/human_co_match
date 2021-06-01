import tensorflow as tf
import args

"""
人岗匹配：
求职者有五类文本特征，求职者文本、求职意向文本、工作经历文本、证书文本、项目经历文本
岗位有一类文本特征：岗位要求
构建层次注意力网络进行文本匹配
"""


class Graph():
    def __init__(self, ):
        # 六类占位符
        self.person = tf.placeholder(dtype=tf.int32, shape=(None, args.person_len, args.bert_size), name='person')
        self.intent = tf.placeholder(dtype=tf.int32, shape=(None, args.intent_len, args.bert_size), name='intent')
        self.work_exp = tf.placeholder(dtype=tf.int32, shape=(None, args.work_exp_len, args.bert_size), name='work_exp')
        self.cert = tf.placeholder(dtype=tf.int32, shape=(None, args.cert_len, args.bert_size), name='cert')
        self.project_exp = tf.placeholder(dtype=tf.int32, shape=(None, args.project_exp_len, args.bert_size),
                                          name='project_exp')
        self.job_desc = tf.placeholder(dtype=tf.int32, shape=(None, args.job_desc_len, args.bert_size), name='job_desc')
        self.y = tf.placeholder(dtype=tf.int32, shape=None, name='y')

        self.keep_prob = tf.placeholder(dtype=tf.float32, name='drop_rate')
        # self.embedding = tf.get_variable(name=embedding_name, initializer=EMBEDDING_DATA, dtype=tf.float32,
        #                                  trainable=False)
        # self.embedding = tf.Variable(tf.cast(EMBEDDING_DATA, dtype=tf.float32, name=embedding_name), name="embedding_w")
        # self.embedding = tf.get_variable(dtype=tf.float32, shape=(args.vocab_size, args.char_embedding_size),
        #                                  name='embedding')

        self.forward()

    def dropout(self, x):
        return tf.nn.dropout(x, keep_prob=self.keep_prob)

    def fully_connect_layer(self, x):
        x = tf.layers.dense(x, 256, activation='tanh')
        x = self.dropout(x)
        x = tf.layers.dense(x, 512, activation='tanh')
        x = self.dropout(x)
        x = tf.layers.dense(x, 256, activation='tanh')
        x = self.dropout(x)
        x = tf.reshape(x, shape=(-1, x.shape[1] * x.shape[2]))

        return x

    def lstm_layer(self, inputs):

        # RNN
        lstmCell = tf.nn.rnn_cell.LSTMCell(args.rnn_size)
        lstmCell = tf.nn.rnn_cell.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
        # 输出每一步
        outputs, _ = tf.nn.dynamic_rnn(lstmCell, inputs, dtype=tf.float32)
        # print('final_output_shape:', outputs.shape)
        # output = tf.reshape(final_output, [-1, rnn_size])
        return outputs

    def attention_layer(self, inputs):  # char_vec->sentence_vec
        # assert (char2sent or sentls2sent) and not (char2sent and sentls2sent), 'please set attention type'  # 异或

        seq_length = tf.shape(inputs)[0]
        attention_w = tf.Variable(tf.truncated_normal([args.rnn_size, args.attention_size], stddev=0.1),
                                  name='attention_w')
        attention_b = tf.Variable(tf.constant(0.1, shape=[args.attention_size]), name='attention_b')
        u_list = []
        for t in range(seq_length):
            u_t = tf.tanh(tf.matmul(inputs[t], attention_w) + attention_b)
            u_list.append(u_t)
        u_w = tf.Variable(tf.truncated_normal([args.attention_size, 1], stddev=0.1), name='attention_uw')
        attn_z = []
        for t in range(seq_length):
            z_t = tf.matmul(u_list[t], u_w)
            attn_z.append(z_t)
        # transform to batch_size * sequence_length
        attn_zconcat = tf.concat(attn_z, axis=1)
        alpha = tf.nn.softmax(attn_zconcat)
        # transform to sequence_length * batch_size * 1 , same rank as outputs
        alpha_trans = tf.reshape(tf.transpose(alpha, [1, 0]), [seq_length, -1, 1])
        final_output = tf.reduce_sum(inputs * alpha_trans, 0)
        return final_output

    @staticmethod
    def cosine(p, h):
        p_norm = tf.norm(p, axis=1, keepdims=True)
        h_norm = tf.norm(h, axis=1, keepdims=True)

        cosine = tf.reduce_sum(tf.multiply(p, h), axis=1, keepdims=True) / (p_norm * h_norm)

        return cosine

    def forward(self):
        """
        @todo: embedding 这里np.array转化为tensor格式，不用embedding_looking up
        @return:
        """
        # embedding
        # person_embedding = tf.nn.embedding_lookup(self.embedding, self.person)  # person为idx，lookingup idx的向量
        # intent_embedding = tf.nn.embedding_lookup(self.embedding, self.intent)
        # work_exp_embedding = tf.nn.embedding_lookup(self.embedding, self.work_exp)
        # cert_embedding = tf.nn.embedding_lookup(self.embedding, self.cert)
        # project_exp_embedding = tf.nn.embedding_lookup(self.embedding, self.project_exp)
        # job_desc_embedding = tf.nn.embedding_lookup(self.embedding, self.job_desc)
        # lstm
        person_char_outputs = self.lstm_layer(self.person)
        intent_char_outputs = self.lstm_layer(self.intent)
        work_exp_char_outputs = self.lstm_layer(self.work_exp)
        cert_char_outputs = self.lstm_layer(self.cert)
        project_exp_char_outputs = self.lstm_layer(self.project_exp)
        job_desc_char_outputs = self.lstm_layer(self.job_desc)

        # attention1 output shape:[sent_vec1,sent_vec2..,sent_vec5];[sent_vec6]
        person_sent_outputs = self.attention_layer(person_char_outputs)
        intent_sent_outputs = self.attention_layer(intent_char_outputs)
        work_exp_sent_outputs = self.attention_layer(work_exp_char_outputs)
        cert_sent_outputs = self.attention_layer(cert_char_outputs)
        project_exp_sent_outputs = self.attention_layer(project_exp_char_outputs)
        job_desc_sent_outputs = self.attention_layer(job_desc_char_outputs)
        # [0,1],[1,0]  [0,0,1]...

        # attention2 output shape :[sentence_vec1],[sentence_vec2]
        outputs = tf.concat([person_sent_outputs, intent_sent_outputs, work_exp_sent_outputs, cert_sent_outputs,
                             project_exp_sent_outputs], axis=0)

        person_info_outputs = self.attention_layer(outputs)

        pos_result = self.cosine(person_info_outputs, job_desc_sent_outputs)
        neg_result = 1 - pos_result

        logits = tf.concat([pos_result, neg_result], axis=1)

        self.train(logits)

    def train(self, logits):
        y = tf.one_hot(self.y, args.class_size)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
        self.loss = tf.reduce_mean(loss)
        self.train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
        prediction = tf.argmax(logits, axis=1)
        correct_prediction = tf.equal(tf.cast(prediction, tf.int32), self.y)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
