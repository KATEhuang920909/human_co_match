import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from .graph import Graph
import tensorflow as tf
from ..utils.load_data import load_data, data2vec, bert_process
import args
import pandas as pd

train_data, test_data = load_data(r"D:\learning\competition\人岗匹配\human_co_match\human_co_match\data\text_data\\")
data = pd.concat([train_data, test_data])
person_list = data2vec(data, "求职者文本内容").apply(lambda x: bert_process(x, args.person_len))
intent_list = data2vec(data, "投递意向文本内容").apply(lambda x: bert_process(x, args.intent_len))
work_exp_list = data2vec(data, "工作经历文本内容").apply(lambda x: bert_process(x, args.work_exp_len))
cert_list = data2vec(data, "证书文本内容").apply(lambda x: bert_process(x, args.cert_len))
project_exp_list = data2vec(data, "项目经历文本内容").apply(lambda x: bert_process(x, args.project_exp_len))
job_desc_list = data2vec(data, "岗位文本内容").apply(lambda x: bert_process(x, args.job_desc_len))
y_train = train_data["标签"].values
person_list_train, person_list_eval = person_list[:train_data.shape[0]], person_list[train_data.shape[0]:]
intent_list_train, intent_list_eval = intent_list[:train_data.shape[0]], intent_list[train_data.shape[0]:]
work_exp_list_train, work_exp_list_eval = work_exp_list[:train_data.shape[0]], work_exp_list[train_data.shape[0]:]
cert_list_train, cert_list_eval = cert_list[:train_data.shape[0]], cert_list[train_data.shape[0]:]
project_exp_list_train, project_exp_list_eval = project_exp_list[:train_data.shape[0]], project_exp_list[
                                                                                        train_data.shape[0]:]
job_desc_list_train, job_desc_list_eval = job_desc_list[:train_data.shape[0]], job_desc_list[train_data.shape[0]:]
y_train, y_eval = y_train[:train_data.shape[0]], y_train[train_data.shape[0]:]

person_holder = tf.placeholder(dtype=tf.int32, shape=(None, args.person_len, args.bert_size), name='person')
intent_holder = tf.placeholder(dtype=tf.int32, shape=(None, args.intent_len, args.bert_size), name='intent')
work_exp_holder = tf.placeholder(dtype=tf.int32, shape=(None, args.work_exp_len, args.bert_size), name='work_exp')
cert_holder = tf.placeholder(dtype=tf.int32, shape=(None, args.cert_len, args.bert_size), name='cert')
project_exp_holder = tf.placeholder(dtype=tf.int32, shape=(None, args.project_exp_len, args.bert_size), name='project_exp')
job_desc_holder = tf.placeholder(dtype=tf.int32, shape=(None, args.job_desc_len, args.bert_size), name='job_desc')
y_holder = tf.placeholder(dtype=tf.int32, shape=None, name='y')

dataset = tf.data.Dataset.from_tensor_slices(
    (person_holder, intent_holder, work_exp_holder, cert_holder, project_exp_holder, job_desc_holder, y_holder))
dataset = dataset.batch(args.batch_size).repeat(args.epochs)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

model = Graph()
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
feed_dict_init = dict(
    zip([person_holder, intent_holder, work_exp_holder, cert_holder, project_exp_holder, job_desc_holder, y_holder],
        [person_list_train, intent_list_train, work_exp_list_train, cert_list_train, project_exp_list_train,
         job_desc_list_train, y_train]))
with tf.Session(config=config)as sess:
    sess.run(tf.global_variables_initializer())
sess.run(iterator.initializer, feed_dict=feed_dict_init)
steps = int(len(y_train) / args.batch_size)
for epoch in range(args.epochs):
    for step in range(steps):
        person_list_batch, intent_list_batch, work_exp_list_batch, cert_list_batch, project_exp_list_batch, job_desc_list_batch, y_batch = sess.run(
            next_element)
        _, loss, acc = sess.run([model.train_op, model.loss, model.acc],
                                feed_dict={model.person: person_list_batch,
                                           model.intent: intent_list_batch,
                                           model.work_exp: work_exp_list_batch,
                                           model.cert: cert_list_batch,
                                           model.project_exp: project_exp_list_batch,
                                           model.job_desc: job_desc_list_batch,
                                           model.y: y_batch,
                                           model.keep_prob: args.keep_prob})
        print('epoch:', epoch, ' step:', step, ' loss: ', loss, ' acc:', acc)

        loss_eval, acc_eval = sess.run([model.loss, model.acc],
                                       feed_dict={model.person: person_list_eval,
                                                  model.intent: intent_list_eval,
                                                  model.work_exp: work_exp_list_eval,
                                                  model.cert: cert_list_eval,
                                                  model.project_exp: project_exp_list_eval,
                                                  model.job_desc: job_desc_list_eval,
                                                  model.y: y_eval,
                                                  model.keep_prob: 1})
        print('loss_eval: ', loss_eval, ' acc_eval:', acc_eval)
        print('\n')
        saver.save(sess, f'../output/dssm/dssm_{epoch}.ckpt')
