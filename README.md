# tf2
    帮助自己简单理解和应用tf2
    
# 1.tf2的几个变化   
    链接：https://www.jianshu.com/p/599c79c3a537
    1.1. API清理
       删除tf.app、tf.flags和tf.logging，tf.contrib，清理主要的 tf.*命名空间
       支持absl-py，一些API已被2.0版本等效替换，如tf.summary, tf.keras.metrics和tf.keras.optimizers。
    1.2. Eager execution
       Eager execution模式，马上就执行代码（就像Python通常那样）不再有session.run()。
       所有代码按顺序执行（在tf.function中，带有副作用的代码按写入的顺序执行），不在需要tf.control_dependencies()。
    1.3. 没有更多的全局变量
       取消了Variables机制，支持：跟踪变量！如果你失去了对tf.Variable的追踪，就会垃圾收集回收。
       跟踪变量的要求为用户创建了一些额外的工作，但是使用Keras对象（见下文），负担被最小化。
    1.4. Functions, not sessions
       # TensorFlow 1.X
       outputs = session.run(f(placeholder), feed_dict={placeholder: input})
       # TensorFlow 2.0
       outputs = f(input)

#  2. 使用TensorFlow 2.0的建议
    2.1. 将代码重构为更小的函数
       用tf.function来修饰高级计算-例如，一个训练步骤，或者模型的前向传递。
    2.2. 使用Keras层和模型来管理变量
       Keras模型和层：variables和trainable_variables属性
       对比如下：
       tf.nn.sigmoid(tf.matmul(x, W) + b)
       @tf.function
       def multilayer_perceptron(x, w0, b0, w1, b1, w2, b2 ...):
          x = dense(x, w0, b0)
       # 各图层调用
        layers = [tf.keras.layers.Dense(hidden_size, activation=tf.nn.sigmoid) for _ in range(n)]
        perceptron = tf.keras.Sequential(layers)
        # layers[3].trainable_variables => returns [w3, b3]
        # perceptron.trainable_variables => returns [w0, b0, ...]
        # 迁移学习
        trunk = tf.keras.Sequential([...])
        head1 = tf.keras.Sequential([...])
        head2 = tf.keras.Sequential([...])
        path1 = tf.keras.Sequential([trunk, head1])
        path2 = tf.keras.Sequential([trunk, head2])
        # 训练模式
        for x, y in data:
          with tf.GradientTape() as tape:
            prediction = path1(x)
            loss = loss_fn_head1(prediction, y)
          gradients = tape.gradient(loss, path1.trainable_variables)
          optimizer.apply_gradients(zip(gradients, path1.trainable_variables))
        # 保存参数
        tf.saved_model.save(trunk, output_path)

    2.3. 结合tf.data.Datesets和@tf.function
        tf.data.Datesets：从磁盘中传输训练数据，数据集可迭代（但不是迭代器）。
            @tf.function
            def train(model, dataset, optimizer):
              for x, y in dataset:
                predict=model(x)
                ...
        如果使用Keras.fit()API，就不必担心数据集迭代：
            model.compile(optimizer=optimizer, loss=loss_fn)
            model.fit(dataset)
    2.4. 利用AutoGraph和Python控制流程
        AutoGraph提供了一种将依赖于数据的控制流转换为图形模式等价的方法，如tf.cond和tf.while_loop。

        数据依赖控制流出现的一个常见位置是序列模型。tf.keras.layers.RNN封装一个RNN单元格，允许你您静态或动态展开递归。
        为了演示，您可以重新实现动态展开如下：

        class DynamicRNN(tf.keras.Model):

          def __init__(self, rnn_cell):
            super(DynamicRNN, self).__init__(self)
            self.cell = rnn_cell

          def call(self, input_data):
            # [batch, time, features] -> [time, batch, features]
            input_data = tf.transpose(input_data, [1, 0, 2])
            outputs = tf.TensorArray(tf.float32, input_data.shape[0])
            state = self.cell.zero_state(input_data.shape[1], dtype=tf.float32)
            for i in tf.range(input_data.shape[0]):
              output, state = self.cell(input_data[i], state)
              outputs = outputs.write(i, output)
            return tf.transpose(outputs.stack(), [1, 0, 2]), state
       
2.5. 使用tf.metrics聚合数据和tf.summary来记录它
要记录摘要，请使用tf.summary.(scalar|histogram|...) 并使用上下文管理器将其重定向到writer。（如果省略上下文管理器，则不会发生任何事情。）与TF 1.x不同，摘要直接发送给writer；没有单独的merger操作，也没有单独的add_summary()调用，这意味着必须在调用点提供步骤值。

summary_writer = tf.summary.create_file_writer('/tmp/summaries')
with summary_writer.as_default():
  tf.summary.scalar('loss', 0.1, step=42)
要在将数据记录为摘要之前聚合数据，请使用tf.metrics，Metrics是有状态的；
当你调用.result()时，它们会累计值并返回累计结果。使用.reset_states()清除累计值。

def train(model, optimizer, dataset, log_freq=10):
  avg_loss = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
  for images, labels in dataset:
    loss = train_step(model, optimizer, images, labels)
    avg_loss.update_state(loss)
    if tf.equal(optimizer.iterations % log_freq, 0):
      tf.summary.scalar('loss', avg_loss.result(), step=optimizer.iterations)
      avg_loss.reset_states()

def test(model, test_x, test_y, step_num):
  loss = loss_fn(model(test_x), test_y)
  tf.summary.scalar('loss', loss, step=step_num)

train_summary_writer = tf.summary.create_file_writer('/tmp/summaries/train')
test_summary_writer = tf.summary.create_file_writer('/tmp/summaries/test')

with train_summary_writer.as_default():
  train(model, optimizer, dataset)

with test_summary_writer.as_default():
  test(model, test_x, test_y, optimizer.iterations)
通过将TensorBoard指向摘要日志目录来显示生成的摘要：


tensorboard --logdir /tmp/summaries
