1. 层命名带‘()’
2. get_variable(),指定参数名称，不然容易出错
3. tf没有prod，使用数学函数应该参考数学文档
4. 在get_variable中，应该注意参数名称复用，最好加上第几层
5. variable_scope(),注意reuse
6. 注意比如AF层的矩阵乘法
7. TensorBoard可视化路径不能有中文
8. 项目目录导入问题，通过打开上级目录可以直接用文件名导入，或者直接添加到环境变量
9. 保存为TFRecord时没有预先创建dataset目录出错
10. 把Features的s漏掉了。。。。
11. 在input_data中，尝试使用self.images = tf.decode_raw(features['img'], tf.float32)
    - 它的输出一个比输入字节多一个维度的张量。添加的维度的大小将等于字节的元素的长度除以要表示 out_type 的字节数
    - 现先用string_to_number代替
    - 各种错误，重新使用decode_raw，加入reshape，取代在tf.FixedLenFeature()中直接加入shape，
    - 因为原来格式是int8，但是一直当成uint8使用，导致出错，总的来说为保存成TFRecord把图片编码成bytes格式，decode_raw()可以转码，但是要求格式匹配