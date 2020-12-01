#!/usr/bin/env python
# coding=utf-8
  
from flask import Flask
from flask import render_template
  
app = Flask(__name__)
  
"""
这是一个展示Flask如何读取服务器本地图片, 并返回图片流给前端显示的例子
"""
  
  
def return_img_stream(img_local_path):
  """
  工具函数:
  获取本地图片流
  :param img_local_path:文件单张图片的本地绝对路径
  :return: 图片流
  """
  import base64
  img_stream = ''
  with open(img_local_path, 'r') as img_f:
    img_stream = img_f.read()
    img_stream = base64.b64encode(img_stream)
  return img_stream
  
  
@app.route('/')
def hello_world():
  img_path = '../../../test_data/images/000000001551.jpg'
  img_stream = return_img_stream(img_path)
  return render_template('index.html',
              img_stream=img_stream)
  
  
if __name__ == '__main__':
  app.run(host='0.0.0.0',debug=True, port=8080)