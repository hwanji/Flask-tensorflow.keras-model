# -*- coding: utf-8 -*-
"""
如何用Flask部署Keras深度学习模型
https://zhuanlan.zhihu.com/p/47349497


预测示例

浏览器中输入要预测的数据，特征数量要与训练时的特征数量相同
http://127.0.0.1:5000/predict?g1=1&g2=0&g3=0&g4=0&g5=0&g6=0&g7=0&g8=0&g9=0&g10=0

"""

import flask
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np

host='127.0.0.1'

#实例化 flask 
app = flask.Flask(__name__)

# 加载模型，传入自定义度量函数
global graph

model = load_model('games.h5')

# 将预测函数定义为一个端点
@app.route("/predict", methods=["GET","POST"])
def predict():
    
    data = {"success": False}
    params = flask.request.json  
    
    if (params == None):
        params = flask.request.args

    # 若发现参数，则返回预测值
    if (params != None):
        print(f"params {'='*50} \n",params)
        x=pd.DataFrame.from_dict(params, orient='index').transpose()
        print(f"x {'='*50} \n",x)
        y_predict = model.predict(x)  #预测结果是一个二元列表
        y_predict = np.argmax(y_predict[0]) #取概率最大的一个id
        print(f"y_predict {'='*50} \n",y_predict)
        data["prediction"] = str(y_predict)
        data["success"] = True

    # 返回Json格式的响应
    return flask.jsonify(data)    

# 启动Flask应用程序，允许远程连接
if __name__ == '__main__':
    app.run(debug=False,host=host)



