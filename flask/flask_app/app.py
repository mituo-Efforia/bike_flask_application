#flaskアプリ構築用ライブラリ
#python app.pyで動かせる
from flask import Flask, render_template , request ,flash
from wtforms import Form, TextAreaField,  FloatField, IntegerField,SubmitField, validators, ValidationError, SelectField 
#機械学習に必要なライブラリ
import numpy as np
import pickle
import os
from sklearn.metrics import r2_score
import pandas as pd
from datetime import datetime

app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'qrjirjijrigjijgiajiajgoiaoijgoi'

#学習済みモデルの読み込み
def predict(feature):
    model = pickle.load(open(os.path.join('model/regressor.pkl'),'rb'))
    pred=model.predict(feature)
    return pred
oe = pickle.load(open(os.path.join('model/ordinalencoder.pkl'),'rb'))

# FloatFieldのままだと0を入力しつつ数値以外の入力があった場合のメッセージを編集できないため、サブクラス化
class MyFloatField(FloatField):
    def process_formdata(self, valuelist):
        if valuelist:
            try:
                self.data = float(valuelist[0].replace(',', '.'))
            except ValueError:
                self.data = None
                raise ValueError(self.gettext('※数値で入力してください'))

class FeatureNameForm(Form):
    modelyear = MyFloatField('モデル年(年)',
                           [validators.InputRequired("この項目は入力必須です。"),
                            validators.NumberRange(min=1894,max=float(datetime.now().year),
                                                   message="※1894年~現在の年までで入力してください")
                            ],
                            default= datetime.now().year)
    mileage = MyFloatField('走行距離(km)',
                         [validators.InputRequired("この項目は入力必須です。"),
                          validators.NumberRange(min=0,message='正値で入力してください')],
                         default=0)
    guarantee_period = MyFloatField('保証年(年)',
                                  [validators.InputRequired("この項目は入力必須です。"),
                                   validators.NumberRange(min=0,message='正値で入力してください')],
                                  default=0)
    displacement = MyFloatField('排気量(cc)',
                              [validators.InputRequired("この項目は入力必須です。"),
                               validators.NumberRange(min=50,message='50cc以上の値を入力してください')],
                               default=250) 
    color = SelectField('色を選択してください',
                        choices=[(i,oe.categories_[0][i])for i,oe.categories_[0][i] in enumerate(oe.categories_[0])],
                        default = 6)
    b_type = SelectField('バイクタイプを選択してください',
                        choices=[(i,oe.categories_[2][i])for i,oe.categories_[2][i] in enumerate(oe.categories_[2])],
                        default=4)
    brand = SelectField('メーカーを選択してください',
                         choices=[(i,oe.categories_[1][i])for i,oe.categories_[1][i] in enumerate(oe.categories_[1])],
                         default = 2)
    

    #html側で表示する予想ボタンの表示
    submit = SubmitField('価格予想')



@app.route('/',methods =['GET','POST'])
def predicts():
    form = FeatureNameForm(request.form)
    if request.method == 'POST':
        if form.validate() == False:
            flash('全て入力する必要があります。')
            return render_template('index.html', form=form)
        else :
            modelyear = float(request.form['modelyear'])
            mileage = float(request.form['mileage'])
            guarantee_period = float(request.form['guarantee_period'])
            displacement = float(request.form['displacement'])
            color= request.form['color']
            brand = request.form['brand']
            b_type = request.form['b_type']
            X = np.array([[modelyear,color,mileage,guarantee_period,brand,b_type,displacement]])
            pred = predict(X)
            pred_man = int(pred) // 10000
            pred_yen = int(pred) % 10000
            categorycaldata = oe.inverse_transform([[float(color),float(brand),float(b_type)]])
            color_name = categorycaldata[0][0]
            brand_name = categorycaldata[0][1]
            b_type_name =categorycaldata[0][2]

            return render_template('result.html',
                                   pred_man = pred_man,
                                   pred_yen=pred_yen,
                                   modelyear = int(modelyear),
                                   mileage = mileage,
                                   guarantee_period = guarantee_period,
                                   displacement = displacement,
                                   color_name = color_name,
                                   brand_name = brand_name,
                                   b_type_name = b_type_name)
    elif request.method == 'GET':

        return render_template('index.html',form = form)

if __name__ == "__main__" :
    app.run(#debug = True, 
            port =5050)

#localで自動でスクレイピングしてそれを加工するのを組んでみる