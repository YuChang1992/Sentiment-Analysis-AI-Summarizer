from flask import Flask, render_template, request, redirect, url_for, send_from_directory, abort
from werkzeug.utils import secure_filename
import os
from openai import AzureOpenAI
from matplotlib import pyplot as plt
from keras.models import load_model
import numpy as np
#from skimage import io
#from skimage.transform import resize
#from skimage import util 
#新文章預測寫成函數
from keras.utils.data_utils import pad_sequences
from keras.models import load_model
import pickle, numpy as np, nltk, re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
from nltk import word_tokenize
import json

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "static/"

@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/hello', methods=['POST'])
def hello():
   name = request.form.get('name')

   if name:
       print('Request for hello page received with name=%s' % name)
       return render_template('hello.html', name = name)
   else:
       print('Request for hello page received with no name or blank name -- redirecting')
       return redirect(url_for('index'))

def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    stop = stopwords.words('english')
    wnl = WordNetLemmatizer()
    text = re_tag.sub('', text)
    text = text.replace('.',' ')
    text = text.replace("'",' ')
    text = text.replace('"',' ')
    text = text.replace(',',' ')
    text = text.replace(':',' ')
    text = text.replace('!',' ')
    text =' '.join([wnl.lemmatize(w) for w in word_tokenize(text.lower()) \
                    if w not in stop and w not in punctuation])
    return text

def predict_review(input_text):

    model2 = load_model('./weights.hdf5')  # 載入模型

    with open('./tokenizer.pickle', 'rb') as handle:
        token2 = pickle.load(handle)
    SentimentDict={1:'正面的', 0:'負面的'}
    input_seq = token2.texts_to_sequences([rm_tags(input_text)])
    pad_input_seq  = pad_sequences(input_seq , maxlen=380)
    predict_result = np.where(model2.predict(np.array(pad_input_seq)) >=0.5, 1, 0)
    return(SentimentDict[predict_result[0][0]])

def openai_review(input_text):
    os.environ["AZURE_OPENAI_API_KEY"] = ""  # 你的金鑰

    client = AzureOpenAI(
    						api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    						api_version="",  # 你的版本 2023-05-15
    						azure_endpoint=""  # 你的連結
    					)

    deployment_name = "gpt-35-turbo"

    completion = client.chat.completions.create(
    			model=deployment_name,
    			messages = [{"role":"system","content":"You are an AI assistant that helps people find information."},
    						{"role":"user","content":"使用中文簡單說明以下的劇情 '''" + input_text + "'''"}],
    			max_tokens=800,
    			temperature=0.7,
    			top_p=0.95,
    			frequency_penalty=0,
    			presence_penalty=0,
    			stop=None,
    			stream=False
    			)

    summary = json.loads(completion.to_json())['choices'][0]['message']['content']
    return(summary)
    
@app.route('/keras', methods=['POST'])
def keras():
   imdb = request.form.get('imdb')
   openai = request.form.get('openai')
   print('openai', openai)
   if imdb:
       email_option = request.form.get('email')
       result = predict_review(imdb)
       icon = '/static/images/smile.jpg' if result == '正面的' else  '/static/images/cheers.jpg'
       if openai:           
            openai_result = openai_review(imdb)+'...'
            print('not openai')
       else:
            openai_result = '尚未選擇OpenAI進行分析...'
            #print('not openai')
            
       if email_option:  # 如果使用者有輸入 email
            from email.message import EmailMessage
            from smtplib import SMTP
            # 組合內文文字
            reply = "This is system message by AI model. \n" + \
                    "The analysis result is [" + result + ']\n' + '-'*50  + " The summary is " + '\n'+ \
                    openai_result + '\n' + '-'*50 + " The source is " + '\n' + imdb
 
            email = EmailMessage()
            email['Subject'] = ''  # 主旨
            email['From'] = ''  # 寄件者
            email['To'] = email_option  # 收件者
            email.set_content(reply, subtype='text')  # 信件內文

            with SMTP('localhost') as s:  # 沒有安裝本地郵件伺服器會出錯
                s.login('***', '***')  # ***為登入郵件伺服器所需的帳號與密碼
                s.send_message(email)
           
       return render_template("contentnlp.html", user_imdb = result, user_email = email_option, user_icon = icon, user_openai = openai_result)
   else:
       print('Request for hello page received with no name or blank name -- redirecting')
       return redirect(url_for('index'))
       
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug = True)