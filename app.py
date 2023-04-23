import nltk
from flask import Flask, request, render_template
from decouple import config
nltk.download('punkt')

import cv2
import numpy as np
import torch
from torch._C import device
import re
import nltk
import pickle
import tweepy
import requests

nltk.download('stopwords')
ar_stopwords = set(nltk.corpus.stopwords.words('arabic'))
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def preprocess_text(text):
    # Preprocess the text data
    # Normalize the text by removing diacritics and converting characters to their basic form
    text = re.sub(r'[\u064b-\u065f\u0640]', '', text)
    text = re.sub(r'[إأٱآا]', 'ا', text)
    text = re.sub(r'[ئؤ]', 'ء', text)
    # Remove non-Arabic characters and numbers
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Remove stop-words
    tokens = [token for token in nltk.word_tokenize(text) if token not in ar_stopwords]
    text = ' '.join(tokens)
    return text


def process_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_res = img_rgb / 255
    img_res = torch.as_tensor(img_res).to(device)
    img_res = img_res.permute(2, 0, 1)
    return img_res


with open('model.pkl', 'rb') as f:
    text_model = pickle.load(f)
text_vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

weapons_model_filename = "weapon_trained_model-1.0a.pt"

cpu_device = torch.device("cpu")
weapons_model = torch.load(weapons_model_filename, map_location=torch.device('cpu'))
weapons_model.eval()


def predict_image(image):
    inp = [image]
    outputs = weapons_model(inp)
    outputs_list = outputs[0]['scores'].tolist()
    if len(outputs_list) > 0:
        return max(outputs_list)
    else:
        return 0


def predict_text(text):
    new_sentence = preprocess_text(text)
    new_sentence_vectorized = text_vectorizer.transform([new_sentence])
    predicted_label = text_model.predict(new_sentence_vectorized)[0]
    return int(predicted_label)


# setup app
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_form', methods=['POST'])
def process_form():
    image_res = ""
    text_res = ""
    final = ""
    img_color = "#506e4a"
    txt_color = "#80403e"
    # Twitter API credentials
    consumer_key = config("consumer_key")
    consumer_secret = config("consumer_secret")
    access_token = config("access_token")
    access_token_secret = config("access_token_secret")

    # Connect to Twitter API
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    username = request.form.get('username')

    user = api.get_user(screen_name=username)
    tweets = api.user_timeline(screen_name=username, count=5)
    favorites = api.get_favorites(screen_name=username, count=1)
    user_profile_image_url = user.profile_image_url_https
    response = requests.get(user_profile_image_url)
    with open(f"{user.screen_name}_profile_image.jpg", "wb") as f:
        f.write(response.content)
    profile_image = cv2.imread(f"{user.screen_name}_profile_image.jpg")
    profile_image_processed = process_image(profile_image)
    profile_image_score = 1 - predict_image(profile_image_processed)

    text_score = 0
    for tweet in tweets:
        text_score += predict_text(tweet.text)
    text_score = len(tweets) - text_score

    # print("Profile image score:", profile_image_score)
    # print("Profile image score:", int(profile_image_score))
    # print("Tweets score:", text_score)

    if int(profile_image_score) <= 0:
        image_res += "تم اكتشاف صورة سلبية"
        img_color = "#80403e"

    if text_score <= 4:
        text_res += "تم اكتشاف نص سلبى"
        txt_color = "#80403e"

    if int(profile_image_score) <= 0 and text_score <= 4:
        final += "هناك احتمال أن يكون لدى الشخص دوافع جنائية"

    if int(profile_image_score) == 1:
        image_res += "لم يتم اكتشاف صورة"

    if text_score == 5:
        text_res += "لم يتم اكتشاف كلمة"

    # Do something with the name and email
    return render_template('result.html', img_color=img_color, txt_color=txt_color, image_res=image_res,
                           text_res=text_res, final_res=final)


if __name__ == "__main__":
    app.run(debug=True)
