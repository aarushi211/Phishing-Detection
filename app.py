from flask import Flask, render_template, request
import pickle
import numpy as np
import tensorflow as tf
from urllib.parse import urlparse
import smtplib
import requests
import pandas as pd

model = tf.keras.models.load_model("Model")

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

def count_dashes(string):
    return string.count("-")

def pct_self_redirect(hyperlinks):
    self_redirect = 0
    for link in hyperlinks:
        if link.startswith("#"):
            self_redirect += 1
    pct_self_redirect = self_redirect / len(hyperlinks) * 100
    return pct_self_redirect

def count_dots(hyperlinks):
    return sum(map(lambda x: x.count("."), hyperlinks))

def pct_external(hyperlinks):
    external = 0
    for link in hyperlinks:
        if not link.startswith("#"):
            external += 1
    pct_external = external / len(hyperlinks) * 100
    return pct_external

def Path_Level(path):
    level = 0
    for char in path:
        if char == "/":
            level += 1
    return level

def HostNameLength(hostname):
    return len(hostname)

def NumDashInhostname(hostname):
    return hostname.count("-")

def NumqueryComponents(url):
    start = url.find("?")
    if start == -1:
        return 0
    query = url[start+1:]
    return len(query.split("&"))

def Pct_ExtNullSelfRedirectHyperlinksRT(urls):
    ext_count = 0
    null_count = 0
    self_redirect_count = 0
    hyperlink_count = 0
    total_count = len(urls)
    for url in urls:
        if url.startswith("http://") or url.startswith("https://"):
            ext_count += 1
        elif url == "#":
            null_count += 1
        elif url.startswith("/") or url.startswith("./"):
            self_redirect_count += 1
        else:
            hyperlink_count += 1
    ext_pct = (ext_count / total_count) * 100
    null_pct = (null_count / total_count) * 100
    self_redirect_pct = (self_redirect_count / total_count) * 100
    hyperlink_pct = (hyperlink_count / total_count) * 100
    return ext_pct, null_pct, self_redirect_pct, hyperlink_pct

def FrequentDomainName_Mismatch(urls):
    domains = {}
    for url in urls:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        if domain in domains:
            domains[domain] += 1
        else:
            domains[domain] = 1
    return domains

sensitive_words = ["password", "credit card", "sensitive", "email", "ssn", "social security number"]
def NumSensitiveWords(url):
    response = requests.get(url, verify = False)
    content = response.text
    count = 0
    for word in sensitive_words:
        if word in content:
            count += 1
    return count


def SubmitInfoto_Email(url):
    response = requests.get(url)
    content = response.text

    for info in sensitive_words:
        if info in content:
            return 1
    else:
        return 0

def ExtMetaScript_LinkRT(tags):
    ext_count = 0
    meta_count = 0
    script_count = 0
    link_count = 0
    total_count = len(tags)
    for tag in tags:
        if tag.startswith("<a href="):
            ext_count += 1
        elif tag == "<meta>":
            meta_count += 1
        elif tag.startswith("<script"):
            script_count += 1
        elif tag.startswith("<link"):
            link_count += 1
    ext_pct = (ext_count / total_count) * 100
    meta_pct = (meta_count / total_count) * 100
    script_pct = (script_count / total_count) * 100
    link_pct = (link_count / total_count) * 100
    return ext_pct, meta_pct, script_pct, link_pct

def target(df, shuffle=True, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((dict(df)))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds

@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        url = request.form['url']
        NumDash = count_dashes(url)
        PctNullSelfRedirectHyperlinks = pct_self_redirect(url)
        NumDots = count_dots(url)
        PctExtHyperlinks = pct_external(url)
        NumSensitiveWord = NumSensitiveWords(url)
        PathLevel = Path_Level(url)
        HostnameLength = HostNameLength(url)
        NumDashInHostname  = NumDashInhostname(url)
        NumQueryComponents = NumqueryComponents(url)
        PctExtNullSelfRedirectHyperlinksRT = Pct_ExtNullSelfRedirectHyperlinksRT(url)
        FrequentDomainNameMismatch = FrequentDomainName_Mismatch(url)
        ExtMetaScriptLinkRT = ExtMetaScript_LinkRT(url)
        SubmitInfoToEmail = SubmitInfoto_Email(url)

        data = np.array([[
            NumDash, 
            PctNullSelfRedirectHyperlinks, 
            NumDots,
            PctExtHyperlinks,
            NumSensitiveWord,
            PathLevel,
            HostnameLength,
            NumDashInHostname,
            NumQueryComponents,
            PctExtNullSelfRedirectHyperlinksRT,
            FrequentDomainNameMismatch,
            ExtMetaScriptLinkRT,
            SubmitInfoToEmail
            ]], dtype = float)

        data = np.array([[1, 0.0, 3, 1.000, 0, 6, 22, 0, 0, -1, 1, 1, 1]])

        data = pd.DataFrame(data)
        data = target(data, shuffle = False, batch_size = 32)
    
        predict = []
        my_prediction = model.predict(data)
        for pred in my_prediction:
            if pred>0:
                predict.append(1)
            else:
                predict.append(0)
 
        return render_template('result.html', predict[0])


if __name__ == '__main__':
	app.run(debug=True)
    