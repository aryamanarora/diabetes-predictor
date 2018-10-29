from flask import Flask, request, json, Response, redirect, url_for, render_template

app = Flask(__name__)

@app.route('/')
def main():
    return render_template("index.html")