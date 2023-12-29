from flask import Flask, jsonify, render_template, request
from googletrans import Translator
from chatBotPackage.chatbotRail import Chatbot
 
app = Flask(__name__)
# Create a list to store messages
messages = []
@app.route('/')
def index():
    return render_template('fontEnd.html')

@app.route('/text_translate', methods=['GET', 'POST'])
def translate():
        if request.method == 'POST':
            trans = Translator()
            inputString = request.get_json().get('text_input')
            lang = request.get_json().get('language_select')
            translatedText = trans.translate(inputString, dest=lang).text
            return jsonify(translated_text=translatedText)

@app.route('/chat_bot', methods=['GET', 'POST'])
def chatting():
        if request.method == 'POST':
            transChat = Translator()
            inputString = request.get_json().get("text_input") # wht is the train time
            enTrans = transChat.translate(inputString, dest='en').text
            botReply = chatbotObj.startChat(enTrans)
            lang = request.get_json().get('language_select')
            botReplyTrans = transChat.translate(botReply, dest=lang).text
            return jsonify(chatReply=botReplyTrans)
        
if __name__=="__main__":
    global chatbotObj
    chatbotObj = Chatbot()
    app.run(debug=True)
