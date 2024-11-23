import os
import uuid
import pyttsx3
import torch
import whisper
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Carregamento do modelo GPT-Neo
tokenizador = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
modelo = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

# Carregamento do modelo Whisper
modelo_whisper = whisper.load_model("base")

# Inicialização do sintetizador de fala
motor_fala = pyttsx3.init()
motor_fala.setProperty('rate', 180)  # Define a velocidade da fala

# Listando e selecionando uma voz masculina
voices = motor_fala.getProperty('voices')
for voice in voices:
    if "male" in voice.name.lower() or "homem" in voice.name.lower():
        motor_fala.setProperty('voice', voice.id)
        break
else:
    # Se nenhuma voz masculina for encontrada, você pode escolher uma voz padrão.
    motor_fala.setProperty('voice', voices[0].id)  # Usar a primeira voz disponível

# Caminho para salvar arquivos de áudio
caminho = os.path.join(os.getcwd(), "audio")
os.makedirs(caminho, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chape', methods=['POST'])
def chape():
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({"error": "Nenhuma pergunta recebida."}), 400

    # Gerar resposta a partir da pergunta
    resposta = gerar_resposta(question)

    # Falar a resposta
    falar(resposta)

    return jsonify({"answer": resposta})

@app.route('/api/recognize', methods=['POST'])
def recognize_audio():
    """ Rota para reconhecer áudio enviado pelo frontend. """
    if 'audio' not in request.files:
        return jsonify({"error": "Nenhum áudio enviado."}), 400

    audio_data = request.files['audio']
    if audio_data.filename == '':
        return jsonify({"error": "Arquivo de áudio vazio."}), 400

    # Gerar um nome de arquivo único
    nome_arquivo = f"audio_{uuid.uuid4().hex}.wav"
    audio_path = os.path.join(caminho, nome_arquivo)
    audio_data.save(audio_path)

    # Reconhecimento de fala usando Whisper
    try:
        texto_transcrito = modelo_whisper.transcribe(audio_path, language='pt', fp16=False)
        pergunta = texto_transcrito["text"].strip()
    except Exception as e:
        os.remove(audio_path)
        return jsonify({"error": f"Falha na transcrição: {str(e)}"}), 500

    if not pergunta:
        os.remove(audio_path)
        return jsonify({"error": "Não foi possível transcrever o áudio."}), 400

    # Processa a pergunta
    response = chape_internal(pergunta)

    # Limpar o arquivo de áudio após o processamento
    os.remove(audio_path)

    return response

def chape_internal(question):
    resposta = gerar_resposta(question)

    # Falar a resposta
    falar(resposta)

    return jsonify({"answer": resposta})


def gerar_resposta(question):
    # Aqui você pode definir a resposta fixa que deseja retornar
    resposta_fixa = " "

    # Impressão para visualizar no terminal
    print(f"Entrada para o modelo: {question.strip()}")
    print(f"Texto gerado pelo modelo: {resposta_fixa}")

    return resposta_fixa


def falar(texto):
    try:
        motor_fala.say(texto)
        motor_fala.runAndWait()
    except Exception as e:
        print(f"Erro na síntese de fala: {e}")

if __name__ == '__main__':
    app.run(debug=True)

