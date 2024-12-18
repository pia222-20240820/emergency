from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import requests
import folium
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tiktoken
from django.views.decorators.csrf import csrf_exempt
import json
from django.conf import settings

# Naver API credentials
naver_client_id = settings.NAVER_CLIENT_ID
naver_client_secret = settings.NAVER_CLIENT_SECRET

# OpenAI API Key
api_key = settings.OPENAI_API_KEY

# Helper function to search location using Naver API
def search_location(query):
    url = "https://openapi.naver.com/v1/search/local.json"
    headers = {
        "X-Naver-Client-Id": naver_client_id,
        "X-Naver-Client-Secret": naver_client_secret,
    }
    params = {"query": query, "display": 5}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return {"items": []}

# Helper function to clean HTML tags
def clean_html(raw_html):
    import re
    cleanr = re.compile('<.*?>')
    return re.sub(cleanr, '', raw_html)

# Main view for the index page
def index(request):
    if 'chat_history' not in request.session:
        request.session['chat_history'] = []

    location_data = {'items': []}
    map_html_paths = []

    if request.method == 'POST':
        if 'location_query' in request.POST:
            location_query = request.POST.get('location_query', '')
            location_data = search_location(location_query)
            map_html_paths = generate_folium_map(location_data)
            print(f"Location data: {location_data}")

        elif 'query' in request.POST:
            query = request.POST.get('query', '')
            response = None

            if query:
                try:
                    print(f"Query type: {type(query)}, Query content: {query}")

                    tokenizer = tiktoken.get_encoding('cl100k_base')
                    encoded_query = tokenizer.encode(query)
                    
                    chainer = EmergencyRAGChainer(db_path='./db/chromadb_1')
                    chain = chainer.create_rag_chain()
                    
                    response = chain.invoke(query)
                except Exception as e:
                    return JsonResponse({'error': str(e)})

            request.session['chat_history'].append({'user': query, 'ai': response})
            request.session.modified = True

            return JsonResponse({
                'user': query,
                'ai': response,
            })

    return render(request, 'index.html', {
        'location_data': location_data,
        'map_html_paths': map_html_paths,
    })

# Folium map generation
def generate_folium_map(location_data):
    map_html_paths = []
    for index, item in enumerate(location_data["items"]):
        mapx = float(item["mapx"]) / 10000000
        mapy = float(item["mapy"]) / 10000000
        m = folium.Map(location=[mapy, mapx], zoom_start=15)
        folium.Marker([mapy, mapx], popup=item["title"]).add_to(m)
        
        # Ensure the directory exists
        import os
        os.makedirs('static/maps', exist_ok=True)
        
        map_html_path = f'static/maps/location_map_{index}.html'
        m.save(map_html_path)
        
        # Debugging statement
        print(f"Map saved to {map_html_path}")
        
        map_html_paths.append(map_html_path)
        
    return map_html_paths

class EmergencyRAGChainer:
    def __init__(self, db_path):
        self.db_path = db_path
        self.embedding_model = OpenAIEmbeddings(api_key=api_key)
        template = '''Answer the question in korean, based only on the following context:

        context :
        {context}

        Question: {question}
        '''
        self.prompt = ChatPromptTemplate.from_template(template)
        self.model = ChatOpenAI(
            # model='gpt-4o-mini',
            model='ft:gpt-4o-mini-2024-07-18:personal:fine-tune-qadataset-model:AY0P3YLq',
            temperature=0,
            max_tokens=500,
            api_key=api_key
        )

    def load_vectorstore(self):
        return Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embedding_model,
            collection_metadata={'hnsw:space': 'cosine'}
        )

    def create_retriever(self, vector_store):
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 20, "alpha": 0.5},
        )

    def format_docs(self, docs):
        return '\n\n'.join([d.page_content for d in docs])

    def create_rag_chain(self):
        vector_store = self.load_vectorstore()
        retriever = self.create_retriever(vector_store)
        return {'context': retriever | self.format_docs, 'question': RunnablePassthrough()} | self.prompt | self.model | StrOutputParser()

# @csrf_exempt
# def toggle_like(request):
#     if request.method == 'POST':
#         data = json.loads(request.body)
#         liked = data.get('liked', False)
#         # 좋아요 상태 업데이트 로직
#         return JsonResponse({'status': 'success'})

@csrf_exempt
def reset_chat(request):
    if request.method == 'POST':
        request.session['chat_history'] = []
        return JsonResponse({'status': 'success'})
