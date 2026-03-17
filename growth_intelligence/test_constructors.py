"""Test which constructors make blocking socket.connect calls."""
import socket
import os

orig_connect = socket.socket.connect
calls = []

def trace_connect(self, address):
    import traceback
    calls.append(''.join(traceback.format_stack()))
    return orig_connect(self, address)

socket.socket.connect = trace_connect

# Test Pinecone constructor  
os.environ.setdefault('PINECONE_API_KEY', 'test')
try:
    from pinecone import Pinecone
    pc = Pinecone(api_key='test-key')
except Exception:
    pass

if calls:
    print('Pinecone() constructor makes socket calls!')
else:
    print('Pinecone() constructor: NO socket calls')
calls.clear()

# Test ChatGoogleGenerativeAI constructor
os.environ.setdefault('GEMINI_API_KEY', 'test')
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    g = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', google_api_key='test')
except Exception:
    pass

if calls:
    print('ChatGoogleGenerativeAI() constructor makes socket calls!')
else:
    print('ChatGoogleGenerativeAI() constructor: NO socket calls')
calls.clear()

# Test OpenAIEmbeddings constructor
os.environ.setdefault('OPENAI_API_KEY', 'test')
try:
    from langchain_openai import OpenAIEmbeddings
    e = OpenAIEmbeddings(model='text-embedding-3-large', openai_api_key='test')
except Exception:
    pass

if calls:
    print('OpenAIEmbeddings() constructor makes socket calls!')
else:
    print('OpenAIEmbeddings() constructor: NO socket calls')
calls.clear()

# Test anthropic.Anthropic constructor
try:
    import anthropic
    a = anthropic.Anthropic(api_key='test')
except Exception:
    pass

if calls:
    print('anthropic.Anthropic() constructor makes socket calls!')
else:
    print('anthropic.Anthropic() constructor: NO socket calls')
