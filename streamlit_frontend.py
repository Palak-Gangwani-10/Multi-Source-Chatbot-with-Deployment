import streamlit as st
import requests
import json
import sys
import subprocess
import time
import os
 

API_BASE_URL = os.environ.get('API_BASE_URL', 'http://127.0.0.1:8000')

def check_backend(timeout: float = 3.0) -> bool:
    try:
        r = requests.get(f'{API_BASE_URL}/health', timeout=timeout)
        if r.status_code == 200:
            j = r.json()
            return bool(j.get('index_ready'))
        return False
    except Exception:
        return False

def start_backend() -> bool:
    py = sys.executable
    try:
        subprocess.Popen([py, '-m', 'uvicorn', 'api:app', '--port', '8000'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        for _ in range(16):
            if check_backend(timeout=1.5):
                return True
            time.sleep(0.5)
        return False
    except Exception:
        return False

def ensure_backend() -> bool:
    if check_backend(timeout=2.0):
        return True
    ok = start_backend()
    return ok

def upload_selected_file():
    if not ensure_backend():
        st.error('Backend API is offline. Please try again in a few seconds.')
        return
    f = st.session_state.get('upload_file')
    if not f:
        return
    name = f.name
    ext = ('.' + name.split('.')[-1].lower()) if '.' in name else ''
    mime = 'application/octet-stream'
    if ext == '.pdf':
        mime = 'application/pdf'
    elif ext == '.txt':
        mime = 'text/plain'
    elif ext == '.md':
        mime = 'text/markdown'
    elif ext == '.docx':
        mime = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    payload = None
    try:
        payload = f.getvalue()
    except Exception:
        try:
            payload = f.read()
        except Exception:
            payload = None
    if payload is None or payload == b'':
        try:
            payload = bytes(f.getbuffer())
        except Exception:
            payload = b''
    if not payload:
        st.error('Selected file is empty or could not be read')
        return
    try:
        res = requests.post(
            f'{API_BASE_URL}/ingest/doc',
            files={'file': (name, payload, mime)},
            data={'name': name},
            timeout=90,
        )
        if res.status_code == 200:
            st.session_state['uploaded_doc_names'].add(name)
            st.success(f"Uploaded {name}")
        else:
            st.error(f"{name}: upload failed: {res.status_code} {res.text}")
    except Exception as e:
        st.error(str(e))

def ingest_url_input():
    if not ensure_backend():
        st.error('Backend API is offline. Please try again in a few seconds.')
        return
    url = (st.session_state.get('url_input') or '').strip()
    st.session_state['show_url'] = True
    if not url:
        st.error('Please paste a valid URL')
        return
    try:
        res = requests.post(
            f'{API_BASE_URL}/ingest/url',
            json={'url': url},
            timeout=60,
        )
        if res.status_code == 200:
            j = res.json()
            st.success("URL ingested")
        else:
            st.error(f'URL ingest failed: {res.status_code} {res.text}')
    except Exception as e:
        st.error(str(e))

def ingest_json_text():
    if not ensure_backend():
        st.error('Backend API is offline. Please try again in a few seconds.')
        return
    st.session_state['show_json'] = True
    s = st.session_state.get('json_text')
    if not s or not str(s).strip():
        st.error('Please paste JSON')
        return
    try:
        payload = json.loads(s)
    except Exception as e:
        st.error(str(e))
        return
    try:
        res = requests.post(f'{API_BASE_URL}/ingest/json', json=payload, timeout=60)
        if res.status_code == 200:
            st.success('JSON ingested')
        else:
            st.error(f'JSON ingest failed: {res.status_code} {res.text}')
    except Exception as e:
        st.error(str(e))

def ingest_json_file():
    if not ensure_backend():
        st.error('Backend API is offline. Please try again in a few seconds.')
        return
    st.session_state['show_json'] = True
    jf = st.session_state.get('json_file')
    if jf is None:
        st.error('Please select a JSON file')
        return
    try:
        data = None
        try:
            data = jf.getvalue()
        except Exception:
            data = jf.read()
        if data is None:
            data = b''
        if not data:
            try:
                data = bytes(jf.getbuffer())
            except Exception:
                data = b''
        payload = json.loads(data.decode('utf-8'))
    except Exception as e:
        st.error(str(e))
        return
    try:
        res = requests.post(f'{API_BASE_URL}/ingest/json', json=payload, timeout=60)
        if res.status_code == 200:
            st.success('JSON ingested')
        else:
            st.error(f'JSON ingest failed: {res.status_code} {res.text}')
    except Exception as e:
        st.error(str(e))

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []
if 'show_upload' not in st.session_state:
    st.session_state['show_upload'] = False
if 'show_url' not in st.session_state:
    st.session_state['show_url'] = False
if 'show_json' not in st.session_state:
    st.session_state['show_json'] = False
if 'uploaded_doc_names' not in st.session_state:
    st.session_state['uploaded_doc_names'] = set()
 

st.markdown(
    """
    <style>
    #icons-row-anchor + div[data-testid="stHorizontalBlock"] { position: fixed; left: 24px; bottom: 20px; z-index: 1000; }
    #icons-row-anchor + div[data-testid="stHorizontalBlock"] .stButton>button { padding: 8px 12px; border-radius: 10px; margin-right: 8px; }
    section.main { padding-bottom: 140px; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    ok = check_backend(timeout=1.5)
    st.markdown(f"**API**: {'Online' if ok else 'Offline'} ({API_BASE_URL})")
    if not ok:
        if st.button('Start Backend'):
            if start_backend():
                st.success('Backend started')
            else:
                st.error('Failed to start backend')

for m in st.session_state['message_history']:
    with st.chat_message(m['role']):
        st.markdown(m['content'])

if st.session_state['show_upload']:
    f = st.file_uploader('Upload Document (PDF/TXT/MD/DOCX)', type=['pdf', 'txt', 'md', 'docx'], accept_multiple_files=False, key='upload_file', on_change=upload_selected_file)
    if f is not None:
        st.session_state['show_upload'] = True

if st.session_state['show_url']:
    url = st.text_input('Paste a website URL', key='url_input', on_change=ingest_url_input)

if st.session_state['show_json']:
    json_text = st.text_area('Structured Records (JSON)', key='json_text', on_change=ingest_json_text)
    json_file = st.file_uploader('Upload JSON', type=['json'], accept_multiple_files=False, key='json_file', on_change=ingest_json_file)
    if st.button('Ingest JSON', key='btn_ingest_json'):
        payload = None
        if json_file is not None:
            try:
                data = None
                try:
                    data = json_file.getvalue()
                except Exception:
                    data = json_file.read()
                if data is None:
                    data = b''
                if not data:
                    try:
                        data = bytes(json_file.getbuffer())
                    except Exception:
                        data = b''
                payload = json.loads(data.decode('utf-8'))
            except Exception as e:
                st.error(str(e))
        elif json_text:
            try:
                payload = json.loads(json_text)
            except Exception as e:
                st.error(str(e))
        if payload:
            try:
                res = requests.post(f'{API_BASE_URL}/ingest/json', json=payload, timeout=30)
                if res.status_code == 200:
                    st.success('JSON ingested')
                else:
                    st.error(f'JSON ingest failed: {res.status_code} {res.text}')
            except Exception as e:
                st.error(str(e))

query_text = st.chat_input('Ask a question...')
if query_text:
    st.session_state['message_history'].append({'role': 'user', 'content': query_text})
    with st.chat_message('user'):
        st.markdown(query_text)
    try:
        with st.spinner('Generating answer...'):
            if not ensure_backend():
                raise RuntimeError('Backend API is offline')
            body = {'question': query_text, 'top_k': 4, 'mode': 'strict'}
            r = requests.post(f'{API_BASE_URL}/query', json=body, timeout=60)
        j = r.json()
        answer = j.get('answer', '')
    except Exception as e:
        answer = str(e)
    st.session_state['message_history'].append({'role': 'assistant', 'content': answer})
    with st.chat_message('assistant'):
        st.markdown(answer)

st.markdown("<div id='icons-row-anchor'></div>", unsafe_allow_html=True)
icons = st.columns([0.08, 0.08, 0.08])
with icons[0]:
    if st.button('📎', help='Upload files', key='btn_files'):
        st.session_state['show_upload'] = not st.session_state['show_upload']
with icons[1]:
    if st.button('🔗', help='Add URL', key='btn_url'):
        st.session_state['show_url'] = not st.session_state['show_url']
with icons[2]:
    if st.button('🗄️', help='Add structured data', key='btn_json'):
        st.session_state['show_json'] = not st.session_state['show_json']

# dynamic active styles for icons
active_css = [
    "<style>",
]
if st.session_state.get('show_upload'):
    active_css.append("#icons-row-anchor + div[data-testid='stHorizontalBlock'] > div:nth-child(1) .stButton>button{background-color:#E8F0FF;border:1px solid #BFD3FF;}")
if st.session_state.get('show_url'):
    active_css.append("#icons-row-anchor + div[data-testid='stHorizontalBlock'] > div:nth-child(2) .stButton>button{background-color:#E8F0FF;border:1px solid #BFD3FF;}")
if st.session_state.get('show_json'):
    active_css.append("#icons-row-anchor + div[data-testid='stHorizontalBlock'] > div:nth-child(3) .stButton>button{background-color:#E8F0FF;border:1px solid #BFD3FF;}")
active_css.append("</style>")
st.markdown("".join(active_css), unsafe_allow_html=True)
