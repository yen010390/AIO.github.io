---
title: "Module 1: Ứng dụng RAG trong việc hỏi đáp tài liệu bài học AIO"
excerpt: "Dự án này xây dựng hệ thống hỏi đáp thông minh dùng kiến trúc RAG, giúp người học khóa AI tại AI Việt Nam (AIO) khai thác hiệu quả nội dung tài liệu học tập."
collection: project
author: "Nguyễn Tuấn Anh - Đoàn Tấn Hưng - Hồ Thị Ngọc Huyền - Trần Thị Mỹ Tú - Đặng Thị Hoàng Yến"
tags:
- RAG
- LLM
- Chatbot
---

**Tác giả**: Nguyễn Tuấn Anh - Đoàn Tấn Hưng - Hồ Thị Ngọc Huyền - Trần Thị Mỹ Tú - Đặng Thị Hoàng Yến

<details>
<summary><strong>📁 Cấu trúc source code  (click để xem)</strong></summary>

- Source code được đặt tạy đây: https://github.com/aio25-mix002/m01-p0102
- Jupiter Notebooks: https://github.com/aio25-mix002/m01-p0102/blob/main/runbook_m01p0102.ipynb

<br/>

```

📦 RAG_AIO_Chatbot
├── assets/                   # Tài sản tĩnh (logo, favicon...)
│   └── logo.png              # Logo của ứng dụng
├── examples/                 # Dữ liệu mẫu để test
│   └── YOLOv10_Tutorials.pdf # File PDF mẫu
├── logs/                     # Thư mục lưu log
├── prompt_templates/         # Các template prompt cho RAG
├── utils/                    # Các tiện ích hỗ trợ
│   ├── logging_utils.py      # Utility logging
│   └── prompt_utils.py       # Utility quản lý prompt
├── .vscode/                  # Cấu hình Visual Studio Code
│   └── launch.json           # Debug configuration
├── .env                      # Biến môi trường production
├── .env.example              # Template biến môi trường
├── .env.local                # Biến môi trường local
├── rag_chatbot.py            # File chính - Streamlit RAG chatbot
├── runbook_m01p0102.ipynb    # Jupyter notebook hướng dẫn
├── requirements.txt          # Dependencies chính
├── requirements-torch.txt    # Dependencies PyTorch
├── .gitignore                # Git ignore rules
└── README.md                 # Tài liệu hướng dẫn
```
</details>

<details>
<summary><strong>📁 Mục lục báo cáo (click để xem)</strong></summary>
<br/>

- [Tóm tắt](#tóm-tắt)
- [1. Giới thiệu 🗂](#1-giới-thiệu-)
- [2. Phương pháp luận 📚](#2-phương-pháp-luận-)
  - [2.1. Quy trình Lập chỉ mục dữ liệu (Indexing)](#21-quy-trình-lập-chỉ-mục-dữ-liệu-indexing)
  - [2.2. Quy trình Truy vấn và Tạo sinh (Retrieval \& Generation)](#22-quy-trình-truy-vấn-và-tạo-sinh-retrieval--generation)
- [3. Thực hiện ⚙](#3-thực-hiện-)
- [4. Kết quả 📈](#4-kết-quả-)
- [5. Mở rộng nâng cao 🖥](#5-mở-rộng-nâng-cao-)
  - [5.1 Tiêu chí cải tiến](#51-tiêu-chí-cải-tiến)
  - [5.2 Code nâng cao](#52-code-nâng-cao)
    - [5.2.1 Nâng cấp cốt lỗi: Ghi nhớ lịch sử hội thoại (Conversation memory)](#521-nâng-cấp-cốt-lỗi-ghi-nhớ-lịch-sử-hội-thoại-conversation-memory)
    - [5.2.2 Quản lý Vector DB nâng cao](#522-quản-lý-vector-db-nâng-cao)
    - [5.2.3. Gỡ lỗi (Debugging) với Logger](#523-gỡ-lỗi-debugging-với-logger)
    - [5.2.4. Xử lý và truy vấn từ nhiều file tài liệu](#524-xử-lý-và-truy-vấn-từ-nhiều-file-tài-liệu)
  - [5.3 Kết quả mở rộng 📍](#53-kết-quả-mở-rộng-)
    - [5.3.1 Hỗ trợ ghi nhớ](#531-hỗ-trợ-ghi-nhớ)
    - [5.3.2 Xử dụng tập tài liệu khác ứng dụng trong y khoa](#532-xử-dụng-tập-tài-liệu-khác-ứng-dụng-trong-y-khoa)
    - [5.3.3 Hỗ trợ làm việc với nhiều tài liệu khác nhau](#533-hỗ-trợ-làm-việc-với-nhiều-tài-liệu-khác-nhau)
- [6. Kết luận 📌](#6-kết-luận-)

</details>

<br/>

# Tóm tắt
Mặc dù LLMs rất mạnh, chúng vẫn bị hạn chế về kiến thức chuyên ngành và tính cập nhật. Dự án này xây dựng hệ thống hỏi đáp thông minh dùng kiến trúc RAG, giúp người học khóa AI tại AI Việt Nam (AIO) khai thác hiệu quả nội dung tài liệu học tập.


# 1. Giới thiệu 🗂 
- Các Mô hình Ngôn ngữ Lớn (LLMs) như ChatGPT có khả năng trả lời linh hoạt nhưng bị giới hạn bởi dữ liệu huấn luyện, nên không xử lý tốt thông tin mới hoặc cá nhân hóa.
- Để khắc phục, kiến trúc Retrieval-Augmented Generation (RAG) cho phép LLM truy xuất thông tin từ nguồn ngoài (như PDF, cơ sở dữ liệu) trước khi tạo câu trả lời, giúp kết quả chính xác và phù hợp hơn.
- Mục tiêu dự án là xây dựng chatbot ứng dụng RAG, hỗ trợ học viên khóa AIO hỏi – đáp trực tiếp dựa trên nội dung tài liệu bài giảng.

# 2. Phương pháp luận 📚 
Hệ thống được xây dựng dựa trên kiến trúc RAG tiêu chuẩn, bao gồm hai quy trình chính: 
- Lập chỉ mục dữ liệu (Indexing) 
- Truy vấn & Tạo sinh (Retrieval & Generation).


![Quy trình RAG tổng quan](/AIO.github.io/images/M01/M01_RAG_1.png)

Hình 1: Sơ đồ tổng quan về chương trình RAG trong project.


## 2.1. Quy trình Lập chỉ mục dữ liệu (Indexing)

<details>
<summary> Bước 1: Tải dữ liệu – Đọc và trích xuất văn bản từ file PDF (PyPDFLoader) </summary>

```python
# Hàm PyPDFLoader
from langchain.document_loaders import PyPDFLoader

# Tải file PDF và trích xuất văn bản
loader = PyPDFLoader(tmp_file_path)
documents = loader.load()
```
</details>

<details>
<summary>Bước 2: Phân đoạn – Chia văn bản thành các đoạn nhỏ (chunks) </summary>
Giải pháp hiện tại là sử dụng SemanticChunker để chia các đoạn dựa theo độ tương đồng về mặt ngữ nghĩa (semantic similarity). 


Quá trình này bao gồm việc tách văn bản thành từng câu, sau đó nhóm mỗi 3 câu lại với nhau, rồi hợp nhất các nhóm có nội dung tương tự nhau dựa trên không gian embedding.

```python
#Hàm SemanticChunker
from langchain.text_splitter import SemanticChunker

semantic_splitter = SemanticChunker(
    embeddings = st.session_state.embeddings,
    buffer_size = 1,
    breakpoint_threshold_type = "percentile",
    breakpoint_threshold_amount = 95,
    min_chunk_size = 500,
    add_start_index = True
)
```

</details>

![Semantic Chunking](/AIO.github.io/images/M01/M01_RAG_3.png)

Hình 2: Sơ đồ về Semantic Chunking.

<details>
<summary>Bước 3: Mã hóa – Chuyển mỗi đoạn văn bản thành vector số học </summary>
Các đoạn văn bản dạng chuỗi cần được chuyển đổi về dạng số học để áp dụng các thuật toán xử lý phù hợp. Việc này gọi là encoding. 

Trong giải pháp hiện tại ta sử dụng `bkai-foundation-models/vietnamese-bi-encoder` làm mô hình embedding để chuyển đổi các đoạn văn bản dạng chuỗi sang không gian vector số.

Việc sử dụng cấu trúc dữ liệu vector giúp việc xử dụng các thuật toán truy vấn vector để tìm kiếm các văn bản tương ướng (ví dụ thuật toán HNSW trong chroma database)


```python
# Sử dụng mô hình embedding bkai-foundation-models/vietnamese-bi-encoder
from langchain.embeddings import HuggingFaceEmbeddings

def load_embeddings():
    return HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")
```
</details>

![Vector database](/AIO.github.io/images/M01/M01_RAG_2.png)

Hình 3: Sơ đồ bước thực hiện xây dựng vector database.


<details>
<summary>Bước 4: Lưu trữ – Lưu các vector vào cơ sở dữ liệu để truy vấn nhanh</summary>

Trong giải pháp hiện tại sử dụng Chroma làm vector database. 

```python    
from langchain.vectorstores import Chroma

#ChromaDB, langchain.vectorstores

# Phân đoạn và lưu trữ vector
docs = semantic_splitter.split_documents(documents)
vector_db = Chroma.from_documents(documents=docs, embedding=st.session_state.embeddings)
retriever = vector_db.as_retriever()

```

</details>


## 2.2. Quy trình Truy vấn và Tạo sinh (Retrieval & Generation)

<details>
<summary>Bước 1: Mã hóa câu hỏi – Chuyển câu hỏi của người dùng thành vector </summary>

Tương tự quy trình lập chỉ mục dữ liệu ở trên. Ta cần chuyển câu hỏi về không gian vector số để có thể áp dụng các thuật toán truy vấn dữ liệu để đối chiếu tới các tài liệu đã được xử lý trước đó. 

Ta cũng sẽ sử dụng mô hình embedding `bkai-foundation-models/vietnamese-bi-encoder` để đảm bảo cả câu hỏi và tài liệu được truy vấn trên cùng một hệ quy chiếu. 


Trong đoạn code sau: 
- Câu hỏi được truyền qua itemgetter("question")
- Sau đó đi qua retriever (được tạo từ vector_db.as_retriever())
- Retriever sử dụng cùng mô hình e`mbedding bkai-foundation-models/vietnamese-bi-encoder` để chuyển câu hỏi thành vector 

```python
retriever = vector_db.as_retriever()
rag_chain = (
    {
        "context": itemgetter("question")
        | retriever  # <-- Tại đây câu hỏi được mã hóa thành vector để tìm kiếm tài liệu liên quan
        ...
    }
    ...
)

```
</details>

<details>
<summary>Bước 2: Truy vấn – Tìm kiếm các đoạn văn bản liên quan nhất trong cơ sở dữ liệu (ChromaDB)</summary>

Tiếp tục Bước 1 được mô tả ở trên: 
- Retriever tiếp tục tìm kiếm các tài liệu tương tự trong vector database.
- Với cấu hình mặc định, retriever sẽ trả về 4 chunks (documents) có độ tương đồng cao nhất với câu hỏi.


</details>


<details>
<summary>Bước 3: Tạo prompt – Kết hợp câu hỏi và đoạn văn bản thành một prompt hoàn chỉnh </summary>

Sau khi tổng hợp các dữ liệu cần thiết (context data), chúng ta sẽ tiến hành tạo Prompt để gởi cho LLM. 

```python
     rag_chain = (
        {
            "context": itemgetter("question")
            | retriever
            | promptManager.format_docs_chunks,
            "question": itemgetter("question"),
            "chat_history": lambda x: promptManager.format_conversation_history(
                x["chat_history"]
            ),
        }
        | prompt # <-- tạo prompt dựa theo câu hỏi và câu trả lời đã truy vấn được. 
        | st.session_state.llm
        | StrOutputParser()
    )
```

Trong giải pháp hiện tại ta sử dụng prompt template sau: 
```yaml
You are an assistant for question-answering tasks. Use the following pieces of retrieved context and conversation history to answer the question. If you don't know the answer, just say that you don't know. 
Instructions:
- Use three sentences maximum
- Keep the answer concise

Conversation history:
{chat_history}

Context:
{context} 

Question: {question} 

Answer:
```

</details>


<details>
<summary>Bước 4: Tạo sinh – Dựa vào prompt đã tăng cường để tạo ra câu trả lời cuối cùng </summary>

Trong giải pháp hiện tại ta:
- Sử dụng mô hình ngôn ngữ `lmsys/vicuna-7b-v1.5`
- Vì giới mặt về mặt phần cứng ta áp dụng quantizing về không gian 4bit để giảm yêu cầu về memory của GPU. 

```python
# Sử dụng mô hình lmsys/vicuna-7b-v1.5  
def load_llm():
  MODEL_NAME = "lmsys/vicuna-7b-v1.5"
  nf4_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
  )
  model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=nf4_config, low_cpu_mem_usage = True
  )
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  model_pipeline = pipeline(
    "text-generation",
    model = model,
    tokenizer = tokenizer,
    max_new_tokens = 512,
    pad_token_id = tokenizer.eos_token_id,
    device_map = "auto"
  )
  return HuggingFacePipeline(pipeline = model_pipeline)
```
</details>



# 3. Thực hiện ⚙ 

Ứng dụng được xây dựng bằng Python với giao diện người dùng tương tác được tạo bởi thư viện Streamlit. Các thư viện chính được sử dụng bao gồm:
- Streamlit: Xây dựng giao diện web cho ứng dụng.
- LangChain: Framework chính để kết nối các thành phần trong chuỗi RAG.
- Hugging Face Transformers: Tải và vận hành các mô hình embedding và LLM.
- ChromaDB: Xây dựng cơ sở dữ liệu vector.
- PyPDF: Xử lý file PDF.

Giao diện ứng dụng cho phép người dùng:
- Tải lên một file tài liệu PDF.
- Nhấn nút "Xử lý PDF" để khởi tạo quy trình lập chỉ mục.
- Nhập câu hỏi vào một khung chat.
- Nhận câu trả lời được tạo ra bởi hệ thống.

Để tối ưu hóa trải nghiệm, các mô hình nặng (embedding và LLM) được cache lại bằng @st.cache_resource của Streamlit, đảm bảo chúng chỉ cần tải một lần duy nhất khi khởi động ứng dụng



# 4. Kết quả 📈 

File code hoàn chỉnh và hình ảnh giao diện của người dùng và kết quả chatbot bằng RAG được ghi nhận sau đây.

**👉 Xem file code**: https://github.com/aio25-mix002/m01-p0102


![Tải model](/AIO.github.io/images/M01/M1-1.png)

Hình 4.1: Giao diện của người dùng - Tải model.


![Tải file](/AIO.github.io/images/M01/M1-2.png)

Hình 4.2: Giao diện của người dùng - Model đã sẵn sàng và tải file.


![Xử lý file](/AIO.github.io/images/M01/M1-3.png)

Hình 4.3: Giao diện của người dùng - Xử lý file.


![Chatbot trả lời](/AIO.github.io/images/M01/M1-5.png)

Hình 4.4: Giao diện của người dùng - Đặt câu hỏi và chatbot trả lời.


# 5. Mở rộng nâng cao 🖥

Điểm cải tiến sau khi thực hiện dự án được đề xuất như nhau:

## 5.1 Tiêu chí cải tiến

| Tiêu chí                                   | Phiên bản cũ                                                                | Phiên bản cải tiến                                                                                                    |
| ------------------------------------------ | --------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Tính năng: Ghi nhớ**                     | ❌ Không ghi nhớ hội thoại trước đó. Mỗi câu hỏi được xử lý độc lập.         | ✅ Ghi nhớ và sử dụng lịch sử trò chuyện để tăng độ chính xác và mạch lạc trong hội thoại.                             |
| **Tính năng: Làm việc với nhiều tài liệu** | ❌ Chỉ làm việc với một tài liệu                                             | ✅ Làm việc được với nhiều tài liệu                                                                                    |
| **Prompt: Prompt template**                | ❌Chỉ bao gồm: tài liệu + câu hỏi                                            | ✅Bao gồm: tài liệu + câu hỏi + lịch sử hội thoại                                                                      |
| **Prompt: Thiết kế prompt**                | ❌Sử dụng prompt template mặc định từ hub                                    | ✅Tự viết prompt cho phép  xử lý linh hoạt hơn.                                                                        |
| **Ứng dụng thực tế**                       | ❌Phù hợp với truy vấn đơn lẻ, không cần bối cảnh trước đó.                  | ✅Thích hợp cho các cuộc hội thoại nhiều lượt cần hiểu ngữ cảnh.                                                       |
| **Coding: Cấu trúc code & Module hóa**     | ❌Mã nguồn đơn giản, ít tách module.                                         | ✅Cấu trúc rõ ràng, chia module tốt với các class riêng như `logging_utils` và `prompt_utils`                          |
| **Coding: Quản lý Vector DB**              | ❌Sử dụng ChromaDB theo mặc định, dễ lỗi khi xử lý nhiều file PDF liên tiếp. | ✅Dùng `chromadb.PersistentClient` và `reset()` trước khi xử lý file mới giúp tránh lỗi và quản lý trạng thái tốt hơn. |
| **Coding: Gỡ lỗi (Debugging)**             | ❌Không có logging.                                                          | ✅Có tích hợp `logger` để ghi lại thông tin debug (ví dụ: các chunks được truy vấn), hỗ trợ phát triển tốt hơn.        |



##  5.2 Code nâng cao

### 5.2.1 Nâng cấp cốt lỗi: Ghi nhớ lịch sử hội thoại (Conversation memory) 
<details>
<summary>5.2.1.1. Xây dựng prompt có chứa lịch sử hội thoại </summary>
Ta sử dụng kỹ thuật Prompting để đưa lịch sử hội thoại vào câu prompt

```yaml
You are an assistant for question-answering tasks. Use the following pieces of retrieved context and conversation history to answer the question. If you don't know the answer, just say that you don't know. 
Instructions:
- Use three sentences maximum
- Keep the answer concise

Conversation history:
{chat_history}

Context:
{context} 

Question: {question} 

Answer:
```
</details>


<details>
<summary>5.2.1.2. Định dạng và truy xuất lịch sử chat</summary>
Lý tưởng thì ta có thể đưa toàn độ đoạn hội thoại vào prompt, tuy nhiên việc này có thể gây vượt quá context windows mà LLM model có thể hỗ trợ. 

Giải pháp hiện tại là áp dụng kỹ thuật đơn giản nhất là lấy 10 tin nhắn gần đây nhất. 
- Ưu điểm: Dễ triển khai
- Nhược điểm/hướng cải tiến: 
    - Vấn đề vượt quá context windows cũng có thể xảy ra
    - Các tin nhắn quá khứ nếu không liên quan đến câu hỏi hiện tại cũng có thể gây nhiễu và ảnh hưởng đến kết quả đầu ra. 

```python

def retrieve_chat_history():
    message_threshold = 10
    return st.session_state.chat_history[-message_threshold:-1]

def format_history(histories):
    formatted_history = ""
    for msg in histories:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted_history += f"{role}: {msg['content']}\n\n"
    return formatted_history.strip()
```
</details>

<details>
<summary>5.2.1.3. Cập nhật RAG Chain để xử lý lịch sử chat </summary>

```python
def process_pdf_updated_chain(retriever, llm):
    prompt = build_prompt_ragprompt_withhistory_en()
    rag_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "chat_history": lambda x: format_history(x["chat_history"]) # <--- chat_history được đưa vào context data tại đây
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
```
</details>


<details>
<summary>5.2.1.4. Cập nhật cách gọi RAG chain (main_updated_invoke) </summary>
```python
#Hàm main_updated_invoke
def main_updated_invoke(user_input):
    output = st.session_state.rag_chain.invoke({
        "question": user_input,
        "chat_history": retrieve_chat_history()
    })
```
</details>

### 5.2.2 Quản lý Vector DB nâng cao
<details>
<summary>Lưu Vector DB xuống ổ đĩa (persistence) để dễ debug và tránh các lỗi trên in-memory </summary>

```python
def get_chroma_client(allow_reset=False):
    """Get a Chroma client for vector database operations."""
    return chromadb.PersistentClient(settings=chromadb.Settings(allow_reset=allow_reset))

def process_pdf_updated_db_handling():
    client = get_chroma_client(allow_reset=True)
    client.reset()
    vector_db = Chroma.from_documents(
        documents=docs, 
        embedding=st.session_state.embeddings,
        client=client
    )
```
</details>


### 5.2.3. Gỡ lỗi (Debugging) với Logger
<details>
<summary>Thêm logger vào ứng dụng để dễ truy vết </summary>

```python
def format_docs_with_logging(docs):
    logger.info(f"**Debug: Retrieved {len(docs)} chunks:**")
    for i, doc in enumerate(docs):
        page_num = doc.metadata.get('page') + 1 if 'page' in doc.metadata else -1
        source = doc.metadata.get('source', 'document')
        file_name = os.path.basename(source) if isinstance(source, str) else 'unknown'

        logger.info(f"""
        ([reference-{i+1}] Page {page_num} - Source: {file_name})
        {doc.page_content}""")
    
    return "\n\n".join(doc.page_content for doc in docs)
```
</details>

### 5.2.4. Xử lý và truy vấn từ nhiều file tài liệu
<details>

```python
def process_pdf(uploaded_files):
    """Process multiple uploaded PDF files, combine their docs, and build a single retriever and RAG chain."""
    all_docs = []
    file_names = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(
            delete=False, prefix=uploaded_file.name, suffix=".pdf"
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        semantic_splitter = SemanticChunker(
            embeddings=st.session_state.embeddings,
            buffer_size=1,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
            min_chunk_size=500,
            add_start_index=True,
        )
        docs = semantic_splitter.split_documents(documents)
        # for doc in docs:
        #     # Add file name to doc metadata for citation
        #     doc.metadata["source"] = uploaded_file.name
        all_docs.extend(docs)
        file_names.append(uploaded_file.name)
        os.unlink(tmp_file_path)

    chroma_client = get_chroma_client(allow_reset=True)
    chroma_client.reset()  # empties and completely resets the database. This is destructive and not reversible.

    vector_db = Chroma.from_documents(
        documents=all_docs, embedding=st.session_state.embeddings, client=chroma_client
    )
    retriever = vector_db.as_retriever()
    # prompt = hub.pull("rlm/rag-prompt")
    prompt = promptManager.load_prompt_template_from_file("rag_with_memory.v1.txt")

    # Build the RAG chain
    # This use  LangChain Expression Language (LCEL) 
    # where  LangChain overrides the Python pipe operator (|) for its Runnable objects
    # Ref: https://python.langchain.com/docs/concepts/lcel/
    rag_chain = (
        {
            "context": itemgetter("question")
            | retriever
            | promptManager.format_docs_chunks,
            "question": itemgetter("question"),
            "chat_history": lambda x: promptManager.format_conversation_history(
                x["chat_history"]
            ),
        }
        | prompt
        | st.session_state.llm
        | StrOutputParser()
    )
    return rag_chain, len(all_docs), file_names

```
</details>

##  5.3 Kết quả mở rộng 📍

Hình ảnh giao diện của người dùng và kết quả chatbot bằng RAG sau khi cải tiến được ghi nhận sau đây.

### 5.3.1 Hỗ trợ ghi nhớ 
![Data mẫu YOLOv10_Tutorials](/AIO.github.io/images/M01/M1-6.png)

Hình 5: Kết quả giao diện của người dùng với file Data mẫu YOLOv10_Tutorials.pdf


### 5.3.2 Xử dụng tập tài liệu khác ứng dụng trong y khoa
![file Medical Report](/AIO.github.io/images/M01/M1-7.png)

Hình 6: Kết quả giao diện của người dùng với file Medical Report

### 5.3.3 Hỗ trợ làm việc với nhiều tài liệu khác nhau
![file Multiple File](/AIO.github.io/images/M01/M1-8.jpg)

Hình 7: Kết quả giao diện làm việc với nhiều tài liệu khác nhau

# 6. Kết luận 📌 

- Dự án đã **xây dựng thành công một chatbot ứng dụng kiến trúc RAG, có khả năng hỏi đáp trực tiếp và hiệu quả với các tài liệu PDF chuyên biệt**, phù hợp với ngữ cảnh bằng cách kết hợp truy vấn thông tin của cơ sở dữ liệu vector và khả năng tạo sinh ngôn ngữ của LLMs.
  
- Chất lượng câu trả lời của hệ thống **phụ thuộc hoàn toàn vào hiệu quả của bước truy vấn thông tin (retrieval)**. Nếu quá trình tìm kiếm ngữ nghĩa không tìm được đúng đoạn văn bản chứa thông tin liên quan trong Vector Database, mô hình LLM sẽ không có đủ ngữ cảnh cần thiết, dẫn đến nguy cơ tạo ra câu trả lời sai, không đầy đủ hoặc không liên quan đến câu hỏi của người dùng.
  
- Với phương pháp này, dự án mở ra nhiều hướng phát triển tiềm năng trong tương lai để **tiếp tục tối ưu hóa tốc độ, độ chính xác và nâng cao trải nghiệm người dùng**.
