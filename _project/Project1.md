---
title: "Project 1.2 -  Ứng dụng RAG trong việc hỏi đáp tài liệu bài học AIO"
excerpt: "Dự án này xây dựng hệ thống hỏi đáp thông minh dùng kiến trúc RAG, giúp người học khóa AI tại AI Việt Nam (AIO) khai thác hiệu quả nội dung tài liệu học tập."
collection: project
---

# Báo cáo dự án
Tác giả: Nguyễn Tuấn Anh - Đoàn Tấn Hưng - Hồ Thị Ngọc Huyền - Trần Thị Mỹ Tú - Đặng Thị Hoàng Yến

Ngày: 30 tháng 6 năm 2025

# Tóm tắt
Mặc dù LLMs rất mạnh, chúng vẫn bị hạn chế về kiến thức chuyên ngành và tính cập nhật. Dự án này xây dựng hệ thống hỏi đáp thông minh dùng kiến trúc RAG, giúp người học khóa AI tại AI Việt Nam (AIO) khai thác hiệu quả nội dung tài liệu học tập.


# 1. Giới thiệu
- Các Mô hình Ngôn ngữ Lớn (LLMs) như ChatGPT có khả năng trả lời linh hoạt nhưng bị giới hạn bởi dữ liệu huấn luyện, nên không xử lý tốt thông tin mới hoặc cá nhân hóa.
- Để khắc phục, kiến trúc Retrieval-Augmented Generation (RAG) cho phép LLM truy xuất thông tin từ nguồn ngoài (như PDF, cơ sở dữ liệu) trước khi tạo câu trả lời, giúp kết quả chính xác và phù hợp hơn.
- Mục tiêu dự án là xây dựng chatbot ứng dụng RAG, hỗ trợ học viên khóa AIO hỏi – đáp trực tiếp dựa trên nội dung tài liệu bài giảng.

# 2. Phương pháp luận
Hệ thống được xây dựng dựa trên kiến trúc RAG tiêu chuẩn, bao gồm hai quy trình chính: Lập chỉ mục dữ liệu (Indexing) và Truy vấn & Tạo sinh (Retrieval & Generation).

![Quy trình RAG tổng quan](/AIO.github.io/images/M01/M01_RAG_1.png)

Hình 1: Sơ đồ tổng quan về chương trình RAG trong project.


## 2.1. Quy trình Lập chỉ mục dữ liệu (Indexing)

<details>
<summary>Bước 1: Tải dữ liệu – Đọc và trích xuất văn bản từ file PDF:  <code>PyPDFLoader</code></summary>


<pre><code class="language-python">
from langchain.document_loaders import PyPDFLoader

# Tải file PDF và trích xuất văn bản
loader = PyPDFLoader(tmp_file_path)
documents = loader.load()
</code></pre>
</details>

<details>
<summary>Bước 2: Phân đoạn – Chia văn bản thành các đoạn nhỏ (chunks) có ý nghĩa:  <code>SemanticChunker</code></summary>

<pre><code class="language-python">
from langchain.text_splitter import SemanticChunker

semantic_splitter = SemanticChunker(
    embeddings = st.session_state.embeddings,
    buffer_size = 1,
    breakpoint_threshold_type = "percentile",
    breakpoint_threshold_amount = 95,
    min_chunk_size = 500,
    add_start_index = True
)
</code></pre>
</details>

![Semantic Chunking](/AIO.github.io/images/M01/M01_RAG_3.png)

Hình 2: Sơ đồ về Semantic Chunking.

<details>
<summary>Bước 3: Mã hóa – Chuyển mỗi đoạn văn bản thành vector số học:  <code>bkai-foundation-models/vietnamese-bi-encoder</code></summary>

<pre><code class="language-python">
from langchain.embeddings import HuggingFaceEmbeddings

def load_embeddings():
    return HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")
</code></pre>
</details>

![Vector database](/AIO.github.io/images/M01/M01_RAG_2.png)

Hình 3: Sơ đồ bước thực hiện xây dựng vector database.

<details>
<summary>Bước 4: Lưu trữ – Lưu các vector vào cơ sở dữ liệu để truy vấn nhanh:  <code>ChromaDB</code></summary>


<pre><code class="language-python">
from langchain.vectorstores import Chroma

# Phân đoạn và lưu trữ vector
docs = semantic_splitter.split_documents(documents)
vector_db = Chroma.from_documents(documents=docs, embedding=st.session_state.embeddings)
retriever = vector_db.as_retriever()

# Tải prompt mẫu từ hub
from langchain import hub
prompt = hub.pull("rlm/rag-prompt")
</code></pre>
</details>
---

## 2.2. Quy trình Truy vấn và Tạo sinh (Retrieval & Generation)

<details>
<summary>Bước 1: Mã hóa câu hỏi – Chuyển câu hỏi của người dùng thành vector:  <code>ChromaDB</code></summary>


<pre><code class="language-python">
@st.cache_resource
def load_embeddings():
  return HuggingFaceEmbeddings(model_name = "bkai-foundation-models/vietnamese-bi-encoder")
</code></pre>
</details>

<details>
<summary>Bước 2: Truy vấn – Tìm kiếm các đoạn văn bản liên quan nhất trong cơ sở dữ liệu:   <code>ChromaDB</code></summary>


<pre><code class="language-python">
@st.cache_resource
def load_embeddings():
  return HuggingFaceEmbeddings(model_name = "bkai-foundation-models/vietnamese-bi-encoder")
</code></pre>
</details>


<details>
<summary>Bước 3: Tăng cường – Kết hợp câu hỏi và đoạn văn bản thành một prompt hoàn chỉnh:   <code>Mẫu Prompt: rlm/rag-prompt </code></summary>

    
<pre><code class="language-python">
 rlm/rag-prompt
</code></pre>
</details>


<details>
<summary>Bước 4: Tạo sinh – Dựa vào prompt đã tăng cường để tạo ra câu trả lời cuối cùng:   <code>lmsys/vicuna-7b-v1.5  </code></summary>


<pre><code class="language-python">
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
</code></pre>
</details>


# 3. Thực hiện

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

# 4. Kết quả

Hình 4: Giao diện ứng dụng khi trả lời câu hỏi của người dùng.

[👉 Xem file code](/AIO.github.io/files/M01_rag_chatbot.py)


# 5. Mở rộng nhân cao

## **5.1 Điểm cải tiến:** khả năng ghi nhớ lịch sử hội thoại

## **5.2 Tiêu chí cải tiến:**

| Tiêu chí                     | Phiên bản cũ                                                                                           | Phiên bản cải tiến                                                                                                   |
|-----------------------------|----------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| **Khả năng ghi nhớ**        | ❌ Không ghi nhớ hội thoại trước đó. Mỗi câu hỏi được xử lý độc lập.                                                       | ✅ Ghi nhớ và sử dụng lịch sử trò chuyện để tăng độ chính xác và mạch lạc trong hội thoại.                                             |
| **Cách gọi `rag_chain`**    | Gọi với **chỉ câu hỏi**:<br>`rag_chain.invoke(user_input)`                                                                 | Gọi với **câu hỏi + lịch sử hội thoại**:<br>`rag_chain.invoke({ "question": user_input, "chat_history": retrieve_chat_history() })`   |
| **Thiết kế prompt**         | Sử dụng prompt mặc định từ hub, không có thông tin hội thoại trước đó.                                                     | Prompt hỗ trợ tích hợp lịch sử hội thoại, tùy chọn tiếng Việt/Anh, cho phép thử nghiệm linh hoạt hơn.                                |
| **Ứng dụng thực tế**        | Phù hợp với truy vấn đơn lẻ, không cần bối cảnh trước đó.                                                                  | Thích hợp cho các cuộc hội thoại nhiều lượt cần hiểu ngữ cảnh.                                                                        |
| **Cấu trúc & Module hóa**   | Mã nguồn đơn giản, ít tách module.                                                                                         | Cấu trúc rõ ràng, chia module tốt với các hàm riêng như `build_prompt_...`, `get_chroma_client()`.                                    |
| **Quản lý Vector DB**       | Sử dụng ChromaDB theo mặc định, dễ lỗi khi xử lý nhiều file PDF liên tiếp.                                                  | Dùng `chromadb.PersistentClient` và `reset()` trước khi xử lý file mới giúp tránh lỗi và quản lý trạng thái tốt hơn.                 |
| **Kỹ thuật Prompt**         | Duy nhất một prompt từ `hub.pull("rlm/rag-prompt")`.                                                                      | Có nhiều tùy chọn prompt: tiếng Việt, tiếng Anh, tích hợp lịch sử hội thoại.                                                         |
| **Giao diện người dùng (UI)**| Giao diện đơn giản.                                                                                                        | Có nút "Xóa lịch sử chat", hiển thị logo (`st.logo`), các nút sử dụng `use_container_width=True` giúp UI gọn gàng, hiện đại hơn.     |
| **Gỡ lỗi (Debugging)**      | Không có logging.                                                                                                           | Có tích hợp `logger` để ghi lại thông tin debug (ví dụ: các chunks được truy vấn), hỗ trợ phát triển tốt hơn.                        |
| **Thư viện phụ thuộc**      | Ít thư viện hơn.                                                                                                            | Thêm thư viện như `chromadb`, `ChatPromptTemplate`, `itemgetter` và module `utils` tùy chỉnh.                                         |


##  5.3 Code nâng cao
[👉 Xem file code cải tiến](/AIO.github.io/files/M01_rag_chatbot_cai_tien.py)


Hình 5: Giao diện ứng dụng khi trả lời câu hỏi của người dùng nâng cao.

### 5.3.1 Nâng cấp cốt lỗi: Ghi nhớ lịch sử hội thoại (Conversation memory) 
<details>
<summary>1.1. Hàm xây dựng prompt có chứa lịch sử hội thoại: <code>build_prompt_ragprompt_withhistory_en</code></summary>


<pre><code class="language-python">
def build_prompt_ragprompt_withhistory_en():
    template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context and conversation history to answer the question. If you don't know the answer, just say that you don't know. 
    Instructions:
    - Use three sentences maximum
    - Keep the answer concise

    Conversation history:
    {chat_history}
    
    Context:
    {context} 

    Question: {question} 

    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    return prompt
</code></pre>
</details>


<details>
<summary>1.2. Hàm định dạng và truy xuất lịch sử chat: <code>retrieve_chat_history, format_history</code></summary>


<pre><code class="language-python">
def retrieve_chat_history():
    message_threshold = 10
    return st.session_state.chat_history[-message_threshold:] if len(st.session_state.chat_history) >= message_threshold else st.session_state.chat_history

def format_history(histories):
    formatted_history = ""
    for msg in histories:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted_history += f"{role}: {msg['content']}\n\n"
    return formatted_history.strip()
</code></pre>
</details>

<details>
<summary>1.3. Cập nhật RAG Chain để xử lý lịch sử chat: <code>process_pdf_updated_chain(retriever, llm)</code></summary>


<pre><code class="language-python">
def process_pdf_updated_chain(retriever, llm):
    prompt = build_prompt_ragprompt_withhistory_en()
    rag_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "chat_history": lambda x: format_history(x["chat_history"])
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
</code></pre>
</details>


<details>
<summary>1.4. Cập nhật cách gọi RAG chain: <code>main_updated_invoke</code></summary>


<pre><code class="language-python">
def main_updated_invoke(user_input):
    output = st.session_state.rag_chain.invoke({
        "question": user_input,
        "chat_history": retrieve_chat_history()
    })
</code></pre>
</details>

### 5.3.2 Quản lý Vector DB nâng cao
<details>
<summary>Quản lý Vector DB nâng cao: <code>get_chroma_client, process_pdf_updated_db_handling</code></summary>

    
<pre><code class="language-python">
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
</code></pre>
</details>


### 5.3.3. Gỡ lỗi (Debugging) với Logger
<details>
<summary>Gỡ lỗi (Debugging) với Logger: <code>format_docs_with_logging</code></summary>


<pre><code class="language-python">
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
</code></pre>
</details>

### 5.3.4. Cải tiến giao diện người dùng (UI)
<details>
<summary>Cải tiến giao diện người dùng: <code>main_sidebar_enhancements</code></summary>


<pre><code class="language-python">
def main_sidebar_enhancements():
    with st.sidebar:
        st.logo("./assets/logo.png")
        st.subheader("💬 Điều khiển Chat")
        if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
            clear_chat()
            st.rerun()
</code></pre>
</details>

# 6. Kết luận
- Dự án đã xây dựng thành công một chatbot ứng dụng kiến trúc RAG, có khả năng hỏi đáp trực tiếp và hiệu quả với các tài liệu PDF chuyên biệt, phù hợp với ngữ cảnh bằng cách kết hợp truy vấn thông tin của cơ sở dữ liệu vector và khả năng tạo sinh ngôn ngữ của LLMs.
- Chất lượng câu trả lời của hệ thống phụ thuộc hoàn toàn vào hiệu quả của bước truy vấn thông tin (retrieval). Nếu quá trình tìm kiếm ngữ nghĩa không tìm được đúng đoạn văn bản chứa thông tin liên quan trong Vector Database, mô hình LLM sẽ không có đủ ngữ cảnh cần thiết, dẫn đến nguy cơ tạo ra câu trả lời sai, không đầy đủ hoặc không liên quan đến câu hỏi của người dùng.
- Với phương pháp này, dự án mở ra nhiều hướng phát triển tiềm năng trong tương lai để tiếp tục tối ưu hóa tốc độ, độ chính xác và nâng cao trải nghiệm người dùng.
