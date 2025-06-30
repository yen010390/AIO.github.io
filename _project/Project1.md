---
title: "Project 1.2 -  Ứng dụng RAG trong việc hỏi đáp tài liệu bài học AIO"
excerpt: "Dự án này xây dựng hệ thống hỏi đáp thông minh dùng kiến trúc RAG, giúp người học khóa AI tại AI Việt Nam (AIO) khai thác hiệu quả nội dung tài liệu học tập."
collection: project
---

# Báo cáo dự án
Tác giả: Nguyễn Tuấn Anh - Đoàn Tấn Hưng - Hồ Thị Ngọc Huyền - Trần Thị Mỹ Tú - Đặng Thị Hoàng Yến
Ngày: 30 tháng 6 năm 2025

## Tóm tắt
Mặc dù LLMs rất mạnh, chúng vẫn bị hạn chế về kiến thức chuyên ngành và tính cập nhật. Dự án này xây dựng hệ thống hỏi đáp thông minh dùng kiến trúc RAG, giúp người học khóa AI tại AI Việt Nam (AIO) khai thác hiệu quả nội dung tài liệu học tập.


## 1. Giới thiệu
- Các Mô hình Ngôn ngữ Lớn (LLMs) như ChatGPT có khả năng trả lời linh hoạt nhưng bị giới hạn bởi dữ liệu huấn luyện, nên không xử lý tốt thông tin mới hoặc cá nhân hóa.

- Để khắc phục, kiến trúc Retrieval-Augmented Generation (RAG) cho phép LLM truy xuất thông tin từ nguồn ngoài (như PDF, cơ sở dữ liệu) trước khi tạo câu trả lời, giúp kết quả chính xác và phù hợp hơn.

- Mục tiêu dự án là xây dựng chatbot ứng dụng RAG, hỗ trợ học viên khóa AIO hỏi – đáp trực tiếp dựa trên nội dung tài liệu bài giảng.

## 2. Phương pháp luận
Hệ thống được xây dựng dựa trên kiến trúc RAG tiêu chuẩn, bao gồm hai quy trình chính: Lập chỉ mục dữ liệu (Indexing) và Truy vấn & Tạo sinh (Retrieval & Generation).

![Quy trình RAG tổng quan](/AIO.github.io/images/M01/M01_RAG_1.png)

Hình 1: Sơ đồ tổng quan về kiến trúc RAG được sử dụng trong dự án.

### 2.1. Quy trình Lập chỉ mục dữ liệu (Indexing)

<details>
<summary>Bước 1: Tải dữ liệu – Đọc và trích xuất văn bản từ file PDF <code>PyPDFLoader</code></summary>

<pre><code class="language-python">
from langchain.document_loaders import PyPDFLoader

# Tải file PDF và trích xuất văn bản
loader = PyPDFLoader(tmp_file_path)
documents = loader.load()
</code></pre>
</details>

<details>
<summary>Bước 2: Phân đoạn – Chia văn bản thành các đoạn nhỏ (chunks) có ý nghĩa <code>SemanticChunker</code></summary>

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

<details>
<summary>Bước 3: Mã hóa – Chuyển mỗi đoạn văn bản thành vector số học <code>bkai-foundation-models/vietnamese-bi-encoder</code></summary>

<pre><code class="language-python">
from langchain.embeddings import HuggingFaceEmbeddings

def load_embeddings():
    return HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")
</code></pre>
</details>

<details>
<summary>Bước 4: Lưu trữ – Lưu các vector vào cơ sở dữ liệu để truy vấn nhanh <code>ChromaDB</code></summary>

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

### 2.2. Quy trình Truy vấn và Tạo sinh (Retrieval & Generation)

<details>
<summary>Bước 1: Mã hóa câu hỏi – Chuyển câu hỏi của người dùng thành vector  <code>ChromaDB</code></summary>

```python
@st.cache_resource
def load_embeddings():
  return HuggingFaceEmbeddings(model_name = "bkai-foundation-models/vietnamese-bi-encoder")
```
</details>

<details>
<summary>Bước 2: Truy vấn – Tìm kiếm các đoạn văn bản liên quan nhất trong cơ sở dữ liệu   <code>ChromaDB</code></summary>

```python
@st.cache_resource
def load_embeddings():
  return HuggingFaceEmbeddings(model_name = "bkai-foundation-models/vietnamese-bi-encoder")
```
</details>


<details>
<summary>Bước 3: Tăng cường – Kết hợp câu hỏi và đoạn văn bản thành một prompt hoàn chỉnh   <code>Mẫu Prompt: rlm/rag-prompt </code></summary>

```python
 rlm/rag-prompt
```
</details>


<details>
<summary>Bước 4: Tạo sinh – Dựa vào prompt đã tăng cường để tạo ra câu trả lời cuối cùng   <code>lmsys/vicuna-7b-v1.5  </code></summary>

```python
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


## 3. Thực hiện

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

## 4. Kết quả

Hình 2: Giao diện ứng dụng khi trả lời câu hỏi của người dùng.

<details> <summary>Hiển thị nội dung file code <code>/AIO.github.io/M01_rag_chatbot.py</code></summary>
</details>


## 5. Mở rộng nhân cao
