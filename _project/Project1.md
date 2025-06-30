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

| Bước     | Tên bước      | Mục đích                                                           | Thư viện/Hàm hoặc Mô hình                            |
|----------|---------------|---------------------------------------------------------------------|------------------------------------------------------|
| 1   | Tải dữ liệu    | Đọc và trích xuất văn bản từ file PDF                              | PyPDFLoader                                           |
| 2   | Phân đoạn      | Chia văn bản thành các đoạn nhỏ (chunks) có ý nghĩa                | SemanticChunker                                       |
| 3   | Mã hóa         | Chuyển mỗi đoạn văn bản thành vector số học                        | bkai-foundation-models/vietnamese-bi-encoder         |
| 4   | Lưu trữ        | Lưu các vector vào một cơ sở dữ liệu để truy vấn nhanh             | ChromaDB                                              |

---

### 2.2. Quy trình Truy vấn và Tạo sinh (Retrieval & Generation)

| Bước     | Tên bước             | Mục đích                                                                 | Thư viện/Hàm hoặc Mô hình                          |
|----------|----------------------|--------------------------------------------------------------------------|----------------------------------------------------|
| 1   | Mã hóa câu hỏi        | Chuyển câu hỏi của người dùng thành vector                              | bkai-foundation-models/vietnamese-bi-encoder       |
| 2   | Truy vấn              | Tìm kiếm các đoạn văn bản liên quan nhất trong cơ sở dữ liệu            | ChromaDB                                            |
| 3   | Tăng cường            | Kết hợp câu hỏi và đoạn văn bản thành một prompt hoàn chỉnh             | Mẫu Prompt: rlm/rag-prompt                         |
| 4   | Tạo sinh              | Dựa vào prompt đã tăng cường để tạo ra câu trả lời cuối cùng           | lmsys/vicuna-7b-v1.5                                |

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

<details> <summary>Hiển thị nội dung file <code>/AIO.github.io/M01_rag_chatbot.py</code></summary>



## 5. Mở rộng nhân cao
