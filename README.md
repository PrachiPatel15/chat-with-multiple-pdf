# 📄 Chat with PDF using Gemini

A simple **RAG-based** (Retrieval-Augmented Generation) application that allows users to **chat with PDFs** using **Google Gemini AI**. This app extracts text from PDFs, processes it using embeddings, and enables conversational Q&A with page number references.

---

## ✨ Features
- 📂 **Upload multiple PDFs** and extract text.
- 🔍 **Search for answers** in the PDFs using **Google Gemini AI**.
- 📄 **Get page numbers** where the answer is found.
- 🖥️ **Minimal and sleek UI** with dark mode compatibility.

---

## 🚀 Installation & Setup

### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### **2️⃣ Create a Virtual Environment**
```sh
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

### **3️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

### **4️⃣ Set Up Environment Variables**
Create a **`.env`** file in the root directory and add your **Google API key**:
```sh
GOOGLE_API_KEY=your_api_key_here
```

### **5️⃣ Run the Application**
```sh
streamlit run app.py
```

---

## 🎯 How It Works

1. **Upload PDF(s)** 📂
2. **Ask a Question** ❓
3. **Get AI-powered answers** with relevant page numbers 📄
4. **View highlighted page numbers** in the UI 🎨

---

## 🖼️ Screenshots
_Add screenshots here to showcase the app interface and responses._

![Upload PDF](path/to/upload_screenshot.png)

![Ask a Question](path/to/question_screenshot.png)

![Get Answer with Page Numbers](path/to/answer_screenshot.png)

---

## 🛠️ Built With
- **Streamlit** - Frontend UI
- **FAISS** - Vector Search
- **Google Gemini AI** - LLM for Q&A
- **PyPDF2** - PDF Text Extraction

---

## 💡 Future Enhancements
- ✅ **Improve answer relevance** by refining embeddings.
- ✅ **Better UI elements** for page number highlighting.
- ✅ **Add multi-PDF comparison support**.

---

## 🤝 Contributing
Pull requests are welcome! Please **open an issue** first to discuss any major changes.

```sh
git checkout -b feature-branch
git commit -m "Your feature"
git push origin feature-branch
```

---
Happy Coding! 🚀

