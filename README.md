# ORANGECHAT2 — English → Darija Translation Model

###  Project Overview
**ORANGECHAT2** is a research and development project conducted for **Orange Maroc**, aiming to build a neural machine translation (NMT) model that translates **English text into Moroccan Darija (Arabic dialect)**.  
The goal is to create a foundation for **Orange’s first voice-based virtual assistant** capable of interacting naturally with Moroccan customers through IVR (Interactive Voice Response) systems.

> **Note:** This repository contains partial code and documentation due to contractual restrictions with Orange. The complete implementation remains proprietary.

---


##  Objectives
- Develop a **Seq2Seq neural model** to translate English → Darija.
- Preprocess and clean large-scale Darija datasets using **Pandas** and **NumPy**.
- Implement a **tokenization and embedding** pipeline to handle the complexities of the Moroccan dialect.
- Train and test the model using **PyTorch/TensorFlow**, evaluating accuracy and loss convergence.
- Explore integration potential with **Orange’s IVR system** for future AI-driven customer service.

---

##  Technical Stack
| Category | Tools & Frameworks |
|-----------|--------------------|
| Programming Language | Python |
| Machine Learning | TensorFlow, PyTorch |
| NLP Libraries | Pandas, NLTK, TensorBoard |
| Model Type | Seq2Seq (Encoder–Decoder with LSTM Layers) |
| Optimization | Adam Optimizer, Sparse Categorical Crossentropy |
| Visualization | TensorBoard, Matplotlib |
| Dataset Management | Pandas DataFrame, CSV Sampling |

---

##  Data Source
The dataset used for model training was derived from **[PyDoDA (Python Darija Open Dataset for AI)](https://www.researchgate.net/publication/350131925_Moroccan_Dialect_-Darija-_Open_Dataset)**.

### Dataset Composition:
- **Semantic Categories**: Colors, art, animals, food, professions, environment.
- **Syntactic Categories**: Sentences, nouns, adverbs, definitives.
- **X-tra Category**: Proverbs, idioms, and region-specific expressions.

Each entry includes:
| Darija | Darija_ar | English |


Darija introduces unique preprocessing challenges due to:
- Mixed use of Arabic and Latin characters.
- Regionally diverse vocabulary (e.g., North influenced by Spanish).
- Variable spellings using numbers (e.g., “7” = خ, “9” = ق).

---

##  Model Design

The translation system is based on a **Sequence-to-Sequence (Seq2Seq)** architecture, composed of:
- **Encoder**: Compresses English input sequence into context vectors.
- **Decoder**: Generates the Darija translation sequentially.
- **Attention Mechanism**: Improves accuracy on longer sequences.

### Key Parameters:
```python
batch_size = 64
learning_rate = 0.001
epochs = 350
optimizer = 'adam'
loss_function = 'sparse_categorical_crossentropy'
```
---


##  Implementation Steps

### 1. Data Preprocessing
- Loaded CSV files and merged multiple variations of words.  
- Applied cleaning functions (`strip()`, `lower()`, remove nulls/duplicates).  
- Retained numeric characters representing Darija phonemes (e.g., `3`, `7`, `9`).  
- Visualized preprocessed samples via `pandas.DataFrame.head()`.

### 2. Tokenization & Tensor Conversion
- Tokenized text sequences with padding, truncation, and `max_length`.  
- Converted processed data into tensors compatible with **TensorFlow**.  
- Verified token indices and vocabulary mapping.

### 3. Model Training
- Built **Encoder–Decoder** using `LSTM`, `Dense`, and `Embedding` layers.  
- Split data into **80% training / 20% evaluation**.  
- Compiled model using **Adam optimizer** and **Sparse Categorical Crossentropy** loss.  
- Trained model with live metrics visualization in **TensorBoard**.

### 4. Model Evaluation
- Plotted **loss** and **learning rate** curves for each epoch.  
- Observed stable convergence indicating healthy training.  
- Created a translation function using `argmax` to predict Darija sequences.

---

##  Results & Evaluation
The model successfully learned preliminary translation patterns between **English** and **Darija**.

**Training metrics showed:**
-  Decreasing loss curve over epochs.  
-  Stable learning rate convergence.  
-  Correct handling of tokenized Darija data.

**However, translation accuracy remained limited due to:**
-  Hardware constraints (estimated full dataset training time: ~367 hours).  
-  Limited compute performance on local machine (GPU unavailability).  
-  Need for more advanced architectures (e.g., **Transformer** or **GRU**).

---

##  Challenges & Limitations
- **Hardware Limitations:** Local resources restricted training speed and model depth.  
- **Dataset Complexity:** The Darija dataset is large and regionally varied, making normalization difficult.  
- **Underfitting Risk:** The simple LSTM Seq2Seq model lacks contextual depth compared to Transformer-based models.  
- **Tokenizer Issues:** Word-level tokenization caused unknown word mismatches.  
  → *Potential fix:* Implement **Byte Pair Encoding (BPE)** for better token generalization.

---

##  Lessons Learned
- Bridging theoretical **machine learning** concepts with real industry applications.  
- Improved understanding of **NLP pipelines** and **data engineering**.  
- Exposure to **professional workflows** and **project management** at **Orange Maroc**.  
- Developed stronger **communication** and **adaptability** skills in a corporate environment.

---

##  Acknowledgments
- **Mr. Abderrahmane Jabri** – Company Supervisor, *Orange Maroc*  
- **PyDoDA Team** – For providing the open-source dataset  
- **Orange Maroc** – For supporting AI innovation and local language technology  

---

## 📩 Contact
For inquiries or collaboration:

**Adnane Ahroum**  
📧 [adnaneahroum69@gmail.com]  
🏢 Orange Maroc — Research Collaboration  
 
---

© 2025 **Adnane Ahroum** — All rights reserved.  
*This project was developed as part of a contractual engagement with Orange Maroc. Full implementation details remain proprietary.*
