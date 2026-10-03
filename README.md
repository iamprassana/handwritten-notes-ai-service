# Handwritten Notes AI - FastAPI Inference Service

This service is intentionally **stateless**.

FastAPI does NOT access:

- PostgreSQL
- Supabase
- object storage
- Qdrant
- user authentication
- user authorization

Node.js owns all application/data logic.

FastAPI only performs:

1. YOLO page detection
2. OpenCV preprocessing
3. ColQwen2 image/question embeddings
4. Qwen3-VL answer generation

## Architecture

```text
Frontend
   |
   v
Node.js
   |
   |-- Database
   |-- Object Storage
   |-- Qdrant
   |
   +---- image ----------> FastAPI
   |                         |
   |                         +--> YOLO
   |                         +--> OpenCV
   |                         +--> ColQwen2
   |                         |
   |<---- embedding --------+
   |
   +---- question --------> FastAPI
   |                         |
   |                         +--> ColQwen2
   |                         |
   |<---- embedding --------+
   |
   +---- question + top-K images --> FastAPI
                                     |
                                     +--> Qwen3-VL
                                     |
                                     +---- answer ------> Node.js
```

## API

### Image embedding

`POST /api/embeddings/image`

Multipart field:

```text
image=<image file>
```

Response:

```json
{
  "embedding": [[...], [...]],
  "shape": [103, 128]
}
```

### Question embedding

`POST /api/embeddings/question`

JSON:

```json
{
  "question": "What is normalization?"
}
```

### Answer

`POST /api/answer`

Multipart:

```text
question=What is normalization?
images=<page1.jpg>
images=<page2.jpg>
images=<page3.jpg>
```

Response:

```json
{
  "answer": "..."
}
```

## Setup

Create a virtual environment:

```bash
python -m venv venv
```

Activate on Windows:

```bash
venv\Scripts\activate
```

Activate on Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env`.

Put your YOLO model here:

```text
models/best.pt
```

Start:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at:

```text
http://localhost:8000
```

Swagger:

```text
http://localhost:8000/docs
```

## Important

The model implementations in this service should match the exact versions and inference code used in your validated notebook. In particular, verify the installed `transformers` and `ColQwen2` APIs before deployment.
