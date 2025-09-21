# Microsoft Azure AI Vs Open Source vs GCP vs AWS 
<p> Alternatives for Modern AI Workflows


| SNo. | MS Tool/Service             | OSS 1 (Rank)             | OSS 2           | OSS 3           | OSS 4                  | OSS 5       | GCP Equivalent             | AWS Equivalent               |
|------|----------------------------|--------------------------|-----------------|-----------------|------------------------|-------------|----------------------------|------------------------------|
| 1    | Azure OpenAI Service       | Hugging Face Transformers| Ollama          | LlamaIndex      | BricksLLM              | FastChat    | Vertex AI                  | Bedrock, SageMaker           |
| 2    | Semantic Kernel            | LangChain                | Haystack        | LlamaIndex      | CrewAI                 | AutoGen     | Vertex AI Agent Builder    | Bedrock, Agents for Bedrock  |
| 3    | Azure AI Foundry           | TensorFlow Hub           | Kubeflow        | Hugging Face    | PyTorch                | FastAI      | Vertex AI Workbench        | SageMaker                    |
| 4    | Azure AI Evaluation SDK    | Evidently AI    | RAGAS           | MLflow Evaluation  | DeepEval, BLEU, ROUGE  | OpenLMEval  | Vertex AI Model Evaluation                   | SageMaker Model Monitor                    |
| 5    | Azure AI Inference SDK     | DeepSpeed                | vLLM            | ONNX Runtime    | Hugging Face Inference | TGI         | Vertex AI                  | SageMaker                    |
| 6    | Azure AI Agent Service     | AutoGen                   | CrewAI        | FastAgency      | Autogen                | Langroid    | Agent Builder (Vertex AI)   | Agents for Bedrock           |
| 7    | Azure Content Safety       | Detoxify                 | Perspective API | WebPurify API   | OpenAI Moderation  API    | Aestron     | Content Safety API (Vertex AI)| AWS Content Moderation    |
| 8    | Azure Language Service     | spaCy                    | Hugging Face NLP| NLTK            | Stanza                 | AllenNLP    | Cloud Translation API       | Comprehend, Translate        |
| 9    | Azure AI Search            | Meilisearch              | Elasticsearch   | Qdrant          | Apache Solr            | Lucene      | Vertex AI Search            | OpenSearch                   |
| 10   | Azure Document Intelligence| Tesseract OCR               | EasyOCR      | PaddleOCR| PaddleOCR              | EasyOCR     | Document AI (GCP)           | Textract                     |



Other supporting services
--------------------------
- Azure Functions
- Azure Kubernetes Service (AKS)
- Azure Container Apps
- Azure App Service
- Azure Logic Apps
- Azure Service Bus
- Azure Event Grid
- Azure Cosmos DB
- Azure Storage
- Azure Key Vault
- Azure Monitor
- Azure Active Directory
- Azure Redis Cache

Vector Databases
--------------------------
- Dedicated Commeercial Vector DB:
    - Pinecone - used by companies like MS, Shopify, Notion
    - DataSax
    - Zilliz cloud (managed milvus offering)
- Dedicated OSS Commercial Vector DB:
    - Weaviate - hybrid search, uses GraphQL API
    - Milvus - used by Salesforce, IKEA, Paypal
    - Chroma - startups, smb
    - ElasticSearch
    - Qdrant - Bosch, Mozilla, Perplexity
    - Faiss - Facebook AI similarity search
    - Vespa - by yahoo, lexical search
    - Deep Lake
    - Vald
    - Valkey
    - Marqo
    - LanceDB
- Traditiona databases with vector capabilities:
    - PostgrSQL with pgVector
    - OpenSearch
    - Apache Cassandra
    - ClickHouse
    - Redis
    - SingleStoreDB
    - MongoDB Atlas Vector Search
    - CockroachDB
    - elasticsearch

=================================================================
- Azure Vector DB options
    - Azure Cosmos DB for NoSQL - Native DiskANN indexing
    - Azure Cosmos DB for MongoDB - vector search with embeddings
    - Azure Cosmos for PostGreSQL - Native vector data type (preview)
    - Azure Database for PostGreSQL - pgvector extension
    - Azure Search AI - Built in vector search
    - Azure SQL Database
- AWS
    - Amazon S3 vectors
    - Aurora postgreSQL - HNSW
    - Amazon RDS for PostgreSQL - pgvector extension
    - Amazon OpenSearch Service - Native k-NN
    - Amazon Kendra - Built in intelligent search and vector capabilities
    - Amazon MemoryDB in Redis
- Google
    - AlloyDB for PostGreSQL - pgvector - scanNN indexing (scalable nearest neighbours)
    - Cloud SQL for PostGreSQL - pgvector extension
    - Vertex AI Vector search
    - Firestore - vector embeddings + vertex ai integration
    - Bigtable - vector embeddings + vertext ai integration
    - Cloud Spanner - vectors columns wiht SQL extensions
      
=================================================================

- For text data embedding - create vector representations of words and sentences
    - OpenAI’s text-embedding-3
	- Google Gemini’s textembedding-gecko models
	- Cohere’s Embed models
	- SentenceTransformers
	- FastText
	- GloVe
	- Word2Vec
- For image data - CNNs like VGG, Inception or CLIP from OpenAI - extract feature vectors from images 
- For audio data - Spectrogram
=================================================================

**Aug/19/2025**
- Companies having universal API endpoint - abstraction framework to handle LLM's in the backend 
    - (1) OpenRouter - managed AI gateway focused on providing unified access to multiple LLM providers
      	- aggregates 400+ models from 50+ providers
      	- single API endpoint with OpenAI-compatible interface
      	- provider abstraction layer handles authentication, billing, and routing
	- (2) TreuFactory - comprehensive MLOps platform that includes an AI gateway as part of a broader ML lifecycle management solution
   		- kubernetes-native platform deployed on your infrastructure
    - (3) Microsoft AI Foundry Universal API Design
      	- easy model switching without code changes
      	- serves as the gateway to 1900+ models from multiple providers
      	- a single inference endpoint
      	- eliminates the need for developers to manage separate endpoints, authentication schemes, or API contracts for different model providers
    - (4) LiteLLM - Multi-cloud flexibility, often cheaper, community-driven
=================================================================

**Sep/20/2025**
- Programming : Python, Jupyter labs, Hugging face, Pytorch, TensorFlow
- Monitoring: Streamlit, TensorBoard, Grafana
=================================================================

**Sep/21/2025**
- Agentic coding
   - IDE Extensions - Github CoPilot, Cursor, Cline
   - CLI toosl - Claude code, codex-cli, gemini-cli, aider, opencode
   - Task specific tooks - coderabbit for code review
   - vide-coding apps - lovable, bolt, v0
     
=================================================================
