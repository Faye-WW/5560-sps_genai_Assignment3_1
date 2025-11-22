SPS GenAI API
1. Introduction
This assignment implements and deploys a **Generative Adversarial Network (GAN)** using **PyTorch**, integrated into a **FastAPI** application.  
It extends the Module 6 class activity by:  
a. Implementing a GAN architecture that matches the assignment specification.  
b. Training the model on the MNIST dataset to generate handwritten digits.  
c. Adding the trained GAN to a FastAPI-based REST API with Docker deployment.  

1.1 GAN Architecture  
**Generator**
- Input: Noise vector `(BATCH_SIZE, 100)`
- Fully connected → reshape to `(128, 7, 7)`
- `ConvTranspose2d(128→64, kernel=4, stride=2, padding=1)` → output `(64, 14, 14)`
  - Followed by `BatchNorm2d`, `ReLU`
- `ConvTranspose2d(64→1, kernel=4, stride=2, padding=1)` → output `(1, 28, 28)`
  - Followed by `Tanh`
**Discriminator**
- Input: Image `(1, 28, 28)`
- `Conv2d(1→64, kernel=4, stride=2, padding=1)` → output `(64, 14, 14)`
  - Followed by `LeakyReLU(0.2)`
- `Conv2d(64→128, kernel=4, stride=2, padding=1)` → output `(128, 7, 7)`
  - Followed by `BatchNorm2d`, `LeakyReLU(0.2)`
- Flatten → `Linear(128×7×7 → 1)` → single logit (real/fake)

2. Environment Setup      
Clone the repository   
```bash
git clone https://github.com/Faye-WW/5560-sps_genai_Assignment3.git
cd 5560-sps_genai_Assignment3
```
Create a virtual environment   
```bash
python3 -m venv .venv
source .venv/bin/activate       # Mac/Linux
.\.venv\Scripts\activate        # Windows
```
Install dependencies   
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

3. Training the Model
Pretrained Models
This repository already includes pretrained model weights:
- `models/cnn.pt` — trained on CIFAR-10 for image classification.
- `artifacts/generator.pt` — trained GAN generator for MNIST digit generation.

You do not need to retrain the models. However, if you wish to retrain manually, you can run:
```bash
python -m app.train_cnn
python -m gan.train_gan
```

4. Run the Server   
From the project root directory, run:  
`uvicorn app.main:app --reload`  
You should see something like:   
`Uvicorn running on http://127.0.0.1:8000`
You’ll see all endpoints, including:
/gan/health
/gan/sample
/classify/image
/embed/word, /embed/sentence


5. Test the API   
Go to http://127.0.0.1:8000/docs   
You can explore and test all endpoints interactively.  

6. Run with Docker  
Build image:  
```
docker build -t sps-genai .
```
Run container (mount model file):  
```
docker run -p 8000:8000 sps-genai
```
Access API:  
http://127.0.0.1:8000/docs  

7. Project Structure  
```text
sps_genai/
│
├── app/
│ ├── main.py # FastAPI entry point
│ ├── train_cnn.py
│ ├── inference.py # CNN image classifier endpoint
│ └── routers/
│   └── gan.py # GAN endpoints 
│
├── gan/
│ ├── models.py # Generator & Discriminator definitions
│ └── train_gan.py # Training script (MNIST dataset)
│
├── artifacts/
│ └── generator.pt # Trained GAN weights
│
├── helper_lib/ # Utility modules from class
├── models/ # CNN model weights
│
├── Dockerfile # Docker deployment
├── requirements.txt # Dependencies
├── README.md # This file
└── .gitignore
```

8. Notes  
Always activate the .venv environment before running the server.  
If you see (base) from Anaconda, deactivate it first:  
```bash
conda deactivate
source .venv/bin/activate
```