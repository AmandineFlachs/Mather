![Banner](./deploy/assets/banner.png)

Access to educative content can be limited in certain regions of the world and parents can feel overwhelmed when helping their children, especially in the field of mathematics. Private tutors are not always available or can be too expensive.

We developed Mather, a Large Language Model (LLM) that serves as a mathematics tutor. Users can ask Mather questions and it will provide individualized answers and guidance.

Mather is a fine-tune of Mistral-7B-Instruct-v0.3, trained on several mathematics datasets (see below). It has been trained on 8x AMD MI210 on an AMD Accelerator Cloud node leveraging ROCm 6.1.1. The model can be directly used locally, without an internet connection, via a dedicated dialog user interface.

## Repository

Our repository includes the following files:

```bash
├── deploy
│   ├── assets
│   │   └── banner.png
│   ├── requirements.txt
│   └── streamlit_app.py
├── LICENSE
├── README.md
└── train
    ├── finetune.py
    └── merge.py
```

There are two main directories:
- *deploy* contains the code to run Mather-v1 locally.
- *train* contains the code to replicate our Mather-v1 model.

## Run Mather-v1 locally

An 8-bit quantized version of our model is hosted on Hugging Face: https://huggingface.co/AmandineFlachs/Mather-v1-gguf

- Install [LM Studio](https://lmstudio.ai/) and [Streamlit](https://docs.streamlit.io/get-started/installation).

- Run our Mather-v1 LLM locally with LM Studio, like any other model hosted on Hugging Face.

- Clone our repository on GitHub and go to the *deploy* folder.

- Create a virtual environment and install our Python dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

- Run our dialog interface for Mather-v1:

```bash
streamlit run streamlit_app.py
```

This will open a new tab in your  default browser, which should look like the screenshot below. You can  easily interact with the model we trained. On the left panel you can  specify whether you prefer concise or detailed answers.

Also, note that our implementation is easy to customize to allow the community to modify it at their will.

## Train Mather-v1

We fine-tuned [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) for 1 epoch on 3 datasets: [MathInstruct](https://huggingface.co/datasets/agicorp/MathInstruct), [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) and [GSM8K](https://huggingface.co/datasets/openai/gsm8k) (train split only, following best practices in LLM  training). Training took 5 hours on 8x AMD MI210 64GB.

To replicate our model, please install dependencies following the instructions provided [here](https://github.com/amd/GenAI-contest/tree/main/01-LLM_Fine-tuning) by AMD. In the *train* folder, there are 2 main scripts to generate a model: *finetune.py* trains a low-rank adapter (LoRA), which then can be merged with the original model using *merge.py*. Follow these instructions to use the 2 scripts:

- To fine-tune the model on 1x AMD MI210:

```bash
python3 finetune.py
```

- To fine-tune the model on 8x AMD MI210:

```bash
OMP_NUM_THREADS=8 python3 -m torch.distributed.launch --nproc_per_node=8 finetune.py
```

- To merge the LoRA adapter that the fine-tuning script generated with the base model, run:

```bash
python3 merge.py
```

- Additionally, the tokenizer configuration files from Mistral-7B-Instruct-v0.3 need to be copied to the merged model folder.

- Optionally, to deploy the model you can then generate a quantized GGUF model for efficient inference by installing [llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md) and running:

```bash
python llama.cpp/convert_hf_to_gguf.py --outfile mather-v1.gguf --outtype q8_0
```

## The Team

- Amandine Flachs

- Alexandre Borghi

## License

Mather (code) has an Apache 2.0 license, as found in the [LICENSE](LICENSE) file.

Mather-v1 (weights) also has an Apache 2.0 license.
