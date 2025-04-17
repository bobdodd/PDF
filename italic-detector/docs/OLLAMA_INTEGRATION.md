# Integrating with Ollama

This guide explains how to integrate the italic detector with Ollama for use in AI applications.

## Installing Ollama

First, you need to install Ollama on your system:

1. Visit [ollama.ai](https://ollama.ai) to download the Ollama application
2. Follow their installation instructions for your operating system
3. Verify Ollama is installed by running `ollama --version` in your terminal

## Deploying the Model to Ollama

After training your model, you can deploy it to Ollama:

```bash
italic-detector deploy-to-ollama --model-name italic-detector --base-model llama2
```

This will:
1. Create a Modelfile with your trained model
2. Build an Ollama model that can analyze text features
3. Make the model available through Ollama's API

## Using the Deployed Model

Once deployed, you can use the model in two ways:

### 1. Command-line Interface

```bash
# Analyze a single image
italic-detector detect-italic --image-path sample.png --model-name italic-detector

# Process an entire document
italic-detector process-document --pdf-path document.pdf --model-name italic-detector
```

### 2. Direct Ollama API

You can also use the Ollama API directly to get predictions:

```bash
# Using the Ollama CLI
ollama run italic-detector "Analyze these features extracted from a text image: [0.1, 0.2, 0.3, ...]. Based on these features, is the text italic?"
```

## Customizing the Model

You can customize the Ollama model by modifying the Modelfile before building.

1. Generate a Modelfile without building:
```bash
# This will create a Modelfile but not build it
python -c "from src.ollama_integration import OllamaIntegration; OllamaIntegration().create_modelfile('Modelfile')"
```

2. Edit the Modelfile as needed
3. Build the model using the Ollama CLI:
```bash
ollama create italic-detector -f Modelfile
```

## Troubleshooting

If you encounter issues with Ollama integration:

1. Verify Ollama is running with `ollama list`
2. Check model files in `models/saved/` directory
3. Look for error messages during model deployment
4. Try a different base model (`--base-model mistral` for example)

## Using Other Base Models

You can use different base LLMs for your Ollama model:

```bash
italic-detector deploy-to-ollama --model-name italic-detector --base-model mistral
```

Available base models depend on your Ollama installation. Run `ollama list` to see available models.