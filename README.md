This is adapted from [llama-recipes](https://github.com/facebookresearch/llama-recipes)

Here are instructions for installation with python3.10
```sh
git clone https://github.com/facebookresearch/llama
git clone https://github.com/facebookresearch/llama-recipes
cd llama-recipes
pip install -U pip setuptools
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 -e .
git clone --no-checkout https://github.com/huggingface/transformers
cd transformers
git checkout tags/v4.37.2
python3 -m pip install protobuf
cd ../llama
./download.sh
cd ../transformers
mv ~/llama/llama-2-{}b/ ~/llama/{}B
python3 src/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir ~/llama --model_size {}B --output_dir ~/llama-recipes/{}B
cd ../llama-recipes
```
