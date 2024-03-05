This subrepo is adapted from [llama-recipes](https://github.com/facebookresearch/llama-recipes)

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
For vanilla time to train run:
```sh
torchrun --nnodes 1 --nproc_per_node 8 examples/finetuning.py --enable_fsdp --model_name ./{}B --fsdp_config.pure_bf16 --output_dir ./saves
```
To run with [deepspeed profiling](https://deepspeed.readthedocs.io/en/latest/flops-profiler.html):
```sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -m pip install deepspeed
torchrun --nnodes 1 --nproc_per_node 8 examples/finetuning.py --enable_fsdp --model_name ./{}B --fsdp_config.pure_bf16 --batch_size_training {} --mult {} --profiler --output_dir ./saves
```
where `batch_size_training` can be tuned for optimization, and `mult` accordingly for an appropriate number of batches
