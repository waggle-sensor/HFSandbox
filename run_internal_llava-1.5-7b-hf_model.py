# source
# https://huggingface.co/llava-hf/vip-llava-13b-hf
import os
import csv
import shutil
import requests
from glob import glob
from PIL import Image
import time
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, VipLlavaForConditionalGeneration


def main():
    image_file="http://images.cocodataset.org/val2017/000000039769.jpg"
    user_question='What are these?'
    user_prompt="A chat between a curious human and an artificial intelligence assistant. The assistant gives detailed, and polite answers to the human's questions."
    prompt = "USER: <image>\n" + user_prompt + "\nASSISTANT:"
    raw_image = Image.open(requests.get(image_file, stream=True).raw)

    model_id = "llava-hf/llava-1.5-7b-hf"

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        output_hidden_states=True
    )
    model.zero_grad()

    processor = AutoProcessor.from_pretrained(model_id)
    inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    print(output.shape)
    output = model(input_ids = inputs['input_ids'], pixel_values = inputs['pixel_values'], attention_mask=inputs['attention_mask'], output_hidden_states=True)
    print(type(output))
    print(output.keys())
    print(len(output.hidden_states))
    for hidden_state in output.hidden_states:
        #print(hidden_state.shape)
        print(hidden_state.mean(1).shape)
    #print(type(output['image_hidden_states']))
    ##output = model.generate(**inputs, max_new_tokens=200, do_sample=False)





# Using the special variable  
# __name__ 
if __name__=="__main__": 
    main() 

