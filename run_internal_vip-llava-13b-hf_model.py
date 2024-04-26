# source
# https://huggingface.co/llava-hf/vip-llava-13b-hf
import os
import csv
import shutil
import requests
from random import shuffle
from glob import glob
from PIL import Image
import time
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, VipLlavaForConditionalGeneration


def main():
    model_id = "llava-hf/vip-llava-13b-hf"

    model = VipLlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        output_hidden_states=True
    )

    processor = AutoProcessor.from_pretrained(model_id)


    directory='/images'
    files = [y for x in os.walk(directory) for y in glob(os.path.join(x[0], '*.jpg'))]
    shuffle(files)

    #os.makedirs('/RESULTS/FIRE_IMAGES/', mode=0o777, exist_ok=True)
    CHECKPOINT, RESULTS, INTERNAL_REP, RGB_INTERNAL_REP, IR_INTERNAL_REP = load_checkpoint()

    counter=len(CHECKPOINT)
    for FILE in files:
        if FILE not in CHECKPOINT:
            user_prompt="Is there fire or not in the image?"
            user_question="Please generate the number 1 if there is fire in the image, otherwise generate the number 0. Only one number has to be generated in your response."

            image_file=FILE
            pattern='###Assistant:'
            response, time=run_llava(model, processor, user_question, user_prompt, image_file)
            internal_rep, time=internal_run_llava(model, processor, user_question, user_prompt, image_file)
            rgb_response, time=rgb_run_llava(model, processor, user_question, user_prompt, image_file)
            rgb_internal_rep, rgb_time=internal_rgb_run_llava(model, processor, user_question, user_prompt, image_file)
            ir_response, time=ir_run_llava(model, processor, user_question, user_prompt, image_file)
            ir_internal_rep, ir_time=internal_ir_run_llava(model, processor, user_question, user_prompt, image_file)

            response=response.split(pattern, 1)[1]
            rgb_response=rgb_response.split(pattern, 1)[1]
            ir_response=ir_response.split(pattern, 1)[1]
            print(FILE)
            print(counter)
            print(response)
            print('--------------->>>>>>>>>>>>>>>>>>')
            RESULTS.append(FILE + ',' + response + ',' + rgb_response + ',' + ir_response)

            CHECKPOINT.append(FILE)
            INTERNAL_REP.append(internal_rep.to('cpu').detach())
            RGB_INTERNAL_REP.append(rgb_internal_rep.to('cpu').detach())
            IR_INTERNAL_REP.append(ir_internal_rep.to('cpu').detach())

            counter += 1
            if counter%10 == 0:
                save_results(CHECKPOINT=CHECKPOINT, RESULTS=RESULTS, INTERNAL_REP=INTERNAL_REP, RGB_INTERNAL_REP=RGB_INTERNAL_REP, IR_INTERNAL_REP=IR_INTERNAL_REP)
        else:
            print('-------------------->>>>>>>>>>>>>>>>>>>>')
            print('-------------------->>>>>>>>>>>>>>>>>>>>')
            print('-------------------->>>>>>>>>>>>>>>>>>>>')
            print('-------------------->>>>>>>>>>>>>>>>>>>>')
            print('File is in checkpoint!!!!!!!!!!')
            print('-------------------->>>>>>>>>>>>>>>>>>>>')
            print('-------------------->>>>>>>>>>>>>>>>>>>>')
            print('-------------------->>>>>>>>>>>>>>>>>>>>')
            print('-------------------->>>>>>>>>>>>>>>>>>>>')

    save_results(CHECKPOINT=CHECKPOINT, RESULTS=RESULTS, INTERNAL_REP=INTERNAL_REP, RGB_INTERNAL_REP=RGB_INTERNAL_REP, IR_INTERNAL_REP=IR_INTERNAL_REP)





def load_checkpoint(checkpoint_path='/RESULTS/checkpoint.txt',
                    output_path='/RESULTS/output.csv',
                    internal_rep_path='/RESULTS/internal_rep.pt',
                    rgb_internal_rep_path='/RESULTS/rgb_internal_rep.pt',
                    ir_internal_rep_path='/RESULTS/ir_internal_rep.pt'):
    CHECKPOINT=[]
    RESULTS=[]
    INTERNAL_REP=[]
    RGB_INTERNAL_REP=[]
    IR_INTERNAL_REP=[]
    if os.path.isfile(checkpoint_path):
        with open(checkpoint_path) as file:
            CHECKPOINT = [line.rstrip() for line in file]

    if os.path.isfile(output_path):
        with open(output_path) as file:
            RESULTS = [line.rstrip() for line in file]

    if os.path.isfile(internal_rep_path):
        aux=torch.load(internal_rep_path)
        for element in aux:
            INTERNAL_REP.append(element.unsqueeze(0))

    if os.path.isfile(rgb_internal_rep_path):
        aux=torch.load(rgb_internal_rep_path)
        for element in aux:
            RGB_INTERNAL_REP.append(element.unsqueeze(0))

    if os.path.isfile(ir_internal_rep_path):
        aux=torch.load(ir_internal_rep_path)
        for element in aux:
            IR_INTERNAL_REP.append(element.unsqueeze(0))


    return CHECKPOINT, RESULTS, INTERNAL_REP, RGB_INTERNAL_REP, IR_INTERNAL_REP





def save_results(CHECKPOINT, RESULTS, INTERNAL_REP, RGB_INTERNAL_REP, IR_INTERNAL_REP,
                 checkpoint_path='/RESULTS/checkpoint.txt',
                 output_path='/RESULTS/output.csv',
                 internal_rep_path='/RESULTS/internal_rep.pt',
                 rgb_internal_rep_path='/RESULTS/rgb_internal_rep.pt',
                 ir_internal_rep_path='/RESULTS/ir_internal_rep.pt'):

    with open(checkpoint_path, 'w') as f:
        for file_line in CHECKPOINT:
            f.write(f"{file_line}\n")

    os.chmod(checkpoint_path, 0o666)

    if len(RESULTS) > 0:
        with open(output_path,'w') as result_file:
            for file_line in RESULTS:
                result_file.write(f"{file_line}\n")

        os.chmod(output_path, 0o666)

    with torch.no_grad():
        if len(INTERNAL_REP) > 0:
            aux = torch.Tensor(len(INTERNAL_REP), INTERNAL_REP[0].shape[0], INTERNAL_REP[0].shape[1]).to(INTERNAL_REP[0].device)
            torch.cat(INTERNAL_REP, out=aux)
            torch.save(aux, internal_rep_path)
            os.chmod(internal_rep_path, 0o666)

        if len(RGB_INTERNAL_REP) > 0:
            aux = torch.Tensor(len(RGB_INTERNAL_REP), RGB_INTERNAL_REP[0].shape[0], RGB_INTERNAL_REP[0].shape[1]).to(RGB_INTERNAL_REP[0].device)
            torch.cat(RGB_INTERNAL_REP, out=aux)
            torch.save(aux, rgb_internal_rep_path)
            os.chmod(rgb_internal_rep_path, 0o666)

        if len(IR_INTERNAL_REP) > 0:
            aux = torch.Tensor(len(IR_INTERNAL_REP), IR_INTERNAL_REP[0].shape[0], IR_INTERNAL_REP[0].shape[1]).to(IR_INTERNAL_REP[0].device)
            torch.cat(IR_INTERNAL_REP, out=aux)
            torch.save(aux, ir_internal_rep_path)
            os.chmod(ir_internal_rep_path, 0o666)


def run_llava(model, processor,
              user_question='What are these?',
              user_prompt="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
              image_file="http://images.cocodataset.org/val2017/000000039769.jpg"):

    prompt = f"{user_prompt}.###Human: <image>\n{user_question}###Assistant:"
    #prompt = f"{user_question}.###Human: <image>\n{user_prompt}###Assistant:"

    #raw_image = Image.open(requests.get(image_file, stream=True).raw)
    raw_image = Image.open(image_file)
    inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

    start_t = time.time()
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    end_t = time.time()
    #print(processor.decode(output[0][2:], skip_special_tokens=True))
    #print('Time elapsed: ', end_t-start_t) 
    return processor.decode(output[0][2:], skip_special_tokens=True), end_t-start_t 


def internal_run_llava(model, processor,
              user_question="Can you explain the image in detail?",
              user_prompt="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
              image_file="http://images.cocodataset.org/val2017/000000039769.jpg"):

    #prompt = "USER: <image>\n" + user_prompt + "\nASSISTANT:"
    prompt = f"{user_prompt}.###Human: <image>\n{user_question}###Assistant:"

    raw_image = Image.open(image_file)
    #raw_image = raw_image.resize((int(raw_image.size[0]/8),int(raw_image.size[1]/8)))
    inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

    with torch.no_grad():
        start_t = time.time()
        output = model(input_ids = inputs['input_ids'], pixel_values = inputs['pixel_values'], attention_mask=inputs['attention_mask'], output_hidden_states=True)
        end_t = time.time()

        return output.hidden_states[-1].mean(1), end_t-start_t 


def rgb_run_llava(model, processor,
              user_question='What are these?',
              user_prompt="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
              image_file="http://images.cocodataset.org/val2017/000000039769.jpg"):

    #prompt = f"{user_question}.###Human: <image>\n{user_prompt}###Assistant:"
    prompt = f"{user_prompt}.###Human: <image>\n{user_question}###Assistant:"

    #raw_image = Image.open(requests.get(image_file, stream=True).raw)
    im = Image.open(image_file)
    width, height = im.size
    left = 0
    top = 0
    right = width / 2
    bottom = height
    raw_image = im.crop((left, top, right, bottom))

    inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

    start_t = time.time()
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    end_t = time.time()
    #print(processor.decode(output[0][2:], skip_special_tokens=True))
    #print('Time elapsed: ', end_t-start_t) 
    return processor.decode(output[0][2:], skip_special_tokens=True), end_t-start_t 



def internal_rgb_run_llava(model, processor,
                  user_question="Can you explain the image in detail?",
                  user_prompt="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
                  image_file="http://images.cocodataset.org/val2017/000000039769.jpg"):

    #prompt = "USER: <image>\n" + user_prompt + "\nASSISTANT:"
    prompt = f"{user_prompt}.###Human: <image>\n{user_question}###Assistant:"

    im = Image.open(image_file)
    #im = im.resize((int(im.size[0]/8),int(im.size[1]/8)))
    width, height = im.size
    left = 0
    top = 0
    right = width / 2
    bottom = height
    raw_image = im.crop((left, top, right, bottom))

    with torch.no_grad():
        inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

        start_t = time.time()
        output = model(input_ids = inputs['input_ids'], pixel_values = inputs['pixel_values'], attention_mask=inputs['attention_mask'], output_hidden_states=True)
        end_t = time.time()

    return output.hidden_states[-1].mean(1), end_t-start_t 



def ir_run_llava(model, processor,
                 user_question='What are these?',
                 user_prompt="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
                 image_file="http://images.cocodataset.org/val2017/000000039769.jpg"):

    #prompt = f"{user_question}.###Human: <image>\n{user_prompt}###Assistant:"
    prompt = f"{user_prompt}.###Human: <image>\n{user_question}###Assistant:"

    im = Image.open(image_file)
    width, height = im.size
    left = width / 2
    top = 0
    right = width
    bottom = height
    raw_image = im.crop((left, top, right, bottom))

    inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

    start_t = time.time()
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    end_t = time.time()

    return processor.decode(output[0][2:], skip_special_tokens=True), end_t-start_t 




def internal_ir_run_llava(model, processor,
                 user_question="Can you explain the image in detail?",
                 user_prompt="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
                 image_file="http://images.cocodataset.org/val2017/000000039769.jpg"):

    #prompt = "USER: <image>\n" + user_prompt + "\nASSISTANT:"
    prompt = f"{user_prompt}.###Human: <image>\n{user_question}###Assistant:"

    im = Image.open(image_file)
    #im = im.resize((int(im.size[0]/8),int(im.size[1]/8)))
    width, height = im.size
    left = width / 2
    top = 0
    right = width
    bottom = height
    raw_image = im.crop((left, top, right, bottom))

    with torch.no_grad():
        inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

        start_t = time.time()
        output = model(input_ids = inputs['input_ids'], pixel_values = inputs['pixel_values'], attention_mask=inputs['attention_mask'], output_hidden_states=True)
        end_t = time.time()

    return output.hidden_states[-1].mean(1), end_t-start_t 





# Using the special variable  
# __name__ 
if __name__=="__main__": 
    main() 



