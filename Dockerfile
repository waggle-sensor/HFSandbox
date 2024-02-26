# syntax=docker/dockerfile:1

FROM nvcr.io/nvidia/pytorch:24.01-py3

COPY requirements.txt requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt


