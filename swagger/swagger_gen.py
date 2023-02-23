#!/usr/bin/python3
import logging
import requests
from jinja2 import Environment, FileSystemLoader
from datetime import datetime
import argparse
import os
import json
import yaml
import time

def get_example(model_names,function):
    example = {}
    for model_name in model_names:
        with open(os.path.join('examples',model_name,function+".json"),'r') as json_file:
            example[model_name] = json.load(json_file)[model_name]
    return example

def get_notes(model_names_path):
    example = {}
    for model_name in model_names:
        with open(os.path.join(model_note_path),'r') as json_file:
            example[model_name] = json.load(json_file)[model_name]
    return example

def remove_type_prefix(models):
    for model in models:
        for i in range(0, len(model["input"])):
            model["input"][i]["data_type"] = model["input"][i]["data_type"].replace(
                "TYPE_", ""
            )

    return models

def main():
    logging.basicConfig(encoding='utf-8', level=logging.INFO)
    with open('./model_names.json','r') as f:
        model_names = json.load(f)
    serving_started = False
    wait_time = 10
    reverse_host = os.getenv("TRITON_SERVER_IP")
    reverse_port = os.getenv("TRITON_REVERSE_PROXY_SERVER_PORT")

    triton_url_template = "http://reverse-proxy:" + reverse_port + "/v2/models/{model}/config"

    while (not serving_started):
        url = triton_url_template.format(model=model_names[0])
        logging.info("Waiting for serving to start: " + url)
        try:
            r = requests.get(url)
            if (r.status_code >= 200 and r.status_code <= 299):
                serving_started = True
                logging.info("Serving started continuing the program")
            else:
                time.sleep(wait_time)
        except Exception as e:
            time.sleep(wait_time)

    models = []
    for model in model_names:
        url = triton_url_template.format(model=model)
        logging.info("The selected triton back-end to generate swagger from: " + url)
        r = requests.get(url)
        models.append(r.json())

    environment = Environment(loader=FileSystemLoader("./"))
    template = environment.get_template("swagger_tmpl.yml")

    example_output = get_example(model_names,function='output')
    example_input = get_example(model_names,function='input')
    context = {"models": models, "example_output": example_output,'example_input': example_input,'triton_server_ip':reverse_host,'triton_server_port':reverse_port}

    content = template.render(context)
    logging.info("Generating the Swagger YAML file ... ")
    with open("swagger_test.yml", mode="w", encoding="utf-8") as yaml:
        yaml.write(content)
    logging.info("Finished Generating the Swagger YAML file ... ")


if __name__ == "__main__":
    main()
