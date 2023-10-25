import json
import os
import time

import requests
import yaml

from client import ApiClient

'''
This client is a generic client for any Grobid application and sub-modules.
At the moment, it supports only single document processing.

Source: https://github.com/kermitt2/grobid-client-python 
'''


class GrobidClientGeneric(ApiClient):

    def __init__(self, config_path=None, ping=False):
        self.config = None
        if config_path is not None:
            self.config = self.load_yaml_config_from_file(path=config_path)
            super().__init__(self.config['grobid']['server'])

            if ping:
                result = self.ping_grobid()
                if not result:
                    raise Exception("Grobid is down.")

        os.environ['NO_PROXY'] = "nims.go.jp"

    @staticmethod
    def load_json_config_from_file(self, path='./config.json', ping=False):
        """
        Load the json configuration
        """
        config = {}
        with open(path, 'r') as fp:
            config = json.load(fp)

        if ping:
            result = self.ping_grobid()
            if not result:
                raise Exception("Grobid is down.")

        return config

    def load_yaml_config_from_file(self, path='./config.yaml'):
        """
        Load the YAML configuration
        """
        config = {}
        try:
            with open(path, 'r') as the_file:
                raw_configuration = the_file.read()

            config = yaml.safe_load(raw_configuration)
        except Exception as e:
            print("Configuration could not be loaded: ", str(e))
            exit(1)

        return config

    def set_config(self, config, ping=False):
        self.config = config
        if ping:
            try:
                result = self.ping_grobid()
                if not result:
                    raise Exception("Grobid is down.")
            except Exception as e:
                raise Exception("Grobid is down or other problems were encountered. ", e)

    def ping_grobid(self):
        # test if the server is up and running...
        ping_url = self.get_grobid_url("ping")

        r = requests.get(ping_url)
        status = r.status_code

        if status != 200:
            print('GROBID server does not appear up and running ' + str(status))
            return False
        else:
            print("GROBID server is up and running")
            return True

    def get_grobid_url(self, action):
        grobid_config = self.config['grobid']
        base_url = grobid_config['server']
        action_url = base_url + grobid_config['url_mapping'][action]

        return action_url

    def process_texts(self, input, method_name='superconductors', params={}, headers={"Accept": "application/json"}):

        files = {
            'texts': input
        }

        the_url = self.get_grobid_url(method_name)
        params, the_url = self.get_params_from_url(the_url)

        res, status = self.post(
            url=the_url,
            files=files,
            data=params,
            headers=headers
        )

        if status == 503:
            time.sleep(self.config['sleep_time'])
            return self.process_texts(input, method_name, params, headers)
        elif status != 200:
            print('Processing failed with error ' + str(status))
            return status, None
        else:
            return status, json.loads(res.text)

    def process_text(self, input, method_name='superconductors', params={}, headers={"Accept": "application/json"}):

        files = {
            'text': input
        }

        the_url = self.get_grobid_url(method_name)
        params, the_url = self.get_params_from_url(the_url)

        res, status = self.post(
            url=the_url,
            files=files,
            data=params,
            headers=headers
        )

        if status == 503:
            time.sleep(self.config['sleep_time'])
            return self.process_text(input, method_name, params, headers)
        elif status != 200:
            print('Processing failed with error ' + str(status))
            return status, None
        else:
            return status, json.loads(res.text)

    def process(self, form_data: dict, method_name='superconductors', params={}, headers={"Accept": "application/json"}):

        the_url = self.get_grobid_url(method_name)
        params, the_url = self.get_params_from_url(the_url)

        res, status = self.post(
            url=the_url,
            files=form_data,
            data=params,
            headers=headers
        )

        if status == 503:
            time.sleep(self.config['sleep_time'])
            return self.process_text(input, method_name, params, headers)
        elif status != 200:
            print('Processing failed with error ' + str(status))
        else:
            return res.text

    def process_pdf_batch(self, pdf_files, params={}):
        pass

    def process_pdf(self, pdf_file, method_name, params={}, headers={"Accept": "application/json"}, verbose=False,
                    retry=None):

        files = {
            'input': (
                pdf_file,
                open(pdf_file, 'rb'),
                'application/pdf',
                {'Expires': '0'}
            )
        }

        the_url = self.get_grobid_url(method_name)

        params, the_url = self.get_params_from_url(the_url)

        res, status = self.post(
            url=the_url,
            files=files,
            data=params,
            headers=headers
        )

        if status == 503 or status == 429:
            if retry is None:
                retry = self.config['max_retry'] - 1
            else:
                if retry - 1 == 0:
                    if verbose:
                        print("re-try exhausted. Aborting request")
                    return None, status
                else:
                    retry -= 1

            sleep_time = self.config['sleep_time']
            if verbose:
                print("Server is saturated, waiting", sleep_time, "seconds and trying again. ")
            time.sleep(sleep_time)
            return self.process_pdf(pdf_file, method_name, params, headers, verbose=verbose, retry=retry)
        elif status != 200:
            desc = None
            if res.content:
                c = json.loads(res.text)
                desc = c['description'] if 'description' in c else None
            return desc, status
        elif status == 204:
            # print('No content returned. Moving on. ')
            return None, status
        else:
            return res.text, status

    def get_params_from_url(self, the_url):
        params = {}
        if "?" in the_url:
            split = the_url.split("?")
            the_url = split[0]
            params = split[1]

            params = {param.split("=")[0]: param.split("=")[1] for param in params.split("&")}
        return params, the_url

    def process_json(self, text, method_name="processJson", params={}, headers={"Accept": "application/json"},
                     verbose=False):
        files = {
            'input': (
                None,
                text,
                'application/json',
                {'Expires': '0'}
            )
        }

        the_url = self.get_grobid_url(method_name)

        params, the_url = self.get_params_from_url(the_url)

        res, status = self.post(
            url=the_url,
            files=files,
            data=params,
            headers=headers
        )

        if status == 503:
            time.sleep(self.config['sleep_time'])
            return self.process_json(text, method_name, params, headers), status
        elif status != 200:
            if verbose:
                print('Processing failed with error ', status)
            return None, status
        elif status == 204:
            if verbose:
                print('No content returned. Moving on. ')
            return None, status
        else:
            return res.text, status
