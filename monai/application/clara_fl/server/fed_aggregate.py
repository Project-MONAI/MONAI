"""Provides a command line interface for federated aggregation server"""

import argparse
import json
import logging.config
import os
import time

from monai.application.clara_fl.trainers.server_trainer import ServerTrainer
from monai.application.clara_fl.argument_utils import parse_vars

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mmar', '-m', type=str, help='MMAR_ROOT folder', required=True)

    parser.add_argument('--server_config', '-s', type=str,
                        help='an aggregation server specification json file', required=True)

    parser.add_argument('--log_config', '-l', type=str, help='log config file')

    parser.add_argument("--set", metavar="KEY=VALUE", nargs='*')

    args = parser.parse_args()
    parsed_vars = parse_vars(args.set)
    for key, value in parsed_vars.items():
        if key.startswith('MMAR_') and value != '':
            parsed_vars[key] = os.path.join(args.mmar, value)

    if args.log_config:
        log_config = args.log_config
    else:
        log_config = os.path.join(args.mmar, 'resources/log.config')

    assert os.path.isfile(log_config), 'missing log config file {}'.format(log_config)
    logging.config.fileConfig(fname=log_config)
    logger = logging.getLogger(__name__)

    start_time = time.time()

    server_config = {}

    # loading server specifications
    try:
        with open(os.path.join(args.mmar, args.server_config), 'r') as f:
            server_config.update(json.load(f))
            # update the SSL certs with mmar root
            for server in server_config['servers']:
                if server.get('ssl_private_key'):
                    server['ssl_private_key'] = os.path.join(args.mmar, server['ssl_private_key'])
                if server.get('ssl_cert'):
                    server['ssl_cert'] = os.path.join(args.mmar, server['ssl_cert'])
                if server.get('ssl_root_cert'):
                    server['ssl_root_cert'] = os.path.join(args.mmar, server['ssl_root_cert'])
    except Exception:
        raise ValueError("Server config error: '{}'".format(args.log_config))

    trainer = ServerTrainer(parsed_vars)
    trainer.set_server_config(server_config['servers'])
    trainer.train()
    trainer.close()

    end_time = time.time()
    logger.info('Total Training Time {}'.format(end_time - start_time))


if __name__ == '__main__':

    main()
