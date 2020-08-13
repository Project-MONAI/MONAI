"""Provides a command line interface for a federated client trainer"""

import argparse
import json
import logging.config
import os
import time
import sys
from fed_learn.client.admin import FedAdminAgent
from fed_learn.client.admin_msg_sender import AdminMessageSender
from fed_learn.client.cai import ClientAdminInterface

from ..trainers.client_trainer import ClientTrainer
from ..argument_utils import parse_vars


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mmar', '-m', type=str, help='MMAR_ROOT folder', required=True)

    parser.add_argument(
        '--client_config',
        '-s',
        type=str,
        help='an aggregation server specification json file',
        required=True)

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

    assert os.path.isfile(log_config), 'missing log config file {}'.format(
        log_config)
    logging.config.fileConfig(fname=log_config)
    logger = logging.getLogger(__name__)

    start_time = time.time()

    server_config = {}

    # loading server specifications
    try:
        with open(os.path.join(args.mmar, args.client_config), 'r') as f:
            server_config.update(json.load(f))
            # update the SSL certs with mmar root
            client = server_config['client']
            if client.get('ssl_private_key'):
                client['ssl_private_key'] = os.path.join(args.mmar, client['ssl_private_key'])
            if client.get('ssl_cert'):
                client['ssl_cert'] = os.path.join(args.mmar, client['ssl_cert'])
            if client.get('ssl_root_cert'):
                client['ssl_root_cert'] = os.path.join(args.mmar, client['ssl_root_cert'])
    except Exception:
        raise ValueError("Server config error: '{}'".format(args.log_config))

    logger.info('Starting aggregation server with config:')
    logger.info('{}'.format(args.client_config))

    trainer = ClientTrainer(
        uid=parsed_vars['uid'],
        num_epochs=server_config['client']['local_epochs'],
        server_config=server_config['servers'],
        client_config=server_config['client'],
        secure_train=parsed_vars['secure_train']
    )

    federated_client = trainer.create_fed_client()
    federated_client.register()
    federated_client.start_heartbeat()
    servers = [{t['name']: t['service']} for t in trainer.server_config]
    admin_agent = create_admin_agent(trainer.client_config, trainer.uid, sorted(servers)[0], federated_client, args)
    admin_agent.start()

    trainer.close()

    end_time = time.time()
    logger.info('Total Training Time {}'.format(end_time - start_time))

    sys.exit(0)


def create_admin_agent(client_args, client_id, server_args, federated_client, args):
    sender = AdminMessageSender(client_name=federated_client.token, root_cert=client_args['ssl_root_cert'],
                                ssl_cert=client_args['ssl_cert'], private_key=client_args['ssl_private_key'],
                                server_args=server_args, secure=secure_train)
    admin_agent = FedAdminAgent(client_name='admin_agent', sender=sender,
                                app_ctx=ClientAdminInterface(federated_client, federated_client.token,
                                                             sender, args))
    admin_agent.app_ctx.set_agent(admin_agent)

    return admin_agent


if __name__ == '__main__':

    main()
