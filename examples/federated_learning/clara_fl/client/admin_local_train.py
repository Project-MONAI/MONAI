"""Provides a command line interface for a federated client trainer"""

import argparse
import os
import sys
import shutil
import time

from dlmed.common.excepts import ConfigError, ErrorHandled
from dlmed.utils.argument_utils import parse_vars
from nvmidl.apps.fed_learn.fl_conf import FLClientConfiger
from fed_learn.client.admin import FedAdminAgent
from fed_learn.client.admin_msg_sender import AdminMessageSender
from fed_learn.client.cai import ClientAdminInterface
from ai4med.utils.multi_gpu_utils import multi_gpu_init, multi_gpu_get_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--workspace', '-m', type=str, help='WORKSPACE folder', required=True)

    parser.add_argument(
        '--fed_client',
        '-s',
        type=str,
        help='an aggregation server specification json file',
        required=True)

    parser.add_argument("--set", metavar="KEY=VALUE", nargs='*')

    args = parser.parse_args()

    args.mmar = args.workspace
    args.train_config = 'config/config_train.json'
    args.client_config = 'config/config_fed_client.json'
    args.env = 'config/environment.json'
    args.log_config = None

    try:
        remove_restart_file(args)
    except BaseException:
        print('Could not remove the restart.fl / shutdown.fl file.  Please check your system before starting FL.')
        sys.exit(-1)

    kv_list = parse_vars(args.set)
    multi_gpu = kv_list.get('multi_gpu', False)
    if multi_gpu:
        multi_gpu_init()
        rank = multi_gpu_get_rank()
    else:
        rank = 0

    try:
        os.chdir(args.workspace)
        if rank == 0:
            create_workspace(args)
        time.sleep(rank * 2)

        # trainer = WorkFlowFactory().create_client_trainer(train_configs, envs)
        conf = FLClientConfiger(mmar_root='/tmp/fl',
                                wf_config_file_name='config_train.json',
                                client_config_file_name=args.fed_client,
                                env_config_file_name='environment.json',
                                log_config_file_name='/tmp/fl/log.config',
                                kv_list=args.set)
        conf.configure()

        trainer = conf.trainer
        federated_client = trainer.create_fed_client()

        if rank == 0:
            federated_client.register()

        if trainer.multi_gpu:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            federated_client.token = comm.bcast(federated_client.token, root=0)

        federated_client.start_heartbeat()

        servers = [{t['name']: t['service']} for t in trainer.server_config]

        admin_agent = create_admin_agent(trainer.client_config, trainer.uid, trainer.req_processors,
                                         trainer.secure_train, sorted(servers)[0], federated_client, args, trainer.multi_gpu, rank)
        admin_agent.start()

        trainer.close()

    except ConfigError as ex:
        print('ConfigError:', str(ex))
    except ErrorHandled as ex:
        print('ErrorHandled:', str(ex))
        pass
    finally:
        # shutil.rmtree('/tmp/fl')
        pass

    sys.exit(0)


def remove_restart_file(args):
    restart_file = os.path.join(args.mmar, 'restart.fl')
    if os.path.exists(restart_file):
        os.remove(restart_file)
    restart_file = os.path.join(args.mmar, 'shutdown.fl')
    if os.path.exists(restart_file):
        os.remove(restart_file)


def create_workspace(args):
    if os.path.exists('/tmp/fl'):
        shutil.rmtree('/tmp/fl')
    startup = os.path.join(args.workspace, 'startup')
    shutil.copytree(startup, '/tmp/fl')

    with open('/tmp/fl/config_train.json', 'wt') as f:
        f.write(r'{"epochs": 1250, "train": {   "loss": {   "name": "Dice" },   "model": {"name": "SegAhnet", "args": {"num_classes": 2, "if_use_psp": false, "pretrain_weight_name": "", "plane": "z", "final_activation": "softmax", "n_spatial_dim": 3}},   "pre_transforms": [{"name": "LoadNifti", "args": {"fields": ["image", "label"]}}],   "image_pipeline": {"name": "SegmentationImagePipelineWithCache", "args": {"data_list_file_path": "", "data_file_base_dir": "", "data_list_key": "training", "output_crop_size": [96, 96, 96]}} }}')

    with open('/tmp/fl/environment.json', 'wt') as f:
        f.write(r'{"PROCESSING_TASK": "segmentation","MMAR_CKPT_DIR": "models"}')

    # with open('/tmp/fl/log.config', 'wt') as f:
    #     f.write("[loggers]\nkeys=root,modelLogger\n[handlers]\nkeys=consoleHandler\n[formatters]\nkeys=fullFormatter\n[logger_root]\nlevel=INFO\nhandlers=consoleHandler\n[logger_modelLogger]\nlevel=DEBUG\nhandlers=consoleHandler\nqualname=modelLogger\npropagate=0\n[handler_consoleHandler]\nclass=StreamHandler\nlevel=DEBUG\nformatter=fullFormatter\nargs=(sys.stdout,)\n[formatter_fullFormatter]\nformat=%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def create_admin_agent(client_args, client_id, req_processors, secure_train, server_args, federated_client, args, is_multi_gpu, rank):
        sender = AdminMessageSender(client_name=federated_client.token, root_cert=client_args['ssl_root_cert'],
                                    ssl_cert=client_args['ssl_cert'], private_key=client_args['ssl_private_key'],
                                    server_args=server_args,
                                    secure=secure_train,
                                    is_multi_gpu=is_multi_gpu,
                                    rank=rank)
        admin_agent = FedAdminAgent(client_name='admin_agent',
                                    sender=sender,
                                    app_ctx=ClientAdminInterface(federated_client, federated_client.token,
                                                                 sender, args, rank))
        admin_agent.app_ctx.set_agent(admin_agent)
        for processor in req_processors:
            admin_agent.register_processor(processor)

        return admin_agent
        # self.admin_agent.start()


if __name__ == '__main__':

    main()

