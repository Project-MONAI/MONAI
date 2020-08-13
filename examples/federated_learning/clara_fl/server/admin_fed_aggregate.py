"""Provides a command line interface for federated aggregation server"""

import argparse
import os
import shutil
import sys

from dlmed.common.excepts import ConfigError, ErrorHandled
from nvmidl.apps.fed_learn.fl_conf import FLServerConfiger
from fed_learn.server.admin import FedAdminServer
from fed_learn.server.sai import ServerAdminInterface


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', '-m', type=str, help='WORKSPACE folder', required=True)

    parser.add_argument('--fed_server', '-s', type=str,
                        help='an aggregation server specification json file', required=True)

    parser.add_argument("--set", metavar="KEY=VALUE", nargs='*')

    args = parser.parse_args()

    args.mmar = args.workspace
    args.train_config = 'config/config_train.json'
    args.server_config = 'config/config_fed_server.json'
    args.env = 'config/environment.json'
    args.log_config = None

    try:
        remove_restart_file(args)
    except BaseException:
        print('Could not remove the restart.fl / shutdown.fl file.  Please check your system before starting FL.')
        sys.exit(-1)

    try:
        os.chdir(args.workspace)
        create_workspace(args)
        # trainer = WorkFlowFactory().create_server_trainer(train_configs, envs)
        conf = FLServerConfiger(mmar_root='/tmp/fl_server',
                                wf_config_file_name='config_train.json',
                                server_config_file_name=args.fed_server,
                                env_config_file_name='environment.json',
                                log_config_file_name='/tmp/fl_server/log.config',
                                kv_list=args.set)
        conf.configure()

        trainer = conf.trainer

        try:
            # Deploy the FL server
            services = trainer.deploy()

            secure_train = conf.cmd_vars.get('secure_train', False)
            first_server = sorted(conf.wf_config_data['servers'])[0]
            admin_server = create_admin_server(services, server_conf=first_server, args=args,
                                               secure_train=secure_train, mmar_validator=trainer.mmar_validator)
            admin_server.start()

            services.set_admin_server(admin_server)
        finally:
            trainer.close()

        print('Server has been started.')

    except ConfigError as ex:
        print('ConfigError:', str(ex))
    except ErrorHandled as ex:
        print('ErrorHandled:', str(ex))
        pass
    finally:
        shutil.rmtree('/tmp/fl_server')


def remove_restart_file(args):
    restart_file = os.path.join(args.mmar, 'restart.fl')
    if os.path.exists(restart_file):
        os.remove(restart_file)
    restart_file = os.path.join(args.mmar, 'shutdown.fl')
    if os.path.exists(restart_file):
        os.remove(restart_file)


def create_admin_server(fl_server, server_conf=None, args=None, secure_train=False, mmar_validator=None):
        sai = ServerAdminInterface(fl_server, args)
        users = {}
        # cmd_modules = [ValidationCommandModule()]

        root_cert = server_conf['ssl_root_cert'] if secure_train else None
        server_cert = server_conf['ssl_cert'] if secure_train else None
        server_key = server_conf['ssl_private_key'] if secure_train else None
        admin_server = FedAdminServer(
                     fed_admin_interface=sai,
                     users=users,
                     cmd_modules=fl_server.cmd_modules,
                     file_upload_dir=os.path.join(args.mmar, server_conf.get('admin_storage', 'tmp')),
                     file_download_dir=os.path.join(args.mmar, server_conf.get('admin_storage', 'tmp')),
                     allowed_shell_cmds=None,
                     host=server_conf.get('admin_host', 'localhost'),
                     port=server_conf.get('admin_port', 5005),
                     ca_cert_file_name=root_cert,
                     server_cert_file_name=server_cert,
                     server_key_file_name=server_key,
                     accepted_client_cns=None,
                     mmar_validator=mmar_validator
        )
        return admin_server


def create_workspace(args):
    if os.path.exists('/tmp/fl_server'):
        shutil.rmtree('/tmp/fl_server')
    startup = os.path.join(args.workspace, 'startup')
    shutil.copytree(startup, '/tmp/fl_server')

    with open('/tmp/fl_server/config_train.json', 'wt') as f:
        f.write(r'{"epochs": 1250, "train": {   "loss": {   "name": "Dice" },   "model": {"name": "SegAhnet", "args": {"num_classes": 2, "if_use_psp": false, "pretrain_weight_name": "", "plane": "z", "final_activation": "softmax", "n_spatial_dim": 3}},   "pre_transforms": [{"name": "LoadNifti", "args": {"fields": ["image", "label"]}}],   "image_pipeline": {"name": "SegmentationImagePipelineWithCache", "args": {"data_list_file_path": "", "data_file_base_dir": "", "data_list_key": "training", "output_crop_size": [96, 96, 96]}} }}')

    with open('/tmp/fl_server/environment.json', 'wt') as f:
        f.write(r'{"PROCESSING_TASK": "segmentation","MMAR_CKPT_DIR": "models"}')

    # with open('/tmp/fl_server/log.config', 'wt') as f:
    #     f.write("[loggers]\nkeys=root,modelLogger\n[handlers]\nkeys=consoleHandler\n[formatters]\nkeys=fullFormatter\n[logger_root]\nlevel=INFO\nhandlers=consoleHandler\n[logger_modelLogger]\nlevel=DEBUG\nhandlers=consoleHandler\nqualname=modelLogger\npropagate=0\n[handler_consoleHandler]\nclass=StreamHandler\nlevel=DEBUG\nformatter=fullFormatter\nargs=(sys.stdout,)\n[formatter_fullFormatter]\nformat=%(asctime)s - %(name)s - %(levelname)s - %(message)s")


if __name__ == '__main__':

    main()

