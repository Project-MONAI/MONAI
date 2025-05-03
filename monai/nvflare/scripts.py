import os
import shutil
from nvflare.apis.fl_context import FLContext
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.abstract.fl_app_script import FLAppScript


class ResultDownloader(FLAppScript):
    def __init__(self, job_id: str, destination_path: str,bundle_root: str, fold_id: int):
        super().__init__()
        self.job_id = job_id
        self.destination_path = destination_path
        self.fold_id = fold_id
        self.bundle_root = bundle_root

    def execute(self, fl_ctx: FLContext):
        engine: ServerEngineSpec = fl_ctx.get_prop(AppConstants.ENGINE)
        job_meta = engine.get_job_store().get_job_meta(self.job_id)
        job_dir = job_meta.get("folder")

        if not job_dir or not os.path.exists(job_dir):
            self.log_error(fl_ctx, f"Job directory for {self.job_id} not found.")
            return

        try:
            # 1. Copy job directory to destination_path
            shutil.copytree(job_dir, self.destination_path, dirs_exist_ok=True)
            self.log_info(fl_ctx, f"Copied results for job {self.job_id} to {self.destination_path}")

            # 2. Locate the global model file
            source_model = os.path.join(self.destination_path, "job", "workspace", "app_server", "FL_global_model.pt")
            if not os.path.exists(source_model):
                self.log_error(fl_ctx, f"Model file not found at expected location: {source_model}")
                return

            # 3. Determine target path inside BUNDLE_ROOT
            target_model_dir = os.path.join(self.bundle_root, "models", f"fold_{self.fold_id}")
            os.makedirs(target_model_dir, exist_ok=True)

            target_model_path = os.path.join(target_model_dir, "FL_global_model.pt")

            # 4. Copy model file
            shutil.copy2(source_model, target_model_path)
            self.log_info(fl_ctx, f"Model copied to {target_model_path}")

        except Exception as e:
            self.log_error(fl_ctx, f"Failed to process job results: {str(e)}")
