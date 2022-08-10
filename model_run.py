from prediction import InsuranceModel
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

if __name__ == '__main__':
    dtc = InsuranceModel()
    exp_id, run_id = dtc.mlflow_run()
    print(f"MLflow Run completed with run_id {run_id} and experiment_id {exp_id}")
    print("<->" * 40)
