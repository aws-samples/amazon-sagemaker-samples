import sys
import argparse
import pathlib
import time

import boto3
import pandas as pd
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup

# Parse argument variables passed via the CreateDataset processing step
parser = argparse.ArgumentParser()
parser.add_argument("--region", type=str, default='us-east-1')
parser.add_argument("--bucket", type=str, required=True)
parser.add_argument("--prefix", type=str, required=True)
args = parser.parse_args()

region = args.region
prefix = args.prefix
bucket = args.bucket

account_id = boto3.client("sts").get_caller_identity()["Account"]

# Get the default role that was created for this domaim
sagemaker_role = sagemaker.get_execution_role()

# create a default session in that region
boto3.setup_default_session(region_name=region)

# Reference that session
boto_session = boto3.Session(region_name=region)

# create a sagemaker client
sagemaker_boto_client = boto_session.client("sagemaker")

# then link the two
sagemaker_session = sagemaker.session.Session(
    boto_session=boto_session, sagemaker_client=sagemaker_boto_client
)

# create an s3 client
s3_client = boto3.client("s3", region_name=region)

featurestore_runtime = boto_session.client(
    service_name="sagemaker-featurestore-runtime", region_name=region
)

feature_store_session = sagemaker.Session(
    boto_session=boto_session,
    sagemaker_client=sagemaker_boto_client,
    sagemaker_featurestore_runtime_client=featurestore_runtime,
)

claims_fg_name = f"{prefix}-claims"
customers_fg_name = f"{prefix}-customers"

claims_feature_group = FeatureGroup(
    name=claims_fg_name, sagemaker_session=feature_store_session
)

customers_feature_group = FeatureGroup(
    name=customers_fg_name, sagemaker_session=feature_store_session
)


claims_dtypes = {
    "policy_id": int,
    "incident_severity": int,
    "num_vehicles_involved": int,
    "num_injuries": int,
    "num_witnesses": int,
    "police_report_available": int,
    "injury_claim": float,
    "vehicle_claim": float,
    "total_claim_amount": float,
    "incident_month": int,
    "incident_day": int,
    "incident_dow": int,
    "incident_hour": int,
    "fraud": int,
    "driver_relationship_self": int,
    "driver_relationship_na": int,
    "driver_relationship_spouse": int,
    "driver_relationship_child": int,
    "driver_relationship_other": int,
    "incident_type_collision": int,
    "incident_type_breakin": int,
    "incident_type_theft": int,
    "collision_type_front": int,
    "collision_type_rear": int,
    "collision_type_side": int,
    "collision_type_na": int,
    "authorities_contacted_police": int,
    "authorities_contacted_none": int,
    "authorities_contacted_fire": int,
    "authorities_contacted_ambulance": int,
    "event_time": float,
}

customers_dtypes = {
    "policy_id": int,
    "customer_age": int,
    "customer_education": int,
    "months_as_customer": int,
    "policy_deductable": int,
    "policy_annual_premium": int,
    "policy_liability": int,
    "auto_year": int,
    "num_claims_past_year": int,
    "num_insurers_past_5_years": int,
    "customer_gender_male": int,
    "customer_gender_female": int,
    "policy_state_ca": int,
    "policy_state_wa": int,
    "policy_state_az": int,
    "policy_state_or": int,
    "policy_state_nv": int,
    "policy_state_id": int,
    "event_time": float,
}

timestamp = pd.to_datetime("now").timestamp()

claims_preprocessed = pd.read_csv(
    filepath_or_buffer="./data/claims_preprocessed.csv", dtype=claims_dtypes
)

# a timestamp column is required by the feature store, so one is added with a current timestamp
claims_preprocessed["event_time"] = timestamp

customers_preprocessed = pd.read_csv(
    filepath_or_buffer="./data/customers_preprocessed.csv", dtype=customers_dtypes
)

customers_preprocessed["event_time"] = timestamp

claims_feature_group.load_feature_definitions(data_frame=claims_preprocessed)
customers_feature_group.load_feature_definitions(data_frame=customers_preprocessed);

record_identifier_feature_name = "policy_id"
event_time_feature_name = "event_time"


try:
    print(f"\nUsing s3://{bucket}/{prefix}")
    claims_feature_group.create(
        s3_uri=f"s3://{bucket}/{prefix}",
        record_identifier_name=record_identifier_feature_name,
        event_time_feature_name=event_time_feature_name,
        role_arn=sagemaker_role,
        enable_online_store=True,
    )
    print(f'Create "claims" feature group: SUCCESS')
except Exception as e:
    code = e.response.get("Error").get("Code")
    if code == "ResourceInUse":
        print(f"Using existing feature group: {claims_fg_name}")
    else:
        raise (e)

try:
    customers_feature_group.create(
        s3_uri=f"s3://{bucket}/{prefix}",
        record_identifier_name=record_identifier_feature_name,
        event_time_feature_name=event_time_feature_name,
        role_arn=sagemaker_role,
        enable_online_store=True,
    )
    print(f'Create "customers" feature group: SUCCESS')
except Exception as e:
    code = e.response.get("Error").get("Code")
    if code == "ResourceInUse":
        print(f"Using existing feature group: {customers_fg_name}")
    else:
        raise (e)

# Wait until feature group creation has fully completed
def wait_for_feature_group_creation_complete(feature_group):
    status = feature_group.describe().get("FeatureGroupStatus")
    while status == "Creating":
        print("Waiting for Feature Group Creation")
        time.sleep(5)
        status = feature_group.describe().get("FeatureGroupStatus")
    if status != "Created":
        raise RuntimeError(f"Failed to create feature group {feature_group.name}")
    print(f"FeatureGroup {feature_group.name} successfully created.")


wait_for_feature_group_creation_complete(feature_group=claims_feature_group)
wait_for_feature_group_creation_complete(feature_group=customers_feature_group)

# Ingest records into the Feature Groups
if "claims_table" in locals():
    print(
        "You may have already ingested the data into your Feature Groups. If you'd like to do this again, you can run the ingest methods outside of the 'if/else' statement."
    )

else:
    claims_feature_group.ingest(data_frame=claims_preprocessed, max_workers=3, wait=True)

    customers_feature_group.ingest(data_frame=customers_preprocessed, max_workers=3, wait=True);

# Wait for offline store data to become available
if "claims_table" not in locals():
    claims_table = claims_feature_group.describe()["OfflineStoreConfig"]["DataCatalogConfig"][
        "TableName"
    ]
if "customers_table" not in locals():
    customers_table = customers_feature_group.describe()["OfflineStoreConfig"]["DataCatalogConfig"][
        "TableName"
    ]

claims_feature_group_s3_prefix = (
    f"{prefix}/{account_id}/sagemaker/{region}/offline-store/{claims_table.replace('_','-')}/data"
)
customers_feature_group_s3_prefix = (
    f"{prefix}/{account_id}/sagemaker/{region}/offline-store/{customers_table.replace('_','-')}/data"
)

print(claims_feature_group_s3_prefix)

offline_store_contents = None
while offline_store_contents is None:
    objects_in_bucket = s3_client.list_objects(
        Bucket=bucket, Prefix=customers_feature_group_s3_prefix
    )
    if "Contents" in objects_in_bucket and len(objects_in_bucket["Contents"]) > 1:
        offline_store_contents = objects_in_bucket["Contents"]
    else:
        print("Waiting for data in offline store...")
        time.sleep(60)

print("\nData available.")

