"""Idempotently provision the EC2 resources used by the income API."""

import json
import os
from pathlib import Path
import time

import boto3
from botocore.exceptions import ClientError


REGION = "ap-southeast-1"
BUCKET = "income-cicd-394877251429"
INSTANCE_NAME = "income-api"
ROLE_NAME = "income-api-ec2-role"
PROFILE_NAME = "income-api-ec2-profile"
SECURITY_GROUP_NAME = "income-api-sg"
KEY_NAME = "income-deploy"
REPOSITORY = (
    "https://github.com/PhamDuong2705/"
    "K4-Track2-Day21-2A202601404-PhamTungDuong.git"
)
ROOT = Path(__file__).resolve().parents[1]


def _ignore_duplicate(action, code: str):
    try:
        return action()
    except ClientError as exc:
        if exc.response["Error"]["Code"] != code:
            raise
        return None


def ensure_instance_role(iam) -> None:
    trust = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    }
    try:
        iam.get_role(RoleName=ROLE_NAME)
    except iam.exceptions.NoSuchEntityException:
        iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust),
            Description="Read the approved Adult Income model from S3",
        )

    model_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Action": "s3:GetObject",
            "Resource": f"arn:aws:s3:::{BUCKET}/artifacts/current/*",
        }],
    }
    iam.put_role_policy(
        RoleName=ROLE_NAME,
        PolicyName="ReadIncomeProductionModel",
        PolicyDocument=json.dumps(model_policy),
    )

    try:
        profile = iam.get_instance_profile(
            InstanceProfileName=PROFILE_NAME
        )["InstanceProfile"]
    except iam.exceptions.NoSuchEntityException:
        profile = iam.create_instance_profile(
            InstanceProfileName=PROFILE_NAME
        )["InstanceProfile"]

    if not any(role["RoleName"] == ROLE_NAME for role in profile.get("Roles", [])):
        iam.add_role_to_instance_profile(
            InstanceProfileName=PROFILE_NAME,
            RoleName=ROLE_NAME,
        )
        time.sleep(10)


def ensure_key_pair(ec2) -> None:
    public_key_path = Path(
        os.getenv(
            "INCOME_PUBLIC_KEY_PATH",
            str(ROOT / ".secrets" / "income_deploy.pub"),
        )
    )
    public_key = public_key_path.read_bytes()
    _ignore_duplicate(
        lambda: ec2.import_key_pair(
            KeyName=KEY_NAME,
            PublicKeyMaterial=public_key,
            TagSpecifications=[{
                "ResourceType": "key-pair",
                "Tags": [{"Key": "Name", "Value": KEY_NAME}],
            }],
        ),
        "InvalidKeyPair.Duplicate",
    )


def ensure_security_group(ec2) -> str:
    vpc = ec2.describe_vpcs(
        Filters=[{"Name": "is-default", "Values": ["true"]}]
    )["Vpcs"][0]
    matches = ec2.describe_security_groups(Filters=[
        {"Name": "group-name", "Values": [SECURITY_GROUP_NAME]},
        {"Name": "vpc-id", "Values": [vpc["VpcId"]]},
    ])["SecurityGroups"]
    if matches:
        group_id = matches[0]["GroupId"]
    else:
        group_id = ec2.create_security_group(
            GroupName=SECURITY_GROUP_NAME,
            Description="SSH deployment and public Income API",
            VpcId=vpc["VpcId"],
            TagSpecifications=[{
                "ResourceType": "security-group",
                "Tags": [{"Key": "Name", "Value": SECURITY_GROUP_NAME}],
            }],
        )["GroupId"]

    for port, description in ((22, "GitHub Actions SSH"), (8080, "Income API")):
        _ignore_duplicate(
            lambda port=port, description=description: ec2.authorize_security_group_ingress(
                GroupId=group_id,
                IpPermissions=[{
                    "IpProtocol": "tcp",
                    "FromPort": port,
                    "ToPort": port,
                    "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": description}],
                }],
            ),
            "InvalidPermission.Duplicate",
        )
    return group_id


def find_ubuntu_ami(ec2) -> str:
    images = ec2.describe_images(
        Owners=["099720109477"],  # Canonical
        Filters=[
            {
                "Name": "name",
                "Values": ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"],
            },
            {"Name": "architecture", "Values": ["x86_64"]},
            {"Name": "state", "Values": ["available"]},
        ],
    )["Images"]
    if not images:
        raise RuntimeError("No Ubuntu 24.04 AMI found")
    return max(images, key=lambda image: image["CreationDate"])["ImageId"]


def user_data() -> str:
    return f"""#!/usr/bin/env bash
set -euxo pipefail
apt-get update
apt-get install -y git
git clone --branch main {REPOSITORY} /tmp/income-api-bootstrap
bash /tmp/income-api-bootstrap/deploy/bootstrap-ec2.sh {BUCKET} {REPOSITORY}
"""


def ensure_instance(ec2, ami_id: str, security_group_id: str) -> dict:
    existing = ec2.describe_instances(Filters=[
        {"Name": "tag:Name", "Values": [INSTANCE_NAME]},
        {"Name": "instance-state-name", "Values": ["pending", "running", "stopped"]},
    ])["Reservations"]
    if existing:
        instance = existing[0]["Instances"][0]
        if instance["State"]["Name"] == "stopped":
            ec2.start_instances(InstanceIds=[instance["InstanceId"]])
        instance_id = instance["InstanceId"]
    else:
        instance_id = ec2.run_instances(
            ImageId=ami_id,
            InstanceType="t3.micro",
            MinCount=1,
            MaxCount=1,
            KeyName=KEY_NAME,
            IamInstanceProfile={"Name": PROFILE_NAME},
            SecurityGroupIds=[security_group_id],
            UserData=user_data(),
            MetadataOptions={"HttpTokens": "required", "HttpEndpoint": "enabled"},
            TagSpecifications=[{
                "ResourceType": "instance",
                "Tags": [{"Key": "Name", "Value": INSTANCE_NAME}],
            }],
        )["Instances"][0]["InstanceId"]

    ec2.get_waiter("instance_running").wait(InstanceIds=[instance_id])
    return ec2.describe_instances(InstanceIds=[instance_id])["Reservations"][0]["Instances"][0]


def main() -> None:
    session = boto3.Session(region_name=REGION)
    iam = session.client("iam")
    ec2 = session.client("ec2")

    ensure_instance_role(iam)
    ensure_key_pair(ec2)
    security_group_id = ensure_security_group(ec2)
    instance = ensure_instance(ec2, find_ubuntu_ami(ec2), security_group_id)
    print(json.dumps({
        "instance_id": instance["InstanceId"],
        "public_ip": instance.get("PublicIpAddress"),
        "state": instance["State"]["Name"],
        "security_group_id": security_group_id,
        "region": REGION,
    }, indent=2))


if __name__ == "__main__":
    main()
