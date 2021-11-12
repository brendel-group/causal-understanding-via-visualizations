"""Loads the credentials needed to access AWS"""

import os


class AWSCredentials:
    """
    Access AWS credentials.
    """

    __loaded = False

    @staticmethod
    def aws_secret_access_key():
        AWSCredentials._load_data()
        return AWSCredentials.__aws_secret_access_key

    @staticmethod
    def aws_access_key_id():
        AWSCredentials._load_data()
        return AWSCredentials.__aws_access_key_id

    @staticmethod
    def _load_data():
        if AWSCredentials.__loaded:
            return

        aws_key_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "aws.key"
        )

        assert os.path.exists(aws_key_path), "AWS key file not found."

        with open(aws_key_path) as f:
            lines = f.readlines()
            AWSCredentials.__aws_access_key_id = lines[0].split("=")[1].strip()
            AWSCredentials.__aws_secret_access_key = lines[1].split("=")[1].strip()
            AWSCredentials.__loaded = True
