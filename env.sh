# Source this file to isolate AWS context to the homecredit project.
#   source env.sh
#
# This file overrides the global ~/.aws/{config,credentials} for the current
# shell only. Opening a new terminal (or `exec $SHELL`) reverts to defaults.

export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]:-${(%):-%x}}" )" && pwd )"
export AWS_SHARED_CREDENTIALS_FILE="${PROJECT_ROOT}/.aws/credentials"
export AWS_CONFIG_FILE="${PROJECT_ROOT}/.aws/config"
export AWS_PROFILE="homecredit"
export AWS_REGION="us-west-2"
export AWS_DEFAULT_REGION="us-west-2"

# CDK picks these up for bootstrap / deploy
export CDK_DEFAULT_REGION="us-west-2"
export JSII_SILENCE_WARNING_UNTESTED_NODE_VERSION=1
CDK_DEFAULT_ACCOUNT="$(aws sts get-caller-identity --query Account --output text 2>/dev/null)"
export CDK_DEFAULT_ACCOUNT

echo "[homecredit] AWS context isolated"
echo "  profile : ${AWS_PROFILE}"
echo "  region  : ${AWS_REGION}"
echo "  account : ${CDK_DEFAULT_ACCOUNT:-<not verified>}"
echo "  config  : ${AWS_CONFIG_FILE}"
echo "  creds   : ${AWS_SHARED_CREDENTIALS_FILE}"
