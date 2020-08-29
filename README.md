# DeeperForensics Challenge Submission Example
This repo provides an example Docker image for submission of DeeperForensics Challenge 2020. The submission is in the form of online evaluation.

## Important hints
Please refer to the following important hints to ensure a pleasant online evaluation experience:

1. We highly recommend you to **test your Docker images locally** before submission. Or your code may not be run properly on the online evaluation platform. Please refer to [Test the Docker image locally](#test-the-docker-image-locally) for instructions.

2. In the first week, you will be allocated two extra evaluation chances. **For an initial test**, we highly recommend you to **extract one frame only for each video** to test your model, in order to estimate the online evaluation time and avoid timeout. The total runtime (`ExecutionTime`, in seconds) can be seen when you successfully submit to the [CodaLab leaderboard](https://competitions.codalab.org/competitions/25228#results) or use `download output from scoring step` at [Submit / View Results](https://competitions.codalab.org/competitions/25228#participate-submit_results).

3. Please carefully read the part regarding the **hidden test set** at the challenge website ([Overview](https://competitions.codalab.org/competitions/25228#learn_the_details-overview) and [Terms and Conditions](https://competitions.codalab.org/competitions/25228#learn_the_details-terms_and_conditions)), which may be helpful for successful submission and performance boost.

4. Preprocessing might occupy most of the runtime.

## Before you start: request resource provision

1. Create an account on the [challenge website (CodaLab)](https://competitions.codalab.org/competitions/25228), as well as an [AWS account](https://aws.amazon.com/account/) (in any region except Beijing and Ningxia);

2. [Register](https://competitions.codalab.org/competitions/25228#participate) DeeperForensics Challenge 2020 using the created CodaLab account;

3. Send your **CodaLab user name (i.e., team name)**, **the number of team members**, **affiliation**, **CodaLab email address**, and **AWS account id (12 digits)** to the organizers' email address: [deeperforensics@gmail.com](mailto:deeperforensics@gmail.com). We will check and allocate evaluation resources for you. You will receive a email notifying you that the resources have been allocated. **If you don't receive the email, it may be recognized as spam.** You will receive 3 emails in one evaluation.

## Install and configure AWS CLI
Then you should install AWS CLI ([version 2](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) recommended).

After installation, you should configure the settings that the AWS Command Line Interface (AWS CLI) uses to interact with AWS:

1. Generate the AWS Access Key ID and AWS Secret Access Key in IAM: AWS Management Console -> Find Services -> Enter 'IAM' -> Choose IAM (Manage access to AWS resources) -> Delete your root access keys -> Manage security credentials -> Access keys (access key ID and secret access key) -> Create New Access Key.

2. Run this command:
```bash
aws configure
```
Then it will require you to input AWS Access Key ID, AWS Secret Access Key, Default region name (**please input us-west-2**) and Default output format (left it empty).
If you still have questions, please refer to [here](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html).

## Install Docker Engine
In order to build your Docker image, you should install Docker Engine first. Please refer to [Install Docker Engine](https://docs.docker.com/engine/install/).

## Install nvidia-docker
Because GPU is necessary for both the local test and online evaluation, we also need to install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

## Obtain this example

Run the following command to clone this submission example repo:

```bash
git clone https://github.com/Drdestiny/DeeperForensicsChallengeSubmissionExample.git
```

## Include your own algorithm & Build Docker image

- Your algorithm codes should be put in `model`.
- Your detector should inherit `DeeperForensicsDetector` class in `eval_kit/detector.py`. For example:

```python
import sys

sys.path.append('..')
from eval_kit.detector import DeeperForensicsDetector

class ToyPredictor(DeeperForensicsDetector): # You can give your detector any name.
    ...
```
You need to implement the abstract function `predict(self, video_frames)` in your detector class:

```python
    @abstractmethod
    def predict(self, video_frames):
        """
        Process a list of video frames, the evaluation toolkit will measure the runtime of every call to this method.
        
        params:
            - video_frames (list): may be a list of numpy arrays with dtype=np.uint8 representing frames of **one** video
        return:
            - probability (float)
        """
        pass
```

- Modify [Line 25](https://github.com/Drdestiny/DeeperForensicsChallengeSubmissionExample/blob/master/run_evaluation.py#L25) in `run_evalution.py` to import your own detector for evaluation.

```python
########################################################################################################
# please change these lines to include your own face forgery detector extending the eval_kit.detector.DeeperForensicsDetector base class.
from toy_predict import ToyPredictor as DeeperForensicsDetector
########################################################################################################
```

- Besides, you can implement your own frame extracting method. If so, you should modify `eval_kit/extract_frames.py`. **Make sure that the return type remains the same (**`<class 'list'>`**).**

- Edit `Dockerfile` to build your Docker image. If you don't know how to use Dockerfile, please refer to:
  -  [Best practices for writing Dockerfiles (English)](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#dockerfile-instructions)
  -  [使用 Dockerfile 定制镜像 (Chinese)](https://yeasy.gitbook.io/docker_practice/image/build)
  -  [Dockerfile 指令详解 (Chinese)](https://yeasy.gitbook.io/docker_practice/image/dockerfile)

## Test the Docker image locally

The online evaluation for submissions may take several hours. It is slow to debug on the cloud (and the evaluation chances are limited). Hence, we provide the tools for the participants to locally test the correctness of the submission.

Before running, modify [Line 28](https://github.com/Drdestiny/DeeperForensicsChallengeSubmissionExample/blob/master/local_test.py#L28) in `local_test.py`.

To verify your algorithm can be run properly, run the following command:

```bash
docker run -it deeperforensics-challenge-<your_aws_id> python local_test.py
```

**Please refer to step 2 and 3 in** [Submit the Docker image](#submit-the-docker-image) **to learn how to tag your Docker image.**

It will run the algorithms in the evaluation workflow on some example videos and print out the results.

The output will look like:

```
    ================================================================================
    all videos finished, showing verification info below:
    ================================================================================
    
INFO:root:Video ID: 000000.mp4, Runtime: 7.3909759521484375e-06
INFO:root:	gt: 0
INFO:root:	output probability: 0.5
INFO:root:	number of frame: 1
INFO:root:	output time: 7.3909759521484375e-06
INFO:root: 
INFO:root:Video ID: 000001.mp4, Runtime: 2.3603439331054688e-05
INFO:root:	gt: 1
INFO:root:	output probability: 0.5
INFO:root:	number of frame: 1
INFO:root:	output time: 2.3603439331054688e-05
INFO:root: 
INFO:root:Video ID: 000002.mp4, Runtime: 8.106231689453125e-06
INFO:root:	gt: 1
INFO:root:	output probability: 0.5
INFO:root:	number of frame: 1
INFO:root:	output time: 8.106231689453125e-06
INFO:root: 
INFO:root:Video ID: 000003.mp4, Runtime: 2.2649765014648438e-05
INFO:root:	gt: 0
INFO:root:	output probability: 0.5
INFO:root:	number of frame: 1
INFO:root:	output time: 2.2649765014648438e-05
INFO:root: 
INFO:root:Done. Average FPS: 64776.896
```

## Submit the Docker image

1. Retrieve the login command to use to authenticate your Docker client to your registry:

```bash
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 212923332099.dkr.ecr.us-west-2.amazonaws.com
```

2. Build your Docker image using the following command. For information on building a Docker file from scratch see the instructions here. You can skip this step if your image has already been built:

```bash
cd ../DeeperForensicsChallengeSubmissionExample
docker build -t deeperforensics-challenge-<your_aws_id> .  # . means the current path. Please don't lose it.
```

For example:

```bash
docker build -t deeperforensics-challenge-123412341234 . 
```

3. After the build is completed, tag your image so you can push the image to the repository:

```bash
docker tag deeperforensics-challenge-<your_aws_id>:latest 212923332099.dkr.ecr.us-west-2.amazonaws.com/deeperforensics-challenge-<your_aws_id>:latest
```

4. Run the following command to push this image to your AWS ECR repository:

```bash
docker push 212923332099.dkr.ecr.us-west-2.amazonaws.com/deeperforensics-challenge-<your_aws_id>:latest
```

After you push to the repo, the evaluation will automatically start and you will receive an email notifying you that the evaluation starts. In **2.5 hours** you should receive another email with the evaluation result. If there is something wrong like timeout or error, you will also receive a reminder email. Please look in the spam if you don't receive any email!

Finally, you can submit the evaluation result to the [challenge website](https://competitions.codalab.org/competitions/25228).
