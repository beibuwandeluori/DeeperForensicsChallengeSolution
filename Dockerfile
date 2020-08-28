# This is the sample Dockerfile for the evaluation.
# You can change it to include your own environment for submission.

FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

RUN apt update \
&& apt install -y -q libfontconfig1 libxrender1 libglib2.0-0 libsm6 libxext6 ucspi-tcp

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple install boto3 opencv-python-headless pillow numpy scipy matplotlib ipython jupyter pandas sympy nose

WORKDIR /workspace/DeeperForensics_Eval
COPY . .

# This command runs the evaluation tool.
# DO NOT MODIFY THE LINE BELOW OTHERWISE THE EVALUATION WILL NOT RUN
CMD [ "python", "run_evaluation.py" ]