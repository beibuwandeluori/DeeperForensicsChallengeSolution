# This is the sample Dockerfile for the evaluation.
# You can change it to include your own environment for submission.

#FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime
#FROM deeperforensics-challenge-320876555777:latest
FROM bokingchen/deeperforensics:latest

#RUN apt update \
#&& apt install -y -q libfontconfig1 libxrender1 libglib2.0-0 libsm6 libxext6 ucspi-tcp
#RUN pip install torch==1.6.0+cu92 torchvision==0.7.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple install boto3 opencv-python-headless pillow numpy scipy matplotlib ipython jupyter pandas sympy nose
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torchsummary albumentations facenet-pytorch pretrainedmodels==0.7.4 efficientnet_pytorch opencv-python==4.0.1.24

WORKDIR /workspace/DeeperForensics_Eval
COPY . .

# This command runs the evaluation tool.
# DO NOT MODIFY THE LINE BELOW OTHERWISE THE EVALUATION WILL NOT RUN
CMD [ "python", "run_evaluation.py" ]