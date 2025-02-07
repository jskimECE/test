FROM python:3.10.1-buster

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.

## Include the following line if you have a requirements.txt file.
RUN pip3 install -r requirements.txt

# Ensure necessary directories exist
RUN mkdir -p model test_outputs


# # Set the command to train and run the model
# CMD ["sh", "-c", "python3 train_model.py SyntheticData_Training.csv model && python3 run_model.py model test_data.csv test_outputs"]
