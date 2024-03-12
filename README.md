# Design of an Explainable Input Control for Cross-Company Federated Learning

This repository contains code to demonstrate the federated learning input control artifact proposed in the ECIS 2024 paper titled "Design of an Explainable Input Control for Cross-Company Federated Learning".

The implementation provided simulates a FL setting and showcases how the porposed input control can improve the performance of a real-world FL system.
In order to increase the ease of setting up the demonstration, we opted for providing a code base to locally simulate FL on a single device. Hence, eleminating the need to take care of sending and synchronizing model updates between diestributed clients.

## Setup
The repository contains two notebooks used to run all the code.

The notebook **create_churn_data** contains the code necessary to create the churn dataset from the original dataset offered on kaggle.
The notebook **flp_input_control** serves to perform and demonstrate all components of the proposed artifact.
