#!/usr/bin/env bash
#
# Description: Quick values that approximate a 25um^3 area, these values are for
# testing only, non-biological. The goal is a simulation that runs within the 
# hardware restrictions of GitHub runners, 7 GB and a job timeout of 6 hrs. 
#
# The paper utilizes cell numbers that approximate a 100um^3 area, which is 
# grounded in anatomical and functional studies of the cerebellum.
export CHAT_NUM_TRIAL=10 CHAT_NUM_MF=78 CHAT_NUM_GRC=1024 CHAT_NUM_GOC=7 CHAT_SEED_TRIAL=451