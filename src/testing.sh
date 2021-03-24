#!/bin/bash

###############################################################################
# Description	: This script uses CDO and parallel to get the ACCESS-OM2 variables needed for " " paper. To run it, you will need access to GADI supercomputer at NCI, projects " "
# Args        :
#     wdir = path to working directory to which save the data
# Author      : Julia Neme
# Email       : j.neme@unsw.edu.au
###############################################################################

set -o errexit
set -o nounset
set -o pipefail

read -p "Please working directory: " wdir
dir_1d=/g/data/ik11/outputs/access-om2/1deg_jra55_iaf_omip2_cycle3
dir_025d=/g/data/ik11/outputs/access-om2-025/025deg_jra55_iaf_omip2_cycle3
# CAREFUL: FOR NOW THE DATA ISNT THERE
dir_01d=/g/data/ik11/outputs/access-om2-01/01deg_jra55v140_iaf_cycle3

var=("tx_trans_int_z"
     "sea_level"
     "salt"
     "pot_temp"
     "net_sfc_heating"
     "pme_river"
     "frazil_3d_int_z")

for var_name in "${var}"; do
  echo "${wdir}"/"${var_name}"
done
