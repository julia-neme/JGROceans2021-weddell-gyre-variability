#!/bin/bash

###############################################################################
# Description	: This script uses CDO and parallel to get the ACCESS-OM2
#               variables needed for " " paper. To run it, you will need access
#               to GADI supercomputer at NCI, projects " "
# Args        :
#     wdir = path to directory in which save data
# Author      : Julia Neme
# Email       : j.neme@unsw.edu.au
###############################################################################

set -o errexit
set -o nounset
set -o pipefail

module load parallel
module load cdo

read -p "Please directory where to save the data: " wdir
dir_1d=/g/data/ik11/outputs/access-om2/1deg_jra55_iaf_omip2_cycle3
dir_025d=/g/data/ik11/outputs/access-om2-025/025deg_jra55_iaf_omip2_cycle3
dir_01d=/g/data/ik11/outputs/access-om2-01/01deg_jra55v140_iaf_cycle3

###############################################################################
# MODEL GRID
###############################################################################

cdo -sellonlatbox,-180,180,-80,-50 /g/data/ik11/grids/ocean_grid_10.nc "${wdir}"/ocean_grid_1deg.nc
cdo -sellonlatbox,-180,180,-80,-50 /g/data/ik11/grids/ocean_grid_025.nc "${wdir}"/ocean_grid_025deg.nc
cdo -sellonlatbox,-180,180,-80,-50 /g/data/ik11/grids/ocean_grid_01.nc "${wdir}"/ocean_grid_01deg.nc

###############################################################################
# OCEAN
###############################################################################

var=("tx_trans_int_z"
     "sea_level"
     "salt"
     "pot_temp"
     "net_sfc_heating"
     "pme_river"
     "frazil_3d_int_z"
     "tau_x"
     "tau_y")

for var_name in "${var}"; do

  # Get variable
  parallel cdo -select,name="${var_name}" "${dir_1d}"/output{}/ocean/ocean_month.nc ${wdir}/"${var_name}"_{}.nc ::: $(seq ${122} ${182})
  # Select Southern Ocean
  parallel cdo -sellonlatbox,-180,180,-80,-50 "${wdir}"/"${var}"_{}.nc "${wdir}"/"${var_name}"-so-{}.nc ::: $(seq ${122} ${182})
  rm "${wdir}"/"${var_name}"_*.nc
  # Merge times
  cdo mergetime "${wdir}"/"${var_name}"-so*.nc "${wdir}"/"${var_name}"-monthly-1958_2018-1deg.nc
  rm "${wdir}"/"${var_name}"-so-*.nc

  # Get variable
  parallel cdo -select,name="${var_name}" "${dir_025d}"/output{}/ocean/ocean_month.nc ${wdir}/"${var_name}"_{}.nc ::: $(seq ${122} ${182})
  # Select Southern Ocean
  parallel cdo -sellonlatbox,-180,180,-80,-50 "${wdir}"/"${var_name}"_{}.nc "${wdir}"/"${var_name}"-so-{}.nc ::: $(seq ${122} ${182})
  rm "${wdir}"/"${var_name}"_*.nc
  # Merge times
  cdo mergetime "${wdir}"/"${var_name}"-so*.nc "${wdir}"/"${var_name}"-monthly-1958_2018-025deg.nc
  rm "${wdir}"/"${var_name}"-so-*.nc

  # Get variable and select Southern Ocean
  parallel cdo -sellonlatbox,-180,180,-80,-50 "${wdir}"/*"${var_name}"-1-monthly-mean*.nc "${wdir}"/"${var_name}"-so-{}.nc ::: $(seq ${488} ${731})
  rm "${wdir}"/"${var_name}"_*.nc
  # Merge times
  cdo mergetime "${wdir}"/"${var_name}"-so*.nc "${wdir}"/"${var_name}"-monthly-1958_2018-01deg.nc
  rm "${wdir}"/"${var_name}"-so-*.nc

done

parallel cdo -select,name=eta_global "${dir_1d}"/output{}/ocean/ocean_scalar.nc "${wdir}"/temp-eta_global_{}.nc ::: $(seq ${122} ${182})
cdo mergetime "${wdir}"/eta_global*.nc "${wdir}"/eta_global-monthly-1958_2018-1deg.nc
rm "${wdir}"/tmp-eta_global*.nc

parallel cdo -select,name=eta_global "${dir_025d}"/output{}/ocean/ocean_scalar.nc "${wdir}"/temp-eta_global_{}.nc ::: $(seq ${122} ${182})
cdo mergetime "${wdir}"/eta_global*.nc "${wdir}"/eta_global-monthly-1958_2018-025deg.nc
rm "${wdir}"/tmp-eta_global*.nc

parallel cdo -select,name=eta_global "${dir_01d}"/output{}/ocean/ocean_scalar.nc "${wdir}"/temp-eta_global_{}.nc ::: $(seq ${122} ${182})
cdo mergetime "${wdir}"/eta_global*.nc "${wdir}"/eta_global-monthly-1958_2018-01deg.nc
rm "${wdir}"/tmp-eta_global*.nc

###############################################################################
# ICE
###############################################################################

parallel cdo -select,name=aice_m "${dir_1d}"/output{}/ice/OUTPUT/iceh.????-??.nc "${wdir}"/tmp-aice_m_{}.nc ::: $(seq ${122} ${182})
cdo mergetime "${dir_1d}"/tmp-aice_m_*.nc "${wdir}"/aice_m-monthly-1958_2018-1deg-global.nc
rm tmp-aice_m*.nc

parallel cdo -select,name=aice_m "${dir_025d}"/output{}/ice/OUTPUT/iceh.????-??.nc "${wdir}"/tmp-aice_m_{}.nc ::: $(seq ${122} ${182})
cdo mergetime "${dir_025d}"/tmp-aice_m_*.nc "${wdir}"/aice_m-monthly-1958_2018-025deg-global.nc
rm tmp-aice_m*.nc

parallel cdo -select,name=aice_m "${dir_01d}"/output{}/ice/OUTPUT/iceh.????-??.nc "${wdir}"/tmp-aice_m_{}.nc ::: $(seq ${488} ${731})
cdo mergetime "${dir_01d}"/tmp-aice_m_*.nc "${wdir}"/aice_m-monthly-1958_2018-01deg-global.nc
rm tmp-aice_m*.nc

###############################################################################
# JRA55 Sea Level Pressure
###############################################################################

mkdir "${wdir}"/tmp
parallel cdo -monmean {} "${wdir}"/{/.}.nc ::: /g/data/qv56/replicas/input4MIPs/CMIP6/OMIP/MRI/MRI-JRA55-do-1-4-0/atmos/3hrPt/psl/gr/v20190429/*
cdo mergetime "${wdir}"/tmp/* "${wdir}"/psl-monthly-1958_2019.nc
rm "${wdir}"/tmp/*
rmdir tmp
