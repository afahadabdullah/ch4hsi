#!/bin/tcsh
# Interactive use from Prism's default tcsh on a grace node, e.g.
#   srun -p grace --gpus=1 -c 16 --mem=64G -t 2:00:00 --pty tcsh
#   cd ~/ch4hsi ; source slurm/activate.csh ; python -m ch4hsi show-config
# (A C shell cannot source the Bash helpers; batch jobs use common.sh.)

if ( ! $?CH4HSI_DATA ) then
  if ( -d "/panfs/ccds02/nobackup/people/$USER" ) then
    setenv CH4HSI_DATA "/panfs/ccds02/nobackup/people/$USER/ch4hsi"
  else
    setenv CH4HSI_DATA "/explore/nobackup/people/$USER/ch4hsi"
  endif
endif
set ch4_arch = `uname -m`
if ( "$ch4_arch" == "aarch64" ) then
  set ch4_env = "$CH4HSI_DATA/.envs/ch4hsi-aarch64"
else
  set ch4_env = "$CH4HSI_DATA/.envs/ch4hsi-x86_64"
endif
if ( ! -x "$ch4_env/bin/python" ) then
  echo "FATAL: no ch4hsi environment for $ch4_arch at $ch4_env"
  goto ch4_done
endif
unsetenv PYTHONHOME
unsetenv PYTHONPATH
setenv PYTHONNOUSERSITE 1
setenv PATH "$ch4_env/bin:$PATH"
if ( -r "$ch4_env/share/proj/proj.db" ) then
  setenv PROJ_DATA "$ch4_env/share/proj"
  setenv PROJ_LIB "$ch4_env/share/proj"
endif
rehash
echo "ch4hsi python: `which python`   data: $CH4HSI_DATA"
echo "run stages with:  python -m ch4hsi <stage> --config configs/gh200.yaml --set paths.data_root=$CH4HSI_DATA"
ch4_done:
unset ch4_arch
unset ch4_env
