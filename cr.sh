cd ~/mount_folder/cuda_pcg/;
python setup.py build_ext --inplace || { echo "Build failed. Exiting."; exit 1; }
# python unit_test/test_precon_vec.py
python unit_test/test_mhm_vec.py