cd ~/mount_folder/cuda_pcg/;
python setup.py build_ext --inplace;
# python unit_test/test_precon_vec.py
python unit_test/test_mhm_vec.py