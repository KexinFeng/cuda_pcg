cd ~/mount_folder/cuda_pcg/;
python setup.py build_ext --inplace || { echo "Build failed. Exiting."; exit 1; }
python unit_test/test_mhm_vec.py
# compute-sanitizer --tool=memcheck python unit_test/test_mhm_vec.py