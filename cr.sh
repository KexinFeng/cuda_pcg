cd ~/mount_folder/cuda_pcg/;
MAX_JOBS=${1:-2} python setup.py build_ext --inplace || { echo "Build failed. Exiting."; exit 1; }
sleep 2  # Wait for 2 seconds to ensure build completion
python unit_test/test_precon_vec.py
python unit_test/test_mhm_vec.py
python unit_test/test_b_vec_mul_per_tau.py
# compute-sanitizer --tool=memcheck python unit_test/test_mhm_vec.py
cp ~/mount_folder/cuda_pcg/qed_fermion_module/*.so ~/hmc/qed_fermion/qed_fermion/
