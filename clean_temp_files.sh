#!/bin/bash
rm -rf *.c
rm -rf *.o
rm -rf *.so
rm -rf acados_sicnav_mpc*.json
rm -rf c_generated_code*
# rm -rf *.pkl
# rm *.mp4

rm -rf *.dylib
rm -rf debug/*.c
rm -rf debug/*.o
rm -rf debug/*.so
rm -rf debug/acados_ocp_nlp.json
rm -rf debug/c_generated_code
rm -rf debug/*.pkl
rm debug/*.mp4

# rm -rf *.log
rm -rf temp_files/*

clear
