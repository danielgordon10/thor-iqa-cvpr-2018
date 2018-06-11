gdrive_download () {
    CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
    rm -rf /tmp/cookies.txt
}
tar_download_and_extract() {
    cd $1
    rm -rf $2*
    gdrive_download $3 $2.tar.gz
    tar -zxvf $2.tar.gz
    rm -rf $2.tar.gz
    cd ..
}

tar_download_and_extract depth_estimation_network depth_network_weights 1KL5RmHO32pXt150Vbk6ixHceU2nBp6Vk
tar_download_and_extract darknet_object_detection yolo_weights 1jBMYeZY7n_dkeRMqq1XDCXljqR6c70DP
cd logs
tar_download_and_extract checkpoints checkpoints 1BVB_JfD-ohxq23cIXKj7nsY47Kn8e9eP
