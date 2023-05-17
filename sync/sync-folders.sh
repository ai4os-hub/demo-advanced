export RCLONE_CONFIG=sync/rclone.conf
rclone copy models: ./models --progress
rclone copy data: ./data --progress