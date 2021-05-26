-- Train requires refactoring
-- First commit, convert readme to md
-- Download kitti_baselines from link
How to test on baselines:

-- Install all the packages from requirements.txt file
-- kitti-baselines folder consists of all reconstructions for a kitti run from all baselines
-- kitti.pth is the LQI model trained on KITTI
-- To run LQI on baselines run the command python test.py --model_path kitti.pth --path kitti_baselines
