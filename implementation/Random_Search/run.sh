pwd

touch matrices.txt
echo matrix,accuracy,f1,loss,precision,recall >results.csv


for i in $(seq 0 1500); do
  printf '\n-------------\n--- Run %s ---\n-------------\n' "$i"
  pyklopp init my_model --save "train/train_$i/model.pth"
  pyklopp train train/train_"$i"/model.pth my_dataset.get_dataset_train --save="train/train_$i/trained.pth" --config='{"dataset_root":"data/cifar10", "num_epochs": 100, "learning_rate": 0.08, "get_optimizer": "optimizer.get_optimizer", "batch_size": 256}'
  pyklopp eval train/train_"$i"/trained.pth my_dataset.get_dataset_test --config='{"dataset_root":"data/cifar10"}' --save="train/train_$i/eval_$i/config.json"
  python3 utils.py --run="$i"
done
#rm -rf matrices.txt
