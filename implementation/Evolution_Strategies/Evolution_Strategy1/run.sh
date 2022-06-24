bash clean.sh
pwd

rm -rf random-matrices.txt
rm -rf specs.txt
rm -rf specs/*

touch matrices.txt
echo generation,spec,accuracy,f1,loss,precision,recall >results.csv

for g in $(seq 0 15); do
    if [ "$g" -eq 0 ]; then
      r=800
    else
      r=50
    fi
  for i in $(seq 0 $r); do
    printf '\n------------------\n--- Gen '$g' Run '$i' ---\n------------------\n'
    pyklopp init my_model --save "train/train_$i/model.pth" --config='{"generation":'''$g'''}'
    pyklopp train train/train_"$i"/model.pth my_dataset.get_dataset_train --save="train/train_$i/trained.pth" --config='{"dataset_root":"data/cifar10", "num_epochs": 100, "learning_rate": 0.08, "get_optimizer": "optimizer.get_optimizer", "batch_size": 256}'
    pyklopp eval train/train_"$i"/trained.pth my_dataset.get_dataset_test --config='{"dataset_root":"data/cifar10"}' --save="train/train_$i/eval_$i/config.json"
    python3 utils.py --run="$i" --gen="$g"
  done
  if [ "$g" -ne 0 ]; then
    echo "Yep"
    python3 temp.py --reps=$r --generation=$g
  fi
  rm -rf train
  rm -rf specs.txt
  rm -rf matrices.txt
done