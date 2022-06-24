for g in $(seq 0 2); do
  if [ "$g" -eq 0 ]; then
    r=99
    echo $r
  else
    r=4
    echo $r
  fi
done