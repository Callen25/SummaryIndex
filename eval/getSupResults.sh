for result in ./supplement_results/*
do
  echo $result
  ./trec_eval-9.0.7/trec_eval -q groundTruth.txt $result > trec_sup_output/$(basename $result)
done

