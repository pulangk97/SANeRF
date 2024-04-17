method=sanerf
datatype=blender
numviews=8
base_dir=./config_files/$method/$datatype$numviews

for gin_file in "$base_dir"/*.gin
do
  echo "Running $gin_file..."
  python3 main.py --ginc "$gin_file" 
done