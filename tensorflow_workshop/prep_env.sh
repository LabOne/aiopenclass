pip uninstall -y tensorflow
pip uninstall -y sonnet
pip install /tmp/sonnet/*.whl

echo 'export PYTHONPATH=/repos/sonnet:$PYTHONPATH'>>~/.bashrc

unzip ./data/GMIC\ TF-20170424T050737Z-001.zip -d ./
mv GMIC\ TF/* data/
rm -r GMIC\ TF/
rm data/GMIC\ TF-20170424T050737Z-001.zip

mkdir -p models/tensorflow