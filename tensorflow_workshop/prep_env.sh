pip uninstall -y tensorflow
pip uninstall -y sonnet
pip install /tmp/sonnet/*.whl

export PYTHONPATH=/sonnet:$PYTHONPATH

unzip ./data/GMIC\ TF-20170424T050737Z-001.zip -d ./
mv GMIC\ TF/* data/
rm -r GMIC\ TF/
rm data/GMIC\ TF-20170424T050737Z-001.zip