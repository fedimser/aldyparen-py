result=0

echo "Running black..."
black --check ./aldyparen
result+=$?

echo "Running pyright..."
pyright ./aldyparen
result+=$?

exit $result