result=0

echo "Running black..."
black --check ./aldyparen
result+=$?

echo "Running pylint..."
pylint ./aldyparen
result+=$?

echo "Running mypy..."
mypy ./aldyparen
result+=$?

exit $result