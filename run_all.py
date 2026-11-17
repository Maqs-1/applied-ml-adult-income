"""Lance toute la chaîne dans l'ordre."""
from src import calibration, data, evaluate, explain, fairness, train

if __name__ == "__main__":
    import sys
    sys.argv = sys.argv[:1]
    for step in (data, train, evaluate, explain, calibration, fairness):
        print(f"\n######## {step.__name__}")
        step.main()
