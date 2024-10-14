from sklearn.pipeline import Pipeline
import LoadWav
import AudioCleaner
import AudioNormalizer
import SpectogramGenerator
import Classifier

def main():
    training_pipeline = Pipeline(steps=[
        ('LoadWav', LoadWav()), 
        ('AudioCleaner', AudioCleaner()),
        ('AudioNormalizer', AudioNormalizer()),
        ('SpectogramGenerator', SpectogramGenerator()),
        ('Classifier', Classifier())
    ])

    