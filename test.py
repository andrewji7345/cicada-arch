import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow.keras.optimizers import Adam
import numpy as np
import optuna

from models import TeacherAutoencoder, CicadaV1, CicadaV2, Trial

def main() -> None:
    teacher = TeacherAutoencoder((18, 14, 1)).get_model()
    teacher.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

    cicada_v1 = CicadaV1((252,)).get_model()
    cicada_v1.compile(optimizer=Adam(learning_rate=0.001), loss="mae")

    cicada_v2 = CicadaV2((252,)).get_model()
    cicada_v2.compile(optimizer=Adam(learning_rate=0.001), loss="mae")

    print(optuna.study.get_all_study_names(f"sqlite:///arch/example.db"))

    #print(f'deprecated size: {cicada_v2.count_params()}')

    #print(f'number of layers: {len(cicada_v2.layers)}')
    #for layer in cicada_v2.layers:
    #    for sublayer in layer.weights:
    #        print(np.array(sublayer).shape)

    
    #print(type(cicada_v1.layers[0].weights))
    #print(type(cicada_v1.layers[1]))
    #print(type(cicada_v1.layers[2]))
    #print(size(teacher))
    #print(size(cicada_v1))
    #print(size(cicada_v2))

if __name__ == "__main__":
    main()