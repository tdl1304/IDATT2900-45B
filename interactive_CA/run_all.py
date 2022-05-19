import subprocess

adam_nonsample_models = [
    ['final_models\\Adam\\NonSamplePools\\9-CARROT-train_05-05-2022_17-55-17\\models\\model_19500.pt', '🥕', 9, False],
    ['final_models\\Adam\\NonSamplePools\\9-RABBIT-FACE-train_05-05-2022_16-07-21\\models\\model_19500.pt', '🐰', 9, False],
    ['final_models\\Adam\\NonSamplePools\\15-CARROT-train_05-05-2022_17-31-03\\models\\model_19500.pt', '🥕', 15, False],
    ['final_models\\Adam\\NonSamplePools\\15-RABBIT-FACE-train_05-05-2022_14-39-24\\models\\model_19500.pt', '🐰', 15, False]
]

adam_sample_models = [
    ['final_models\\Adam\\SamplePools\\9-CARROT-train_05-05-2022_17-04-48\\models\\model_19500.pt', '🥕', 9, False],
    ['final_models\\Adam\\SamplePools\\9-RABBIT-FACE-train_05-05-2022_11-43-14\\models\\model_19500.pt', '🐰', 9, False],
    ['final_models\\Adam\\SamplePools\\15-CARROT-train_05-05-2022_17-00-31\\models\\model_19500.pt', '🥕', 15, False],
    ['final_models\\Adam\\SamplePools\\15-RABBIT-FACE-train_05-05-2022_13-44-43\\models\\model_19500.pt', '🐰', 15, False]
]

es_nonsample_models = [
    ['final_models\\ES\\NonSamplePools\\9-CARROT-train_05-05-2022_09-14-06\\models\\model_2212000', '🥕', 9, True],
    ['final_models\\ES\\NonSamplePools\\9-RABBIT-FACE-train_06-05-2022_12-18-56\\models\\model_1999000', '🐰', 9, True ],
    ['final_models\\ES\\NonSamplePools\\15-CARROT-train_06-05-2022_11-06-58\\models\\model_1999000', '🥕', 15, True],
    ['final_models\\ES\\NonSamplePools\\15-RABBIT-FACE-train_06-05-2022_12-21-00\\models\\model_1999000', '🐰', 15, True]
]

es_sample_models = [
    ['final_models\\ES\\SamplePools\\9-CARROT-train_29-04-2022_11-33-16\\models\\model_1036000', '🥕', 9, True],
    ['final_models\\ES\\SamplePools\\9-RABBIT-FACE-train_01-05-2022_10-32-24\\models\\model_1157000', '🐰', 9, True],
    ['final_models\\ES\\SamplePools\\15-CARROT-train_29-04-2022_11-18-06\\models\\model_1105000', '🥕', 15, True],
    ['final_models\\ES\\SamplePools\\15-RABBIT-FACE-train_01-05-2022_10-32-40\\models\\model_1126000', '🐰', 15, True]
]

models = [adam_nonsample_models, adam_sample_models,
          es_nonsample_models, es_sample_models]

if __name__ == '__main__':
    # Run all models:
    # for i in models:
    #     for model in i:
    #         load_model = model[0]
    #         emoji = model[1]
    #         size = model[2]
    #         es = model[3]
    #         command = "python .\interactive_CA\main.py -i %s -s %i -l %s -e %r" % (emoji, size, load_model, es)
    #         subprocess.run(command)

    # # Run single model:
    model = models[3][2]
    load_model = model[0]
    emoji = model[1]
    size = model[2]
    es = model[3]
    command = "python .\interactive_CA\main.py -i %s -s %i -l %s -e %r" % (emoji, size, load_model, es)
    subprocess.run(command)