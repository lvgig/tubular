import cProfile
import pstats
import time

# stop console filling up with future warnings & mappingtransformer warnings
import warnings

import create_dataset as create
import numpy as np
import profile_helpers as ph
from pipeline_generator import TubularPipelineGenerator

warnings.simplefilter(action="ignore")


if __name__ == "__main__":
    print("***Use time function to inspect transform calls")
    pipe_1 = TubularPipelineGenerator()
    pipe_1 = pipe_1.generate_pipeline()

    df_1 = create.create_standard_pandas_dataset()

    pipe_1.fit(df_1, df_1["AveRooms"])

    pd_rows_to_transform = []
    for _index in range(1000):
        pd_data_row = df_1.iloc[[0]].copy()
        pd_rows_to_transform.append(pd_data_row.copy())

    standard_tubular_times = []

    for i in range(1000):
        start = time.process_time()
        pipe_1.transform(pd_rows_to_transform[i])
        standard_tubular_times.append(time.process_time() - start)

    print("Tubular mean CPU time and SD")
    print(np.mean(standard_tubular_times))
    print(np.std(standard_tubular_times))

    print("***Use cprofile to inspect transform calls")

    df_2 = create.create_standard_pandas_dataset()

    pipe_2 = TubularPipelineGenerator()
    pipe_2 = pipe_2.generate_pipeline()

    pipe_2.fit(df_2, df_2["AveRooms"])

    pd_rows_to_transform_2 = []
    for _index in range(1000):
        pd_data_row = df_2.iloc[[0]].copy()
        pd_rows_to_transform_2.append(pd_data_row.copy())

    cprofile_results = []
    for i in range(1000):
        with cProfile.Profile() as profile:
            pipe_2.transform(pd_rows_to_transform_2[i])

        cprofile_result = pstats.Stats(profile)
        cprofile_results.append(cprofile_result)

    ph.profiling_wrapper(cprofile_results)
