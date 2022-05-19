tutorial
======================

Create a fake dataset
######################

.. code-block:: python

    from nsbm import nsbm
    import pandas as pd
    import numpy as np
    df = pd.DataFrame(
        index=["w{}".format(w) for w in range(1000)],
        columns=["doc{}".format(d) for d in range(250)],
        data=np.random.randint(1, 100, 250000).reshape((1000, 250)))
    df_key_list = []
    # an additional feature
    df_key_list.append(
        pd.DataFrame(
            index=["keyword{}".format(w) for w in range(100)],
            columns=["doc{}".format(d) for d in range(250)],
            data=np.random.randint(1, 10, (100, 250)))
    )

    # another additional feature
    df_key_list.append(
        pd.DataFrame(
            index=["author{}".format(w) for w in range(10)],
            columns=["doc{}".format(d) for d in range(250)],
            data=np.random.randint(1, 5, (10, 250)))
    )

    # other features
    df_key_list.append(
        pd.DataFrame(
            index=["feature{}".format(w) for w in range(25)],
            columns=["doc{}".format(d) for d in range(250)],
            data=np.random.randint(1, 5, (25, 250)))
    )

- *df* is a `Bag of Words <https://en.wikipedia.org/wiki/Bag-of-words_model>`_ (BoW) representation of the documents.
- *df_key_list* is a list of (BoW), all of them have to share the same columns (**documents**) in this case *keywords*, *authors* and *features* are the additional (more than words) information about the documents.


Create and fit a model
#######################

Create a model
***************

.. code-block:: python

    model = nsbm()
    model.make_graph_multiple_df(df, df_key_list)

Fit the model
**************

.. code-block:: python
    
    model.fit(n_init=1, B_min=50, verbose=False)

Parameters:

- n_init the number of initializations: olny the one with the shortest DL will be kept
- B_min minimum number of blocks
- B_max maximum number of blocks
- parallel the model will be fitted with heavy parallelization
- verbose if True, print the progress

Save the results
*****************

.. code-block:: python

    model.save_data()