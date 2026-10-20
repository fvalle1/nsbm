import unittest
import sys
sys.path.append("/".join(__file__.split("/")[:-2]))
from trisbm import trisbm


class TriTest(unittest.TestCase):

    def test_import(self):
        model = trisbm()
        self.assertIsInstance(model, trisbm)

    def test_import_nsbm(self):
        from nsbm import nsbm
        model = nsbm()
        self.assertIsInstance(model, trisbm)
        self.assertIsInstance(model, nsbm)

    def test_make_graph(self):
        import pandas as pd
        import numpy as np

        df = pd.DataFrame(
            index=["w_{}".format(w) for w in range(100)],
            columns=["doc_{}".format(doc) for doc in range(25)],
            data=np.random.randint(0, 10, size=2500).reshape((100, 25
                                                              ))
        )

        model = trisbm()
        model.make_graph(df, lambda w: 1 if int(w.split("_")[1]) < 90 else 2)

    def test_make_bigraph(self):
        import pandas as pd
        import numpy as np

        df = pd.DataFrame(
            index=["w_{}".format(w) for w in range(100)],
            columns=["doc_{}".format(doc) for doc in range(25)],
            data=np.random.randint(0, 10, size=2500).reshape((100, 25
                                                              ))
        )

        model = trisbm()
        model.make_graph_from_BoW_df(df)
        model.fit(B_min=2, verbose=False)

    def test_save_read_graph(self):
        import pandas as pd
        import numpy as np

        df = pd.DataFrame(
            index=["w_{}".format(w) for w in range(100)],
            columns=["doc_{}".format(doc) for doc in range(25)],
            data=np.random.randint(0, 10, size=2500).reshape((100, 25))
        )

        model = trisbm()
        model.make_graph(df, lambda w: 1 if int(w.split("_")[1]) < 90 else 2)
        model.save_graph()
        saved_shape = model._get_shape()
        nb = model.nbranches
        words = model.words
        docs = model.documents
        keywords = model.keywords
        del model
        model = trisbm()
        model.load_graph()
        self.assertEqual(nb, model.nbranches)
        self.assertEqual(list(saved_shape), list(model._get_shape()))
        self.assertListEqual(list(words), list(model.words))
        self.assertListEqual(list(docs), list(model.documents))
        for i in range(nb):
            self.assertListEqual(list(keywords[i]), list(model.keywords[i]))

        import os
        os.system("rm -rf graph.xml.gz")

    def test_make_graph_multiple(self):
        import pandas as pd
        import numpy as np

        df = pd.DataFrame(
            index=["w{}".format(w) for w in range(1000)],
            columns=["doc{}".format(w) for w in range(250)],
            data=np.random.randint(1, 100, 250000).reshape((1000, 250)))

        df_key_list = [
            pd.DataFrame(
                index=["w{}".format(w) for w in range(100 + ik)],
                columns=["doc{}".format(w) for w in range(250)],
                data=np.random.randint(1, 5 + ik, (100 + ik) * 250).reshape((100 + ik, 250))
                )

            for ik in range(3)
        ]

        model = trisbm()
        model.make_graph_multiple_df(df, df_key_list)

    def test_fit(self):
        import pandas as pd
        import numpy as np

        df = pd.DataFrame(
            index=["w_{}".format(w) for w in range(100)],
            columns=["doc_{}".format(doc) for doc in range(25)],
            data=np.random.randint(0, 10, size=2500).reshape((100, 25
                                                              ))
        )

        model = trisbm()
        model.make_graph(df, lambda w: 1 if int(w.split("_")[1]) < 90 else 2)
        model.fit(verbose=False)

    def test_save(self):
        import pandas as pd
        import numpy as np

        df = pd.DataFrame(
            index=["w_{}".format(w) for w in range(100)],
            columns=["doc_{}".format(doc) for doc in range(25)],
            data=np.random.randint(0, 10, size=2500).reshape((100, 25
                                                              ))
        )

        model = trisbm()
        model.make_graph(df, lambda w: 1 if int(w.split("_")[1]) < 90 else 2)
        model.fit(verbose=False)
        model.save_data()


if __name__ == "__main__":
    unittest.main()
