Repo for the advanced ML class 290C

Includes code and a report of a thorough comparison of a dynamic Bayesian Network and an LSTM Network in Natural Language Generation. We use the a semantically conditioned LSTM network proposed in (http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP199.pdf) and compare it with a DBN proposed in (http://mi.eng.cam.ac.uk/~ky219/papers/mairesse-acl10.pdf).

The code base includes our original implementation of the DBN (bagel folder), while we use the implementation of Wen et al (with minor modifications in the loading of the data) for the LSTM network (RNNLG folder).

For the comparison we use the BAGEL dataset after some post processing (data wrangling folder)

A detailed approach, results, insights and conlcusions are included the Report.pdf file.