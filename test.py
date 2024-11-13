import unittest
import numpy as np
from models.nn_model_oop import sigmoid, softmax, simpleNN

"""Useful stuff"""
#python -m unittest discover -s tests

class TestNN_functions(unittest.TestCase):
    #
    def setUp(self):
        # Initialize variables for tests
        self.x = [1,-1]
        self.target_class = 0
        self.nn_model = simpleNN()
        self.y_pred = self.nn_model.forward(self.x)

    
    def test_sigmoid(self):
        x = [0, -1]
        result_1 = sigmoid(x[0])
        result_2 = sigmoid(x[-1])
        self.assertEqual(result_1,0.5)
        self.assertGreater(result_2,0)

    def test_softmax(self):
        v = np.random.rand(10)
        v_neg =v*-1
        r_1 = softmax(v)
        r_2 = softmax(v_neg)
        self. assertAlmostEqual(sum(r_1),1)
        self. assertAlmostEqual(sum(r_2),1)
    
    def test_forwardpass(self):
        self.assertGreater(sum(self.y_pred),0)
        self.assertAlmostEqual(sum(self.y_pred),1.)
        self.assertEqual(len(self.y_pred),2)
        return self.y_pred

    def test_loss(self):
        loss = self.nn_model.calculate_loss(self.target_class)
        self.assertGreater(loss,0)
    

if __name__ == "__main__":
    # Set verbosity equals 2 to view the output from the test
    unittest.main(verbosity=2)
    """Uncomment this if you want to test tests individually add the desired tests in the following list """

    #test_list = ['test_sigmoid','test_softmax','test_forwardpass']
    #suite = unittest.TestSuite()
    #for test in test_list:
    #    suite.addTest(TestNN_functions(test))
    #suite.addTest(TestNN_functions('test_sigmoid'))
    #suite.addTest(TestNN_functions('test_softmax'))
    #suite.addTest(TestNN_functions('test_backpropagation'))

    #runner = unittest.TextTestRunner()
    #runner.run(suite)


    