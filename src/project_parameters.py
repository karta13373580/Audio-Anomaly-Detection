# import
import os
import sys
sys.path.append(os.path.realpath('DeepLearningTemplate/'))
from DeepLearningTemplate.project_parameters import ProjectParameters

if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # display each parameter
    for name, value in vars(project_parameters).items():
        print('{:<20}= {}'.format(name, value))
