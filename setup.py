from setuptools import setup, find_packages

setup(
    name             = 'aihems',
    version          = '0.0',
    description      = 'AIHEMS',
    author           = 'Shin, Byungjin',
    author_email     = 'byungjin0826@cslee.co.kr',
    url              = 'https://github.com/byungjin0826/ai-hems-new',
    download_url     = 'https://github.com/byungjin0826/ai-hems-new/archive/master.zip',
    install_requires = [ ],
    packages         = find_packages(),
    keywords         = ['home energy', 'home automation'],
    python_requires  = '>=3',
    zip_safe=False,
    classifiers      = [
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ]
)
