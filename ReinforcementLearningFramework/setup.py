from setuptools import setup, find_packages

setup(name='drlt',
		version='1.0.6',
		url='https://github.com/gwangmin/ReinforcementLearning/tree/master/drlt',
		author='gwangmin',
		author_email='ygm.gwangmin@gmail.com',
		license='MIT',
		description='Provide Deep Reinforcement Learning agents.',
		long_discription=open('README.md','r').read(),
		packages=find_packages(),
		zip_safe=False,
		install_requires=[
		'tensorflow==2.5.3','Keras>=2.2.4','gym>=0.9.2','numpy>=1.16.4','matplotlib>=2.0.0'
		]
		)
