from setuptools import setup, find_packages

setup(name='drlt',
		version='1.0.0',
		url='https://github.com/gwangmin/ReinforcementLearning',
		author='gwangmin',
		author_email='ygm.gwangmin@gmail.com',
		license='MIT',
		description='Provide Deep Reinforcement Learning agents.',
		long_discription=open('README.md','r').read(),
		packages=find_packages(),
		zip_safe=False,
		install_requires=[
		'tensorflow>=1.0.0','Keras>=2.0.2','gym>=0.9.2','numpy>=1.12.1','matplotlib>=2.0.0',
		]
		)
