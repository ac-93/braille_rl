from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(name='braille_rl',
      version='0.1',
      description='Deep reinforcement learning for tactile robotics, learning to type on a braille keyboard.',
      long_description=long_description,
      url='http://github.com/ac-93/braille_rl',
      author='Alex Church',
      author_email='alexchurch1993@gmail.com',
      license='MIT',
      packages=['braille_rl'],
      install_requires=[
          'evdev',
          'scikit-image',
          'scikit-learn'
      ],
      zip_safe=False)
