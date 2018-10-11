# Coders
This directory contains implementations of the [MLPEncoder](mlp_coder.py),
[ConvEncoder](conv_encoder.py), and [MLP Decoder](mlp_coder.py) proposed in the
paper. All encoders and decoders derive from a common [Coder](coder.py) class.

### Adding a new encoder/decoder
To add a new encoder/decoder, implement a class that derives from [Coder](coder.py),
and add the classpath of your encoder/decoder to a configuration file, passing along
any arguments (as described in [conf](../conf)) that are added outside of those currently in Coder.
