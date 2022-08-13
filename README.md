# NVIDIA Frequency Scanner
Modern NVIDIA GPUs have a memory bus which has inbuilt error correction. This
can mean that performance drops due to the overhead of errors when overclocked
without the errors being shown to the application.

This program scans a range of memory clocks and measures memory performance at
each clock to determine which clock maximizes performance, which may not be
the highest stable clock.
