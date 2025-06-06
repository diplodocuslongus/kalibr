# docker

docker build -t kalibr -f Dockerfile_ros1_20_04 .


Ended ok (had multiple errors earlier in the day, apparently due to failing to download apt related installs). This build ended with :

2 warnings found (use docker --debug to expand):
 - LegacyKeyValueFormat: "ENV key=value" should be used instead of legacy "ENV key value" format (line 19)
 - JSONArgsRecommended: JSON arguments recommended for ENTRYPOINT to prevent unintended behavior related to OS signals (line 38)


