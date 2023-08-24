import os


from loguru import logger


class TrainLog:
    def __init__(self, mode, training_args, is_master):
        if not is_master:
            self.enable = False
        else:

            train_dir = os.path.join(
                training_args.train_dir, training_args.signature)

            if not os.path.exists(train_dir):

                os.makedirs(train_dir)

            # reset training_args.train_dir to the actual train_dir, which is a subdir of the original train_dir
            training_args.train_dir = train_dir
            self.enable = True
            self.mode = mode
            self.train_dir = train_dir
            self.uuid_str = training_args.signature
            self.log_file = os.path.join(self.train_dir, f"{self.mode}.log")
            self.trace = logger.add(self.log_file,
                                    format='[{time:YYYY-MM-DD HH:mm:ss} | {name} | {level}] [' +
                                    self.mode + '-LOGGER:] {message}',
                                    backtrace=True,
                                    diagnose=True,
                                    catch=True,
                                    encoding="utf-8",
                                    enqueue=False)
            logger.bind(name=self.uuid_str)

            self.logger = logger

    def remove(self):
        if self.enable:
            self.logger.remove(self.trace)
            self.enable = False
