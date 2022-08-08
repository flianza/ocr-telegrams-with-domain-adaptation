from tllib.utils.meter import ProgressMeter


class StringProgressMeter(ProgressMeter):
    def __init__(self, num_batches, meters, prefix=""):
        super().__init__(num_batches, meters, prefix)

    def display(self, batch) -> str:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return "\t".join(entries)
