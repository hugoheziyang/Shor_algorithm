from dataclasses import dataclass
import math


@dataclass
class ShorConfig:
    """
    High-level runtime settings for the Shor simulator.
    """
    num_phase_bits: int | None = None
    max_attempts: int = 10
    rng_seed: int | None = 1234
    max_multiple: int | None = None
    verbose: bool = False

    def resolved_num_phase_bits(self, N: int) -> int:
        """
        If num_phase_bits is not explicitly set, use the default choice 2 * ceil(log2 N).
        """
        if self.num_phase_bits is not None:
            return self.num_phase_bits
        return 2 * math.ceil(math.log2(N))