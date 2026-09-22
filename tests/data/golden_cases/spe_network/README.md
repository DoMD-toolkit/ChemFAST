# Reduced SPE-network golden case

This is a deterministic minimal regression fixture derived from the reaction chemistry in the legacy `spe_network.zip` case.

It contains one connected B–A–A–C–A–A–B network plus isolated Li, TFSI and nitrile-solvent components. The same C bead participates in two C–A reactions, so the fixture exercises multifunctional crosslinking. The isolated L/T/S species retain the multicomponent character without making the default pytest suite large.
