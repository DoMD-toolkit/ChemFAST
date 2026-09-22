# Au–PEO rigid-hybrid golden case

This fixture is adapted from the legacy `PEOGyatedAuNPs.zip` case and expressed using the current v1 Reaction-DSL keys.

The original CG XML and ReactionPath are retained. The two Au nanoparticles are represented as two filler definitions because the legacy system uses five mapping sites on body 0 and one mapping site on body 1.

The regression test does **not** compare absolute atomistic XYZ coordinates. It checks rigid-body internal distances, the explicitly rotated mapping-atom positions, and the number of rigid–flexible covalent bonds after ReactionPath replay.

During migration, the final legacy event `('b', 186, 296)` is stored as `('b', 296, 186)` so its ordered participants match the current `Au_G, PEO` reaction rule. The underlying CG bond is unchanged.
