"""Center-Aware components for Polar Voronoi Splitter (v1.3 §9.7).

The CA (Center-Aware) subpackage contains the CenterAwareEncoder used
to gate the secondary Polar Voronoi path. The encoder returns whether
input features exhibit a center-biased activation pattern, which is a
proxy for "this region is structurally a Voronoi cell with a peak at
the center and NSEW neighbors of comparable magnitude".
"""
