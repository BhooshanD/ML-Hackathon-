# PathSnap – Dataset Exploration and Data Quality Report

## 1. Dataset Overview

The PathSnap dataset contains two partitions:

| Partition | Number of Trips |
|---|---:|
| Training | 80 |
| Student Test | 10 |
| **Total** | **90** |

Each trip contains four files:

- `.track` – GPS observations containing longitude, latitude and timestamp.
- `.nodes` – Road-network node coordinates.
- `.arcs` – Source and destination node relationships.
- `.route` – Reference route represented by arc IDs.

The dataset contains 227,577 GPS observations across the 90 trips.

## 2. GPS Observation Analysis

| Partition | Trips | Mean GPS Points | Median | Minimum | Maximum |
|---|---:|---:|---:|---:|---:|
| Training | 80 | 2,605.21 | 1,529 | 361 | 19,313 |
| Student Test | 10 | 1,916.00 | 1,420 | 262 | 3,958 |

The mean number of GPS observations exceeds the median in both partitions, indicating variation in trip lengths and a right-skewed distribution.

## 3. Sampling Interval Analysis

| Metric | Training | Student Test |
|---|---:|---:|
| Mean interval (seconds) | 5.102 | 1.091 |
| Median interval (seconds) | 1.0 | 1.0 |
| Minimum interval (seconds) | 1.0 | 1.0 |
| Maximum interval (seconds) | 2,706.4 | 25.5 |
| Average gaps over 1.5 seconds per trip | 90.91 | 26.40 |

Both partitions have a median sampling interval of one second. However, training trips have a higher mean interval, indicating occasional large timestamp gaps.

No negative or zero timestamp differences were detected.

## 4. Data Quality Assessment

The following checks were performed on GPS observations:

- Missing longitude, latitude and timestamp values.
- Longitude and latitude values outside valid geographic ranges.
- Zero or negative timestamps.
- Duplicate GPS rows.

No occurrences of these issues were detected in either partition.

| Data Quality Check | Training | Student Test |
|---|---:|---:|
| Missing longitude | 0 | 0 |
| Missing latitude | 0 | 0 |
| Missing timestamp | 0 | 0 |
| Invalid longitude | 0 | 0 |
| Invalid latitude | 0 | 0 |
| Zero/negative timestamps | 0 | 0 |
| Duplicate GPS rows | 0 | 0 |

These findings represent the results of the checks performed and do not establish the absence of every possible data-quality issue.

## 5. Road Network Analysis

| Metric | Training | Student Test |
|---|---:|---:|
| Average nodes per trip | 24,168.56 | 9,068.90 |
| Average arcs per trip | 48,461.30 | 18,504.10 |
| Average connected components | 39.01 | 22.70 |
| Average isolated nodes | 0 | 0 |

Road-network analysis was performed using NetworkX and an undirected graph representation.

All source and target node references in the `.arcs` files were validated against the corresponding `.nodes` files. No invalid references were detected.

## 6. Geographic Coverage

| Metric | Training | Student Test |
|---|---:|---:|
| Mean longitude range | 0.1873° | 0.1500° |
| Mean latitude range | 0.1078° | 0.0831° |
| Maximum longitude range | 1.1039° | 0.7150° |
| Maximum latitude range | 0.6781° | 0.2412° |

Training trips show greater average geographic spread than student-test trips.

The trips cover different geographic regions, so individual trip visualisations are more meaningful than plotting all trajectories as a single local route.

## 7. Methodology and Assumptions

1. GPS files were interpreted as longitude, latitude and timestamp columns.
2. Road-network node files were interpreted as longitude and latitude columns.
3. Arc files were interpreted as source-node and destination-node IDs.
4. Route IDs were validated as zero-based row indices into the corresponding arc file for the inspected trip.
5. Connected components were calculated using undirected NetworkX graphs.
6. Timestamp gaps greater than 1.5 seconds were flagged for exploratory analysis. This threshold is not a confirmed dataset rule.
7. Road-network map visualisations use sampled segments for display efficiency; the original network files remain unchanged.
8. The student-test partition was inspected for dataset inventory and quality assessment only and was not used for model training or tuning.

## 8. Conclusion

The dataset inventory confirms 90 complete trip folders with four expected files per trip. The GPS quality checks found no missing values, invalid geographic coordinates, non-positive timestamps or duplicate GPS observations.

The analysis also identified variable trip lengths, occasional timestamp gaps, differences in road-network size and geographic coverage between partitions.

These findings provide an exploratory foundation for subsequent PathSnap route modelling and visualisation.
