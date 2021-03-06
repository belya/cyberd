package rank

import (
	. "github.com/cybercongress/cyberd/app/storage"
	. "github.com/cybercongress/cyberd/app/types"
	"sync"
)

const (
	d         float64 = 0.85
	tolerance float64 = 1e-3
)

func calculateRankCPU(data *InMemoryStorage) ([]float64, int) {

	inLinks := data.GetInLinks()

	size := data.GetCidsCount()
	if size == 0 {
		return []float64{}, 0
	}

	rank := make([]float64, size)
	defaultRank := (1.0 - d) / float64(size)
	danglingNodesSize := uint64(0)

	for i := range rank {
		rank[i] = defaultRank
		if len(inLinks[CidNumber(i)]) == 0 {
			danglingNodesSize++
		}
	}

	innerProductOverSize := defaultRank * (float64(danglingNodesSize) / float64(size))
	defaultRankWithCorrection := float64(d*innerProductOverSize) + defaultRank

	change := tolerance + 1

	steps := 0
	prevrank := make([]float64, 0)
	prevrank = append(prevrank, rank...)
	for change > tolerance {
		rank = step(defaultRankWithCorrection, prevrank, data)
		change = calculateChange(prevrank, rank)
		prevrank = rank
		steps++
	}

	return rank, steps
}

func step(defaultRankWithCorrection float64, prevrank []float64, data *InMemoryStorage) []float64 {

	rank := append(make([]float64, 0, len(prevrank)), prevrank...)

	var wg sync.WaitGroup
	wg.Add(len(data.GetInLinks()))

	for cid := range data.GetInLinks() {

		go func(i CidNumber) {
			defer wg.Done()
			_, sortedCids, ok := data.GetSortedInLinks(i)

			if !ok {
				rank[i] = defaultRankWithCorrection
			} else {
				ksum := float64(0)
				for _, j := range sortedCids {
					//todo add pre-calculation of overall stake for cid and links
					linkStake := data.GetOverallLinkStake(j, i)
					jCidOutStake := data.GetOverallOutLinksStake(j)
					weight := float64(linkStake) / float64(jCidOutStake)
					ksum = float64(prevrank[j]*weight) + ksum //force no-fma here by explicit conversion
				}

				rank[i] = float64(ksum*d) + defaultRankWithCorrection //force no-fma here by explicit conversion
			}

		}(CidNumber(cid))
	}
	wg.Wait()
	return rank
}

func calculateChange(prevrank, rank []float64) float64 {

	maxDiff := 0.0
	diff := 0.0
	for i, pForI := range prevrank {
		if pForI > rank[i] {
			diff = pForI - rank[i]
		} else {
			diff = rank[i] - pForI
		}
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	return maxDiff
}
