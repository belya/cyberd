package storage

import (
	"encoding/binary"
	sdk "github.com/cosmos/cosmos-sdk/types"
)

var cidsCount = []byte("cids_count")

type CidIndexStorage struct {
	mainStoreKey *sdk.KVStoreKey
	indexKey     *sdk.KVStoreKey
}

func NewCidIndexStorage(mainStoreKey *sdk.KVStoreKey, indexKey *sdk.KVStoreKey) CidIndexStorage {
	return CidIndexStorage{
		mainStoreKey: mainStoreKey,
		indexKey:     indexKey,
	}
}

// CIDs index is array of all added CIDs, sorted asc by first link time.
//   - for given link, CIDs added in order [CID1, CID2] (if they both new to chain)
// This method performs lookup of CIDs, returns index value, or create and put in index new value if not exists.
func (cis CidIndexStorage) GetOrPutCidIndex(ctx sdk.Context, cid Cid) CidNumber {

	cidsIndex := ctx.KVStore(cis.indexKey)

	cidAsBytes := []byte(cid)
	cidIndexAsBytes := cidsIndex.Get(cidAsBytes)

	// new cid, get new index
	if cidIndexAsBytes == nil {

		lastIndex := cis.GetCidsCount(ctx)
		lastIndexAsBytes := make([]byte, 8)
		binary.LittleEndian.PutUint64(lastIndexAsBytes, lastIndex)

		mainStore := ctx.KVStore(cis.mainStoreKey)
		cidsIndex.Set(cidAsBytes, lastIndexAsBytes)
		mainStore.Set(cidsCount, lastIndexAsBytes)
		return CidNumber(lastIndex)
	}

	return CidNumber(binary.LittleEndian.Uint64(cidIndexAsBytes))
}

// returns overall added cids count
func (cis CidIndexStorage) GetCidsCount(ctx sdk.Context) uint64 {

	mainStore := ctx.KVStore(cis.mainStoreKey)
	lastIndexAsBytes := mainStore.Get(cidsCount)

	if lastIndexAsBytes == nil {
		return 0
	}

	return binary.LittleEndian.Uint64(lastIndexAsBytes) + 1
}

// returns all added cids
func (cis CidIndexStorage) GetFullCidsIndex(ctx sdk.Context) map[Cid]CidNumber {

	cidsIndex := ctx.KVStore(cis.indexKey)
	iterator := cidsIndex.Iterator(nil, nil)

	index := make(map[Cid]CidNumber)

	for iterator.Valid() {
		index[Cid(iterator.Key())] = CidNumber(binary.LittleEndian.Uint64(iterator.Value()))
		iterator.Next()
	}
	iterator.Close()
	return index
}
