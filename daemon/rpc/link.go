package rpc

import "github.com/cosmos/cosmos-sdk/types"
import cbd "github.com/cybercongress/cyberd/app/types"

func IsLinkExist(from string, to string, address string) (bool, error) {

	accAddress, err := types.AccAddressFromBech32(address)
	if err != nil {
		return false, err
	}

	return cyberdApp.IsLinkExist(cbd.Cid(from), cbd.Cid(to), accAddress), nil
}
