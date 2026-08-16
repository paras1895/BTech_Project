// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

contract MarketSimulator {
    struct MarketState {
        uint256 timestamp;
        uint256 utilization;
        uint256 supplyRate;
        uint256 borrowRate;
        uint256 price;
    }

    MarketState public state;

    event MarketUpdated(
        uint256 timestamp,
        uint256 utilization,
        uint256 supplyRate,
        uint256 borrowRate,
        uint256 price
    );

    function setMarketState(
        uint256 utilization,
        uint256 supplyRate,
        uint256 borrowRate,
        uint256 price
    ) external {
        require(utilization <= 1e18, "BAD_UTILIZATION");

        state = MarketState({
            timestamp: block.timestamp,
            utilization: utilization,
            supplyRate: supplyRate,
            borrowRate: borrowRate,
            price: price
        });

        emit MarketUpdated(
            block.timestamp,
            utilization,
            supplyRate,
            borrowRate,
            price
        );
    }

    function getState()
        external
        view
        returns (MarketState memory)
    {
        return state;
    }
}
