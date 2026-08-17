// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

contract StrategyRegistry {
    struct Strategy {
        string name;
        address adapter;
        address asset;
        bool active;
    }

    uint256 public strategyCount;

    mapping(uint256 => Strategy) public strategies;

    event StrategyAdded(
        uint256 indexed id,
        string name,
        address adapter,
        address asset
    );

    event StrategyStatusChanged(
        uint256 indexed id,
        bool active
    );

    function addStrategy(
        string calldata name,
        address adapter,
        address asset
    ) external returns (uint256 id) {
        require(adapter != address(0), "ZERO_ADAPTER");
        require(asset != address(0), "ZERO_ASSET");

        id = strategyCount++;

        strategies[id] = Strategy({
            name: name,
            adapter: adapter,
            asset: asset,
            active: true
        });

        emit StrategyAdded(
            id,
            name,
            adapter,
            asset
        );
    }

    function setActive(
        uint256 id,
        bool active
    ) external {
        require(id < strategyCount, "INVALID_ID");

        strategies[id].active = active;

        emit StrategyStatusChanged(
            id,
            active
        );
    }

    function getStrategy(
        uint256 id
    ) external view returns (Strategy memory) {
        require(id < strategyCount, "INVALID_ID");

        return strategies[id];
    }
}
