// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

contract ScenarioRunner {
    struct Observation {
        uint256 timestamp;
        uint256 userId;
        uint256 strategyId;

        uint256 capital;

        uint256 supplyRate;
        uint256 borrowRate;
        uint256 utilization;

        uint256 action;
        uint256 amount;

        int256 reward;
    }

    Observation[] public observations;

    event ObservationRecorded(
        uint256 indexed id,
        uint256 indexed userId,
        uint256 indexed strategyId,
        uint256 action,
        uint256 amount,
        int256 reward
    );

    function record(
        uint256 userId,
        uint256 strategyId,
        uint256 capital,
        uint256 supplyRate,
        uint256 borrowRate,
        uint256 utilization,
        uint256 action,
        uint256 amount,
        int256 reward
    ) external {
        uint256 id = observations.length;

        observations.push(
            Observation({
                timestamp: block.timestamp,
                userId: userId,
                strategyId: strategyId,
                capital: capital,
                supplyRate: supplyRate,
                borrowRate: borrowRate,
                utilization: utilization,
                action: action,
                amount: amount,
                reward: reward
            })
        );

        emit ObservationRecorded(
            id,
            userId,
            strategyId,
            action,
            amount,
            reward
        );
    }

    function observationCount()
        external
        view
        returns (uint256)
    {
        return observations.length;
    }

    function getObservation(
        uint256 id
    ) external view returns (Observation memory) {
        return observations[id];
    }
}
