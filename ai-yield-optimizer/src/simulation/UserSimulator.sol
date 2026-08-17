// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

contract UserSimulator {
    struct UserProfile {
        uint256 capital;
        uint256 riskTolerance;
        uint256 minYield;
        uint256 maxAllocation;
        bool active;
    }

    mapping(uint256 => UserProfile) public users;

    uint256 public userCount;

    event UserCreated(
        uint256 indexed userId,
        uint256 capital,
        uint256 riskTolerance,
        uint256 minYield,
        uint256 maxAllocation
    );

    function createUser(
        uint256 capital,
        uint256 riskTolerance,
        uint256 minYield,
        uint256 maxAllocation
    ) external returns (uint256 userId) {
        require(capital > 0, "ZERO_CAPITAL");
        require(riskTolerance <= 10000, "BAD_RISK");
        require(maxAllocation <= 10000, "BAD_ALLOCATION");

        userId = userCount++;

        users[userId] = UserProfile({
            capital: capital,
            riskTolerance: riskTolerance,
            minYield: minYield,
            maxAllocation: maxAllocation,
            active: true
        });

        emit UserCreated(
            userId,
            capital,
            riskTolerance,
            minYield,
            maxAllocation
        );
    }

    function setActive(
        uint256 userId,
        bool active
    ) external {
        require(userId < userCount, "INVALID_USER");

        users[userId].active = active;
    }

    function getUser(
        uint256 userId
    ) external view returns (UserProfile memory) {
        return users[userId];
    }
}
