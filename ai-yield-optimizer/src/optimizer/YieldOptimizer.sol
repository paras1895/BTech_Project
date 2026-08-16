// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IERC20} from "../interfaces/IERC20.sol";
import {AaveAdapter} from "../adapters/AaveAdapter.sol";

contract YieldOptimizer {
    struct Position {
        uint256 strategyId;
        uint256 principal;
        uint256 timestamp;
    }

    address public owner;

    mapping(uint256 => AaveAdapter) public adapters;

    mapping(address => Position) public positions;

    event Deposited(
        address indexed user,
        uint256 indexed strategyId,
        uint256 amount
    );

    event Withdrawn(
        address indexed user,
        uint256 indexed strategyId,
        uint256 amount
    );

    modifier onlyOwner() {
        require(msg.sender == owner, "NOT_OWNER");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function registerAdapter(
        uint256 strategyId,
        address adapter
    ) external onlyOwner {
        require(adapter != address(0), "ZERO_ADAPTER");

        adapters[strategyId] = AaveAdapter(adapter);
    }

    function deposit(
        uint256 strategyId,
        address asset,
        uint256 amount
    ) external {
        require(amount > 0, "ZERO_AMOUNT");

        AaveAdapter adapter = adapters[strategyId];

        require(
            address(adapter) != address(0),
            "STRATEGY_NOT_REGISTERED"
        );

        IERC20(asset).transferFrom(
            msg.sender,
            address(this),
            amount
        );

        IERC20(asset).approve(
            address(adapter),
            amount
        );

        adapter.supply(
            asset,
            amount,
            address(this)
        );

        positions[msg.sender] = Position({
            strategyId: strategyId,
            principal: amount,
            timestamp: block.timestamp
        });

        emit Deposited(
            msg.sender,
            strategyId,
            amount
        );
    }

    function withdraw(
        uint256 strategyId,
        address asset,
        uint256 amount
    ) external {
        Position memory position = positions[msg.sender];

        require(
            position.strategyId == strategyId,
            "WRONG_STRATEGY"
        );

        require(
            amount <= position.principal,
            "EXCEEDS_PRINCIPAL"
        );

        AaveAdapter adapter = adapters[strategyId];

        adapter.withdraw(
            asset,
            amount,
            msg.sender
        );

        if (amount == position.principal) {
            delete positions[msg.sender];
        } else {
            positions[msg.sender].principal -= amount;
        }

        emit Withdrawn(
            msg.sender,
            strategyId,
            amount
        );
    }

    function getPosition(
        address user
    ) external view returns (Position memory) {
        return positions[user];
    }
}
