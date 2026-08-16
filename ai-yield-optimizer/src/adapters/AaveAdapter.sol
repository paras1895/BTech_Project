// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IAavePool} from "../interfaces/IAavePool.sol";
import {IERC20} from "../interfaces/IERC20.sol";

contract AaveAdapter {
    IAavePool public immutable pool;

    constructor(address _pool) {
        require(_pool != address(0), "ZERO_POOL");
        pool = IAavePool(_pool);
    }

    function supply(
        address asset,
        uint256 amount,
        address receiver
    ) external {
        IERC20(asset).approve(address(pool), amount);

        pool.supply(
            asset,
            amount,
            receiver,
            0
        );
    }

    function withdraw(
        address asset,
        uint256 amount,
        address receiver
    ) external returns (uint256) {
        return pool.withdraw(
            asset,
            amount,
            receiver
        );
    }

    function getSupplyRate(
        address asset
    ) external view returns (uint256) {
        (
            ,
            ,
            uint128 liquidityRate,
            ,
            ,
            ,
            ,
            ,
            ,
            ,
            ,
            ,
            ,
        ) = pool.getReserveData(asset);

        return uint256(liquidityRate);
    }
}
