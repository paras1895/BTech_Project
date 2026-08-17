// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Script} from "forge-std/Script.sol";
import {AaveAdapter} from "../src/adapters/AaveAdapter.sol";
import {YieldOptimizer} from "../src/optimizer/YieldOptimizer.sol";
import {StrategyRegistry} from "../src/optimizer/StrategyRegistry.sol";
import {UserSimulator} from "../src/simulation/UserSimulator.sol";
import {MarketSimulator} from "../src/simulation/MarketSimulator.sol";
import {ScenarioRunner} from "../src/simulation/ScenarioRunner.sol";

contract Deploy is Script {
    function run()
        external
        returns (
            AaveAdapter adapter,
            YieldOptimizer optimizer,
            StrategyRegistry registry,
            UserSimulator users,
            MarketSimulator market,
            ScenarioRunner scenarios
        )
    {
        address pool = vm.envAddress("AAVE_POOL");

        vm.startBroadcast();

        adapter = new AaveAdapter(pool);

        optimizer = new YieldOptimizer();

        registry = new StrategyRegistry();

        users = new UserSimulator();

        market = new MarketSimulator();

        scenarios = new ScenarioRunner();

        registry.addStrategy(
            "Aave",
            address(adapter),
            address(0)
        );

        optimizer.registerAdapter(
            0,
            address(adapter)
        );

        vm.stopBroadcast();
    }
}
