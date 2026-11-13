---
title: "An SPX call butterfly before expiration"
date: 2026-11-13
description: "How the profit curve of a 7,575 call butterfly changes as time runs out, with an example of a spread bought for 3.30 points and sold for 8.00."
topics: [data-science]
---

A butterfly on the S&P 500 Index (SPX) has a sharp profit peak at expiration. Before expiration, the peak is rounded. A 7,575 call butterfly was bought for 3.30 points and sold for 8.00 on the same day. A pricing model shows how those fills could arise while the index is still below 7,575.

## The trade and its payoff

One long call butterfly buys one 7,550 call, sells two 7,575 calls, and buys one 7,600 call. All four contracts have the same expiration. For a cash-settled index call, the buyer receives the positive difference between the settlement level and the strike, times the contract multiplier. The seller pays the corresponding amount. The outer strikes are 25 points from the center. The spread pays 25 points at 7,575, and the payout falls by one point for each point away from the center until it reaches zero at a wing.

Each SPX point is worth \$100 per contract, so a 3.30-point net entry debit costs \$330 for one spread. At expiration it breaks even at **7,553.30** and **7,596.70**. Its maximum profit is **\$2,170** at 7,575, and its maximum loss is **\$330** at or beyond either wing, before fees. SPX Weeklys (SPXW) use European-style exercise and cash settlement. The settlement value for an expiration comes from the component stocks' closing prices. Opening the spread on its expiration date is a zero-days-to-expiration (0DTE) trade. [Cboe contract specifications](https://cdn.cboe.com/resources/spx/spx-fact-sheet.pdf) · [Options Industry Council call butterfly](https://www.optionseducation.org/strategies/all-strategies/long-call-butterfly)

## What the curve looks like before expiration

![Five modeled profit curves for the 7,575 call butterfly. The rounded peak rises and narrows with less time remaining until it becomes a triangle at expiration.](intraday.svg "Figure 1. Each curve holds time to expiration fixed and varies the index level. A and B mark two model scenarios. Their labels show the spread quote in index points. The vertical axis shows profit or loss in dollars after the entry debit and before fees.")

For each curve, I use the European Black–Scholes model to price the four calls and take the net value of the position. All three strikes use **20% annualized volatility**, with interest and dividend yields set to zero. I divide the time remaining in minutes by 525,600 to express it in years. The vertical axis is **100 × (spread value − 3.30)**. At expiration, the chart uses the spread's cash payout. [Options Industry Council Black–Scholes model](https://prd-web.optionseducation.org/advancedconcepts/black-scholes-formula)

With interest set to zero, the model price is the average of the expiration payouts across possible settlement levels, weighted by their probabilities in the model. With several hours left, a wider range of settlement levels rounds off the peak and gives the spread some value outside the wings. With fewer minutes left, possible settlement levels cluster more tightly around the current index level. That can help near the center and hurt farther away. The table holds the index level fixed and shows the modeled spread quote in points. [Options Industry Council on time decay](https://www.optionseducation.org/strategies/all-strategies/long-call-butterfly)

| Index level | 231 min | 100 min | 30 min | 5 min | Expiration |
|---|---:|---:|---:|---:|---:|
| 7,575, at the center | 7.47 | 10.69 | 15.98 | 21.27 | 25.00 |
| 7,535, below the lower wing | 3.63 | 2.44 | 0.50 | 0.00 | 0.00 |

## How a 3.30-point spread can trade at 8.00

The example has a buy fill at 11:09 a.m. for 3.30 points and a sell fill at 1:20 p.m. for 8.00. If those timestamps are in U.S. Central Time, with settlement at 3:00 p.m. Central (4:00 p.m. Eastern), they correspond to 231 and 100 minutes remaining. Keeping the model volatility fixed at 20%, I solve for an index level below the center that produces each fill price. The case supplies the fills. The index levels and clock convention below are modeling assumptions. [Cboe SPXW trading hours](https://www.cboe.com/tradable-products/sp-500/spx-options/spx-specifications/)

| Scenario | Minutes left | Modeled SPX level | Spread quote | Profit before fees |
|---|---:|---:|---:|---:|
| A, matched to the entry | 231 | 7,532.48 | 3.30 | \$0 |
| B, matched to the exit | 100 | 7,557.23 | 8.00 | \$470 |

In this scenario, the index moves about 25 points closer to the center. It remains about 18 points below 7,575 at B. An 8.00-point exit produces **\$470** before fees, since (8.00 − 3.30) × 100 = 470. Holding the index at A while reducing the remaining time to 100 minutes puts the modeled spread at 2.01 points.

In the market, implied volatility, the volatility backed out of an option's price, can change at each strike. A drop in volatility generally helps a long butterfly near its center, where more of the model's settlement outcomes fall near the peak. Far from the center, higher volatility can give the index more room to reach the wings. The executable spread price also depends on available bids and offers. The two fills alone leave the actual market path uncertain. [Options Industry Council on butterfly volatility](https://www.optionseducation.org/strategies/all-strategies/long-call-butterfly) · [Options Industry Council on option pricing](https://www.optionseducation.org/optionsoverview/options-pricing)

## What changes when a put butterfly joins it

Using all puts with the same 1:−2:1 weights, the same equidistant strikes and the same expiry gives the same expiration cash payout as using all calls. The profit can differ if the two spreads have different entry costs. At each strike the call payoff minus the put payoff equals the settlement level minus the strike. The three settlement terms cancel under the butterfly's weights, and the equally spaced strikes cancel as well. [Options Industry Council put butterfly](https://prd-web.optionseducation.org/strategies/all-strategies/long-put-butterfly)

The case also shows a 7,460 / 7,485 / 7,510 put butterfly bought for 2.00 points. Its center is 90 points below the call butterfly's center. If both positions are held through the same settlement, they give two separate profit ranges. Either spread has to earn the full 5.30-point combined debit for the pair to break even. A settlement from **7,510 through 7,550**, inclusive, pays zero on both spreads and loses the full **\$530** before fees.

![Two individual expiration payoff peaks and their combined curve. The gap from 7,510 to 7,550 is shaded and marked as a 530-dollar combined loss.](combined-expiry.svg "Figure 2. Each dashed line shows one spread's profit or loss at expiration. The solid curve adds the two positions, including both entry premiums. The four markers are the combined break-even levels.")

| Position | Expiration break-even levels | Maximum profit | Maximum loss |
|---|---:|---:|---:|
| 7,485 put butterfly | 7,462.00 and 7,508.00 | \$2,300 | \$200 |
| 7,575 call butterfly | 7,553.30 and 7,596.70 | \$2,170 | \$330 |
| Both, held to expiration | 7,465.30, 7,504.70, 7,555.30 and 7,594.70 | \$1,970 | \$530 |

Figure 2 shows what would have happened if both positions had been held. The call butterfly closed intraday. The available trades show the put opening fill, so its realized profit or loss remains unknown.
