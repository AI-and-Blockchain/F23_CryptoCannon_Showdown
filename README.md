# Github Structure

"AI-Training" branch for the AI model training.

"blockchain" branch for smart contract implementation.


# CryptoCannon Showdown
[Click here to play](https://xbubonic.github.io/)
Play AI in a game of [Battleship](https://www.officialgamerules.org/battleship) (also known as Sea Battle) for the chance to win some cryptocurrency.

Users bet an amount of cryptocurrency before starting a game, and then play a game against the AI. If they win, they get their crypto back plus a bit extra. They can also purchase cosmetic NFTs to customize how the game looks, like their ships.

# Sequence Diagram

```mermaid
sequenceDiagram
    Participant AI
    Participant Game
    Participant Smart Contract
    Participant User's Wallet
    AI->>Game: Give policy
    Game->>Smart Contract: Start game, wager crypto
    Smart Contract->>User's Wallet: Get crypto from user's account
    User's Wallet->>Smart Contract: return crypto
    loop until game ends
        Game->>Smart Contract: Send move
        Game->>Smart Contract: Send and verify AI move
    end
    Game->>Smart Contract: Game finished
    Smart Contract->>User's Wallet: Pay out(if user wins)
    Game->>AI: Use game data to improve AI
    Game->>Smart Contract: Player buys cosmetic
    Smart Contract->>User's Wallet: Mint + Award NFT to user
```

# Component Disagram

```mermaid
stateDiagram-v2
    SmartContract --> Game_Webapp
    AI_Model --> SmartContract
    SmartContract --> Blockchain
    Blockchain --> SmartContract
    Game_Webapp --> Player

```
