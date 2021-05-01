# Projects!
Some interesting projects that I have done:

Feed forward neural network and LSTM built off of it using the eigen library. Written in C++. Files: LSTM.cpp, LSTM.h, NN.cpp, NN.h

Contains functionality to save trained neural network weights in text files and retrieve them for use after closing program.

Completed Game using Unreal Engine 4 Showcase video: https://www.youtube.com/watch?v=_0EHRcCsuEo&ab_channel=BBBAAA

Automatic Pokemon Shiny hunting Machines using Arduino video: https://youtu.be/quXotwqyeH4       
Files: ArduinoShinyHunter, ArduinoLegendaryPokemonShinyHunter

My YouTube channel: https://www.youtube.com/channel/UC5FYeXus9XNnPKATCq_G4Ng

Reinforcement learning project using LSTM:  
Simple 2d game testing LSTM functionality. Files: Env.cpp, Env.h, Animal.cpp, Animal.h, main.cpp
Reinforcement learning using epsilon greedy. Epsilon decays as epochs are consistently more successful. Reward function has no discount factor or immediate rewards, reward is decided and evenly distributed between timesteps at the end. Backpropagation at each timestep occurs after each epoch with no random sampling. Backpropagation through time on LSTM using constant error carousel truncated at two timesteps past. 
After 12 hours of training significant performance improvements but is still unable to succeed in certain cases. Expect the reason to be the sparse nature of inputs, poor reward function, or epsilon function difficulties. Random positioning of enemies each epoch could also contribute to this. Will be looking into changing gradient clipping values, learning rate, and exploring different reinforcement learning techniques such as POMDP and MCTS.
