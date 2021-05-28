# Projects!
Some interesting projects that I have done:

Feed forward neural network and LSTM built off of it using the eigen library. Written in C++. Files: LSTM.cpp, LSTM.h, NN.cpp, NN.h  
Contains functionality to save trained neural network weights in text files and retrieve them for use after closing program.

Completed Game using Unreal Engine 4 Showcase video: https://www.youtube.com/watch?v=_0EHRcCsuEo&ab_channel=BBBAAA

Reinforcement learning project using LSTM:  
Simple 2d game testing LSTM functionality. Files: Env.cpp, Env.h, Animal.cpp, Animal.h, main.cpp  
I used model free reinforcement learning using an LSTM as the q function. The learning algorithm was on-policy with a no offline data or batches, learning from the most recent epoch training data in order with truncated backpropagation then disposing of it. The environment was completely observable so POMDP was not necessary. The state space is not continuous and the action space is discrete. The policy does not use planning and instead simply runs through the LSTM choosing the highest estimated reward value of the actions and has an epsilon chance to choose a random action. The epsilon decays as epochs are consistently more successful. Reward function has no discount factor or immediate rewards, reward is decided and evenly distributed between timesteps at the end. Backpropagation through time on LSTM using constant error carousel truncated at two timesteps past. 
After around 2000 epochs of training significant performance improvements but is still unable to succeed in certain cases. I expect the reason to be not enough samples to reach generalization, the sparse nature of inputs, poor reward function, or epsilon function difficulties. Random positioning of enemies each epoch could also contribute to this difficulty. I will also be looking into changing gradient clipping values, learning rate, and exploring planning.

Automatic Pokemon Shiny hunting Machines using Arduino video: https://youtu.be/quXotwqyeH4       
Files: ArduinoShinyHunter, ArduinoLegendaryPokemonShinyHunter

My YouTube channel: https://www.youtube.com/channel/UC5FYeXus9XNnPKATCq_G4Ng
