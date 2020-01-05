#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "mlcore.h"

#define PI 3.14159265359f

#define TRAIN 1
#define NUMBER_OF_GAMES 1000
#define P -1    // penalty for the holes
#define R +1    // reward for the end position
#define T -0.1  // punisment for each move

// Q-Table Size 7x5 simple board game
// Action space: 4
// State-Space size: 7x5
struct game_t
{
    // x,y position of the target
    int targetPosition[2];

    // state indices of the reward squares
    int isEndS[35];
    float reward[35];
};

// reset the game
void game_reset(struct game_t *game)
{
    // state indices of the end squares
    // 9-11-16-17-18-23-25
    int   isEndS[35] = {0,0,0,0,0,0,0,0,0,1, 0,1,0,0,0,0,0,1,0,0, 0,0,0,1,0,1,0,0,0,0, 0,0,0,0,0};
    float reward[35] = {T,T,T,T,T,T,T,T,T,P, T,P,T,T,T,T,T,R,T,T, T,T,T,P,T,P,T,T,T,T, T,T,T,T,T};

    // set the structure members
    int i = 0;
    for(i = 0; i < 35; i++)
    {
        game->isEndS[i] = isEndS[i];
        game->reward[i] = reward[i];
    }

    // pick random position for the target which not immediately ends the game
    int initialState = random_int(0, 34);
    while (game->isEndS[initialState])
    {
        initialState = random_int(0, 34);
    }

    // x,y position of the target
    game->targetPosition[1] = initialState / 7;
    game->targetPosition[0] = initialState - 7 * game->targetPosition[1];
}

// return the state of the game
uint32_t get_state(struct game_t *game)
{
    return (uint32_t)(game->targetPosition[0] + 7 * game->targetPosition[1]);
}

// update the position of the target
void update_target_position(struct game_t *game, uint32_t action)
{
    // actions: LEFT-UP-RIGHT-DOWN

    // convert the action to position
    if(action == 0)
    {
        game->targetPosition[0] = max(0, game->targetPosition[0] - 1);
    }
    // go UP
    else if(action == 1)
    {
        game->targetPosition[1] = max(0, game->targetPosition[1] - 1);
    }
    // go RIGHT
    else if (action == 2)
    {
        game->targetPosition[0] = min(6, game->targetPosition[0] + 1);
    }
    // go DOWN
    else if (action == 3)
    {
        game->targetPosition[1] = min(4, game->targetPosition[1] + 1);
    }
}

// play single game and return the game result
int play_game(struct q_table_t *Q, struct game_t *game, float exploration)
{
    uint32_t current_state = get_state(game);

    // play a single game untill the end
    while (!game->isEndS[current_state])
    {
        uint32_t action = q_table_get_action(Q, current_state, exploration);

        // update the target position
        update_target_position(game, action);

        // get the next state after the selected action
        uint32_t next_state = get_state(game);

        // do learning
        q_table_update(Q, current_state, action, next_state, 0.01, game->reward[next_state]);

        // goto next state
        current_state = next_state;
    }

    return game->reward[current_state] == R ? 1:0;
}

float get_epsilon(int t)
{
    return max(0.01f, min(1, 1.0f - log10((t + 1) / 2)));
}

// main function
int main(int argc, char *argv[])
{
    //uint32_t state[4] = {200, 100, 50, 1};
    //random_setseed(state);
    random_seed();
    
    // create a game with the following parameters
    struct game_t game;
    game_reset(&game);

#ifdef TRAIN

    // create learning table
    struct q_table_t *Qtable = q_table_create(35, 4, 0.98);

    int generations = 0;
    while (generations < 100)
    {
        float exploration = get_epsilon(generations++);//explorationCoeffcients[generations++];

        int total_winners = 0;

        // check all the members and find winners
        int i = 0;
        for (i = 0; i < NUMBER_OF_GAMES; i++)
        {
            game_reset(&game);
            int game_result = play_game(Qtable, &game, exploration);

            // increase the winner count
            if (game_result > 0)
            {
                total_winners++;
            }
        }

        printf("Training [GEN=%04d, E=%04.3f]: %05d Total Winners\n", generations, exploration, total_winners);
    }

    // save the table
    q_table_write(Qtable, "gamemodel.bin");

    // free the table
    q_table_free(&Qtable);
#endif

    ////////////////////////////PLAY A GAME AND VISUALIZE/////////////////////////////////////
    // load the model
    struct q_table_t *model = q_table_read("gamemodel.bin");

    // print the table
    int i,j, a;
    char astr[4] = {'L','U','R','D'};
    printf("\n");
    for(j = 0; j < 5; j++)
    {
        for(i = 0; i < 7; i++)
        {
            uint32_t max_a = q_table_get_action(model, i + 7 * j, 0);

            printf("%c ",astr[max_a]);
        }
        printf("\n");
    }


    // set the num actions
    int num_actions = 0;

    // intialize a new game and save all actions
    game_reset(&game);
    uint32_t current_state = get_state(&game);
    
    // play a single game untill the end
    while (!game.isEndS[current_state])
    {
        uint32_t action = q_table_get_action(model, current_state, 0);

        printf("State (%d) Action(%d)\n", current_state, action);
        num_actions++;

        // update the target position
        update_target_position(&game, action);

        // get the next state after the selected action
        uint32_t next_state = get_state(&game);

        // goto next state
        current_state = next_state;
    }

    printf("\ntarget %s the game in %d movement!\n", game.reward[current_state] == R ? "won" : "lost", num_actions);
    ///////////////////////////////////////////////////////////////////////////////

    // free the table
    q_table_free(&model);

    // end of the app
    return 0;
}