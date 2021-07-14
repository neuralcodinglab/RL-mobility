using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace indoorMobility.Scripts.Hallway
{
    public class Player : MonoBehaviour
    {
        private bool playerHitBox;
        private int _action;
        private float _forwardSpeed;

        public float SetForwardSpeed
        {
            set => _forwardSpeed = value;
        }

        public byte Input
        {
            set => _action = value;
        }

        private void OnCollisionEnter(Collision collision) { //Test to see if the agents capsule is colliding with another object, atm this can only be a box since the movement is restricted
            playerHitBox = true;
            Debug.Log("player hit");
        }

        private void Start() {
            playerHitBox = false;
        }

        public bool getplayerHitBox() {
            return playerHitBox;
        }

        public void resetPlayerHitBox() {
            playerHitBox = false;
        }


        public void Move()
        {
            Vector3 currentPos = transform.position;
            transform.position = currentPos + new Vector3(0f, 0f, _forwardSpeed);
            //stepsTaken++;
            //playerMovedForward = true;
        }

        public void Reset()
        {
            transform.position = new Vector3(0f, 1.1f, 4f);
            resetPlayerHitBox(); //reset agents hitbox to False 
        }

        /*

       private void MovePlayer(int move)
       {
           if (move == 0)
           { //forward
               _environment.setReward(forwardStepReward);//reward for stepping forward
               Vector3 currentPos = player.transform.position;
               player.transform.position = currentPos + new Vector3(0f, 0f, forwardSpeed);
               stepsTaken++;
               playerMovedForward = true;
           }
           else if (move == 1)
           { //agent wants to move left
               Vector3 currentPos = player.transform.position;
               if (currentPos.x == -sideStepDistance)
               { //if the agent is already at the left wall
                   _environment.setReward(wallBumpReward); //negative reward for bumping into the wall
                   player.transform.position = currentPos + new Vector3(0f, 0f, 0f);
               }
               else
               {
                   _environment.setReward(leftRightStepReward); //reward for moving to the left
                   player.transform.position = currentPos + new Vector3(-sideStepDistance, 0f, 0f);
               }
           }
           else if (move == 2)
           { //agent wants to move right
               Vector3 currentPos = player.transform.position;
               if (currentPos.x == sideStepDistance)
               { //if the agent is already at the right wall
                   _environment.setReward(wallBumpReward); //negative reward for bumping into the wall
                   player.transform.position = currentPos + new Vector3(0f, 0f, 0f);
               }
               else
               {
                   _environment.setReward(leftRightStepReward); //reward for moving to the right
                   player.transform.position = currentPos + new Vector3(sideStepDistance, 0f, 0f);
               }
           }
       }

   */


    }
}