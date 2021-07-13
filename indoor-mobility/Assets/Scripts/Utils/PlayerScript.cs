using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerScript : MonoBehaviour
{
    private bool playerHitBox;
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
}
