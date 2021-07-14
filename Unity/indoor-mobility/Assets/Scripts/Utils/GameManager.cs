using System;
using System.Collections;
using System.Net;
using UnityEngine;
using Random = UnityEngine.Random;
using Environment = indoorMobility.Scripts.Hallway.Environment;
using Player = indoorMobility.Scripts.Hallway.Player;
//using ImgSynthesis = indoorMobility.Scripts.ImageSynthesis.ImgSynthesis;


namespace indoorMobility.Scripts.Utils
{
    public class GameManager : MonoBehaviour
    {
        #region;

        private Command _command;
        private Environment _environment;
        private Server _server;
        private Player _player;

#pragma warning disable 0649 //disable warnings about serializefields not being assigned that occur in certain unity versions
        [SerializeField] private AppData appData;
        [SerializeField] private GameObject environment;
        //[SerializeField] private GameObject player;
        [SerializeField] private GameObject player;
        [SerializeField] private GameObject capsule1;
        [Header("Connection settings, need to be consistent with the settings in python:")]
        [SerializeField] public string ipAddress;
        [SerializeField] private int _port;
        [SerializeField] private int _size;
        [SerializeField] private float _timescale;
        [Header("Agent movement settings, if changed need to also change things in the c# code:")]
        [SerializeField] private float forwardSpeed;
        [SerializeField] private float sideStepDistance;
        [Header("Complex texture hallway or not:")]
        [SerializeField] private bool complexHallway;
        [Header("Max amount of steps reachable in training and validation, and amount of camera jitter:")]
        [SerializeField] private int maxSteps;
        [SerializeField] private float jitterAmount;
        [Header("A random number seed to keep validation and testing consistent:")]
        [SerializeField] private int randomValSeed;
        [SerializeField] private int randomTestSeed;
        [Header("Reward settings, these are bytes so can only use numbers between 0 and 256:")]
        [SerializeField] private byte forwardStepReward;
        [SerializeField] private byte leftRightStepReward;
        [SerializeField] private byte boxBumpReward;
        [SerializeField] private byte wallBumpReward;
#pragma warning restore 0649 //reanable the unassigned variable warnings 
        public delegate void DataSentEventListener(byte[] data);

        public event DataSentEventListener DataSent;

        private IPAddress _ip;

        private int stepsTaken;
        private bool playerMovedForward;
        private bool validation_run;
        private bool test_run;
        private int randomTrainingSeed;
        private int maxStepsTest;







        #endregion;

        #region;

        public Camera Camera { get; private set; }

        #endregion;

        #region;

        private void OnDataReceived(byte[] data)
        {
            //MovePlayer(data[1]);
            _environment.Input = data[1];
            _player.Input = data[1];
            _command = (Command)data[0];
        }

        private void OnDataSent()
        {
            DataSent?.Invoke(_environment.Output);
        }



        private IEnumerator Tick(float timescale)
        {
            _command = Command.None;
            Time.timeScale = 0;
            while (true)
                switch (_command)
                {   case Command.None:
                        yield return null;
                        continue;

                    case Command.Reset:
                        _environment.Reset();
                        _player.Reset();
                        Time.timeScale = timescale;
                        yield return new WaitForFixedUpdate();
                        Time.timeScale = 0;
                        OnDataSent();
                        _command = Command.None;
                        break;

                    case Command.Step:
                        _player.Move();
                        Time.timeScale = timescale;
                        yield return new WaitForFixedUpdate();
                        Time.timeScale = 0;
                        OnDataSent();
                        _command = Command.None;
                        break;

                    default: throw new ArgumentOutOfRangeException();
                }
        }

        /*   

         // OLD CODE (by Sam)

        private IEnumerator Tick(float timescale)
        {
            _command = Command.None;
            Time.timeScale = 0;
            while (true)
                switch (_command)
                {
                    case Command.None:
                        yield return null;
                        continue;
                    case Command.Reset: //normal reset command for a training loop
                        if (validation_run || test_run) { //check if it is the first training loop
                            Camera.transform.rotation = Quaternion.Euler(0, 0, 0);
                            Random.InitState(randomTrainingSeed);
                            Debug.Log(Random.value);
                            validation_run = false;
                            test_run = false;
                        }
                        _environment.setTrainingOrTesting(false);
                        _environment.Reset();
                        CameraJitter(); //add a small amount of camera rotation for this loop
                        yield return null; 
                        player.transform.position= new Vector3(0f,1.1f,4f);
                        capsule1.GetComponent<PlayerScript>().resetPlayerHitBox(); //reset agents hitbox to False
                        stepsTaken = 1;
                        goto case Command.Step;
                    case Command.Reset_val: //reset command for a validation loop
                        if (!validation_run) { //check if it is the first validation loop
                            Camera.transform.rotation = Quaternion.Euler(0, 0, 0);
                            randomTrainingSeed = Random.Range(0, 9999999);
                            Debug.Log(randomTrainingSeed);
                            Random.InitState(randomValSeed);
                            validation_run = true;
                            test_run = false;
                        }
                        _environment.Reset();
                        _environment.setTrainingOrTesting(false);
                        CameraJitter(); //add a small amount of camera rotation for this loop

                        yield return null;
                        player.transform.position = new Vector3(0f, 1.1f, 4f);
                        capsule1.GetComponent<PlayerScript>().resetPlayerHitBox(); //reset agents hitbox to False
                        stepsTaken = 1;
                        goto case Command.Step_val;
                    case Command.Reset_test: //reset command for a test loop
                        if (!test_run) { //check if it is the first test loop
                            Camera.transform.rotation = Quaternion.Euler(0, 0, 0);
                            test_run = true;
                            validation_run = false;
                            randomTrainingSeed = Random.Range(0, 9999999);
                            Debug.Log(randomTrainingSeed);
                            Random.InitState(randomTestSeed);
                        }
                        Camera.transform.rotation = Quaternion.Euler(0, 0, 0);

                        _environment.setTrainingOrTesting(true);
                        _environment.Reset();
                        CameraJitter(); //add a small amount of camera rotation for this loop
                        yield return null;
                        player.transform.position = new Vector3(0f, 1.1f, 4f);
                        capsule1.GetComponent<PlayerScript>().resetPlayerHitBox(); //reset agents hitbox to False
                        stepsTaken = 1;
                        goto case Command.Step_test;
                    case Command.Step: //step for a training loop
                        Time.timeScale = timescale;
                        yield return new WaitForFixedUpdate();
                        Time.timeScale = 0;
                        if (capsule1.GetComponent<PlayerScript>().getplayerHitBox()) { //check if the player has hit a box
                            _environment.setReward(boxBumpReward);
                            capsule1.GetComponent<PlayerScript>().resetPlayerHitBox(); //reset hitbox because the agent needs to continue in the validation hallway 
                           // _environment.setEnd(1); //send command to end the training loop because it hit a box                       
                        }
                        //else {
                            //if (stepsTaken >= maxSteps) { //check if the agent has reached the max steps
                           //     Debug.Log("Max steps reached");
                           //     _environment.setReward(forwardStepReward);
                            //    _environment.setEnd(2);
                           // }
                        if (stepsTaken % 4 == 0 && stepsTaken != 0 && playerMovedForward) { //Every 4 steps forward the hallway gets updated
                            _environment.updateHallway();
                            playerMovedForward = false;
                        }
                        //}
                        OnDataSent();
                        _command = Command.None;
                        break;
                    case Command.Step_val: //step for a validation loop
                        Time.timeScale = timescale;
                        yield return new WaitForFixedUpdate();
                        Time.timeScale = 0;
                        if (capsule1.GetComponent<PlayerScript>().getplayerHitBox()) { //check if the agent has hit a box
                            _environment.setReward(boxBumpReward);
                            capsule1.GetComponent<PlayerScript>().resetPlayerHitBox(); //reset hitbox because the agent needs to continue in the validation hallway 
                        } else {
                            if (stepsTaken >= maxSteps) { //check if the agent has reached the end of the hallway
                                Debug.Log("Reached end validation hallway");
                                _environment.setEnd(2);
                            }        
                        }
                        if (stepsTaken % 4 == 0 && stepsTaken != 0 && playerMovedForward) { //Every 4 steps forward the hallway gets updated
                            _environment.updateHallway();
                            playerMovedForward = false;
                        }
                        OnDataSent();
                        _command = Command.None;
                        break;
                    case Command.Step_test: //step for a test loop
                        Time.timeScale = timescale;
                        yield return new WaitForFixedUpdate();
                        Time.timeScale = 0;
                        if (capsule1.GetComponent<PlayerScript>().getplayerHitBox()) { //check if the agent has hit a box
                            _environment.setReward(boxBumpReward);
                            capsule1.GetComponent<PlayerScript>().resetPlayerHitBox(); //reset hitbox because the agent needs to continue in the test hallway 
                        } else {
                            if (stepsTaken >= maxStepsTest) { //check if the agent has reached the end of the hallway
                                Debug.Log("Reached end test hallway");
                                _environment.setEnd(2);
                            }
                        }
                        if (stepsTaken % 4 == 0 && stepsTaken != 0 && playerMovedForward) { //Every 4 steps forward the hallway gets updated
                            _environment.updateTestHallway();
                            playerMovedForward = false;
                        }
                        OnDataSent();
                        _command = Command.None;
                        break;
                    default: throw new ArgumentOutOfRangeException();
                }
        }

            //OLD CODE (BY SAM)
        */

      



 

        #endregion;

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

        #region;


        private void CameraJitter() { //Small camera rotations to help against overfitting
Camera.transform.rotation = Quaternion.Euler(Random.Range(-jitterAmount, jitterAmount), Random.Range(-jitterAmount, jitterAmount), Random.Range(-jitterAmount, jitterAmount));
}

protected void Awake() //Gets run when the game starts once
{

// Load AppData
ipAddress = appData.IpAddress;
_port = appData.Port;
forwardStepReward = appData.Forward;
leftRightStepReward = appData.Side;
boxBumpReward = appData.Box;
wallBumpReward = appData.Wall;
complexHallway = appData.Complex;



Camera = GetComponentInChildren<Camera>();
_ip = IPAddress.Parse(ipAddress);
stepsTaken = 0;
maxStepsTest = 560; // 14 * 10 * 4 different test hallways * pieces per test hallway * steps needed to pass one piece
playerMovedForward = false;
validation_run = false;
test_run = false;
_environment = environment.GetComponent<Environment>(); //Instantiate(environment);
_environment.setHallwayType(complexHallway);
_server = new Server(_ip, _port);
_player = player.GetComponent<Player>();
_player.SetForwardSpeed = forwardSpeed;
_server.DataRead += OnDataReceived;
DataSent += _server.OnDataSent;
_server.Start();
(Camera.targetTexture = new RenderTexture(_size, _size, 0)).Create();
StartCoroutine(Tick(_timescale));
}

#endregion;
}
}
 
 