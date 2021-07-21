using System;
using System.Collections;
using System.Net;
using UnityEngine;
using Random = UnityEngine.Random;
using indoorMobility.Scripts.Utils;
using Environment = indoorMobility.Scripts.Game.Environment;
//using ImgSynthesis = indoorMobility.Scripts.ImageSynthesis.ImgSynthesis;


namespace indoorMobility.Scripts.Game
{
    public class GameManager : MonoBehaviour
    {
        #region;
        private Camera _camera;
        private Command _command;
        private Server _server;

#pragma warning disable 0649 //disable warnings about serializefields not being assigned that occur in certain unity versions
        [SerializeField] public AppData appData;
        [SerializeField] private GameObject Environment; // Env. GameObject
        private Environment environment; // Env. script

        /*

        //[SerializeField] private GameObject player;
        [SerializeField] private GameObject player;
      //  [SerializeField] private GameObject capsule1;
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
        */

#pragma warning restore 0649 //reanable the unassigned variable warnings 
        public delegate void DataSentEventListener(byte[] data);
        public event DataSentEventListener DataSent;



        #endregion;



        #region;

        private void OnDataReceived(byte[] data)
        {
            environment.Input = data[1];
            _command = (Command)data[0];
        }

        private void OnDataSent()
        {
            DataSent?.Invoke(environment.Output);
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
                        environment.Reset();
                        Time.timeScale = timescale;
                        yield return new WaitForFixedUpdate();
                        Time.timeScale = 0;
                        OnDataSent();
                        _command = Command.None;
                        break;

                    case Command.Step:
                        environment.Step();
                        Debug.Log("environment step command was executed");
                        Time.timeScale = timescale;
                        yield return new WaitForFixedUpdate();
                        Time.timeScale = 0;
                        OnDataSent();
                        _command = Command.None;
                        break;

                    case Command.SetSeed:
                        environment.ChangeSeed();
                        Debug.Log("Random seed was changed");
                        Time.timeScale = timescale;
                        yield return new WaitForFixedUpdate();
                        Time.timeScale = 0;
                        OnDataSent();
                        _command = Command.None;
                        break;


                    default: throw new ArgumentOutOfRangeException();
                }
        }


        #endregion;


        #region;



        protected void Awake() //Gets run when the game starts once
        {   // Instantiate environment
            appData.RandomSeed = (int)System.DateTime.Now.Ticks;
            environment = Environment.GetComponent<Environment>(); //Instantiate(environment);

            // Initialize camera
            _camera = Camera.main;
            (_camera.targetTexture = new RenderTexture(appData.Width, appData.Height, 0)).Create();

            // Start server
            _server = new Server(IPAddress.Parse(appData.IpAddress), appData.Port);
            _server.DataRead += OnDataReceived;
            DataSent += _server.OnDataSent;
            _server.Start();
            StartCoroutine(Tick(appData.TimeScale));
        }

#endregion;
    }
}
 
 