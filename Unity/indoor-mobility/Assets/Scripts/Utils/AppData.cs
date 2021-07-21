using UnityEngine;

namespace indoorMobility.Scripts.Utils
{
    [CreateAssetMenu(menuName = "IndoorMobility/AppData")]
    public class AppData : ScriptableObject
    {

        #region
        // Game data
        [SerializeField] private float _timescale;
        [SerializeField] private int _width;
        [SerializeField] private int _height;
        public float TimeScale
        {
            get => _timescale;
            set => _timescale = value;
        }

        public int Width
        {
            get => _width;
            set => _width = value;
        }
        public int Height
        {
            get => _height;
            set => _height = value;
        }


        #endregion


        #region
        // Server data 
        [SerializeField] private string _ipAddress;
        [SerializeField] private int _port;
        public string IpAddress
        {
            get => _ipAddress;
            set => _ipAddress = value;
        }
        public int Port
        {
            get => _port;
            set => _port = value;
        }
        #endregion


        #region
        // Rewards
        [SerializeField] private byte _forwardStepReward;
        [SerializeField] private byte _sideStepReward;
        [SerializeField] private byte _boxBumpReward;
        [SerializeField] private byte _wallBumpReward;

        public byte ForwardStepReward
        {
            get => _forwardStepReward;
            set => _forwardStepReward = value;
        }
        public byte SideStepReward
        {
            get => _sideStepReward;
            set => _sideStepReward = value;
        }
        public byte BoxBumpReward
        {
            get => _boxBumpReward;
            set => _boxBumpReward = value;
        }
        public byte WallBumpReward
        {
            get => _wallBumpReward;
            set => _wallBumpReward = value;
        }
        #endregion

        #region



        // Environment data
        [SerializeField] private int _randomSeed;
        [SerializeField] private float _forwardSpeed;
        [SerializeField] private float _sideStepDistance;
        [SerializeField] private int _maxSteps;
        [SerializeField] private float _camRotJitter;
        [SerializeField] private int _visibleHallwayPieces;

        public int RandomSeed
        {
            get => _randomSeed;
            set => _randomSeed = value;
        }
        public float ForwardSpeed
        {
            get => _forwardSpeed;
            set => _forwardSpeed = value;
        }
        public float SideStepDistance
        {
            get => _sideStepDistance;
            set => _sideStepDistance = value;
        }
        public int MaxSteps
        {
            get => _maxSteps;
            set => _maxSteps = value;
        }
        public float CamRotJitter
        {
            get => _camRotJitter;
            set => _camRotJitter = value;
        }
        public int VisibleHallwayPieces
        {
            get => _visibleHallwayPieces;
            set => _visibleHallwayPieces = value;
        }
        #endregion



        private void Reset()
        {
            // Game Manager
            _timescale = 1;
            _width = 128;
            _height = 128;

            // Server
            _ipAddress = "127.0.0.1";
            _port =  13000;

            // Rewards
            _forwardStepReward = (byte) 10;
            _sideStepReward = (byte) 101;
            _boxBumpReward = (byte) 120;
            _wallBumpReward = (byte) 110;

            // Random seed (for different hallway variations, random camera rotations)
            _randomSeed = 0;

            // Hallway data
            _visibleHallwayPieces = 20;

            // Player data
            _forwardSpeed = 0.5f;
            _sideStepDistance = 0.95f;
            _maxSteps = 100;
            _camRotJitter = 3.0f;
        }
    }
}