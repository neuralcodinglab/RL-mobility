using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using ImgSynthesis = indoorMobility.Scripts.ImageSynthesis.ImgSynthesis;
using indoorMobility.Scripts.Utils;

namespace indoorMobility.Scripts.Game
{
    public class Environment : MonoBehaviour {
        #region;
        private AppData appData;
        private int _action;
        private byte[] _data;
        private byte _end, _reward;
        private int _height, _width;
        private List<Color32[]> _state;
        private RenderTexture _targetTexture;
 

        // Children environment GameObjects and corresponding scripts
        [SerializeField] private GameObject Hallway;
        [SerializeField] private GameObject Player;
        private Player player;
        private Hallway hallway;

        // Image processing script (attached to camera)
        private ImgSynthesis imgSynthesis;

        // TODO MOVE THIS TO HALLWAY
        private bool complexHall;
        private bool testing;

        // Can be accessed by player script or game manager
        public byte Reward { set => _reward = value;}
        public byte End { set => _end = value;}

        #endregion

        #region;

        public byte Input { set => _action = value; }

        public byte[] Output { //output to be send to python, consists of 1 byte to determine if the loop ended, 1 byte for the reward and x bytes with the camera view of the agent
            get {//the amount of bytes needed for the camera view is dependent on the size selected
                _data[0] = _end;
                _data[1] = _reward;

                // Render the state
                // (for the different render types: colors, semantic segmentation, depth, etc.)
                var tex = new Texture2D(_width, _height);
                _state= new List<Color32[]>();
                for(var idx = 0; idx<=5; idx++)
                {
                    // Get hidden camera 
                    var cam = ImgSynthesis.capturePasses[idx].camera;

                    // Render
                    RenderTexture.active = _targetTexture; //renderRT;
                    cam.targetTexture = _targetTexture; // renderRT;
                    cam.Render();
                    tex.ReadPixels(new Rect(0, 0, _targetTexture.width, _targetTexture.height), 0, 0);
                    tex.Apply();
                    _state.Add(tex.GetPixels32());
                }
                Object.Destroy(tex);

                // Color32 arrays for each of the render types:
                var colors  = _state.ElementAt(0);
                var objseg  = _state.ElementAt(1);
                var semseg  = _state.ElementAt(2);
                var depth   = _state.ElementAt(3);
                var normals = _state.ElementAt(4);
                var flow    = _state.ElementAt(5);
                
                // Write state to _data
                for (var y = 0;
                    y < _height;
                    y++)
                    for (var x = 0;
                        x < _width;
                        x++) {
                        var i = 16 * (x - y * _width + (_height - 1) * _width);
                        var j = 1 * (x + y * _width);
                        _data[i + 2]  = colors[j].r;
                        _data[i + 3]  = colors[j].g;
                        _data[i + 4]  = colors[j].b;
                        _data[i + 5]  = objseg[j].r;
                        _data[i + 6]  = objseg[j].g;
                        _data[i + 7]  = objseg[j].b;
                        _data[i + 8]  = semseg[j].r;
                        _data[i + 9]  = semseg[j].g;
                        _data[i + 10] = semseg[j].b;
                        _data[i + 11] = normals[j].r;
                        _data[i + 12] = normals[j].g;
                        _data[i + 13] = normals[j].b;
                        _data[i + 14] = flow[j].r;
                        _data[i + 15] = flow[j].g;
                        _data[i + 16] = flow[j].b;
                        _data[i + 17] = depth[j].r;

                    }

                return _data;
            }
        }

        #endregion;

        #region;

        public void Reset() 
        { //reset all values, delete all hallwaypieces and rebuild a new starting hallway
           Random.InitState(appData.RandomSeed);
           hallway.Reset(_action);
           player.Reset(_action);
           imgSynthesis.OnSceneChange();
           appData.RandomSeed = (int)System.DateTime.Now.Ticks;
        }



        public void Step()
        {   //Move the player (environment.Reward and environment.End are updated by player)
            player.Move(_action);

            // update hallway if necessary
            if (hallway.EndPosition - player.transform.position.z <= 36) //TODO: hardcoded  hallway length
                hallway.updateHallway();
        }

        public void ChangeSeed()
        {
            appData.RandomSeed = _action;
        }


        private void Start() {
            appData = GameObject.Find("GameManager").GetComponent<GameManager>().appData;
            player = Player.GetComponent<Player>();
            hallway = Hallway.GetComponent<Hallway>();

            if (Camera.main != null) {
                _targetTexture = Camera.main.targetTexture;
                _height = _targetTexture.height;
                _width = _targetTexture.width;
                imgSynthesis = Camera.main.GetComponent<ImgSynthesis>();
            }
            _data = new byte[2 + 16 * _width * _height];      

            
        }



        public void FixedUpdate()
        {
            Debug.Log(player.transform.position.z);
        }


        #endregion;

   
        
    }
}