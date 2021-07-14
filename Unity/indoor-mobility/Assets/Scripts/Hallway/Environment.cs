using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using ImgSynthesis = indoorMobility.Scripts.ImageSynthesis.ImgSynthesis;

namespace indoorMobility.Scripts.Hallway
{
    public class Environment : MonoBehaviour {
        #region;

        private byte[] _data;
        private byte _end, _reward;
        private int _height, _width;
        private List<Color32[]> _state;
        private RenderTexture _targetTexture;
        private ImgSynthesis _imgSynthesis;
        private int _action;


        #pragma warning disable 0649 //disable warnings about serializefields not being assigned that occur in certain unity versions
        [SerializeField] private GameObject[] smallBoxPieces;
        [SerializeField] private GameObject[] bigBoxPieces;
        [SerializeField] private GameObject[] emptyHallWays;
        [SerializeField] private GameObject[] smallBoxPiecesComplex;
        [SerializeField] private GameObject[] bigBoxPiecesComplex;
        [SerializeField] private GameObject[] emptyHallWaysComplex;
        [SerializeField] private GameObject pointLight;
        [SerializeField] private int nrOfCurrentPieces;
        [SerializeField] private float lightIntensityLowerBound;
        [SerializeField] private float lightIntensityUpperBound;
        #pragma warning restore 0649 //reanable the unassigned variable warnings 
        private GameObject[] currentPieces;
        private GameObject[] currentLights;
        private GameObject[] currentPiecesEmpty;
        private int nrOfEmptyPieces;
        private int nrOfLights;
        private float currentZLights;
        private int currentLightsIndex;
        private int currentZEmpty;
        private int currentIndexEmpty;
        private int currentZ;
        private int currentIndex;
        private bool hallWayMade;
        private bool updateLightsNow;
        private bool complexHall;
        private bool testing;
        
        //index 
        private int[] testHall0 = { 1, 1, 5, 0, 2, 1, 4, 0, 0, 3 };
        private int[] testHall1 = { 0, 1, 5, 0, 0, 2, 3, 2, 0, 5 };
        private int[] testHall2 = { 0, 1, 5, 1, 0, 1, 4, 0, 2, 5 };
        private int[] testHall3 = { 0, 0, 5, 0, 2, 1, 3, 2, 2, 5 };
        private int[] testHall4 = { 1, 1, 3, 2, 0, 1, 4, 2, 1, 3 };
        private int[] testHall5 = { 0, 1, 5, 2, 0, 1, 4, 0, 1, 3 };
        private int[] testHall6 = { 0, 1, 5, 0, 2, 1, 3, 2, 2, 3 };
        private int[] testHall7 = { 1, 1, 3, 2, 0, 1, 4, 2, 2, 5 };
        private int[] testHall8 = { 2, 1, 3, 2, 2, 0, 5, 0, 2, 3 };
        private int[] testHall9 = { 2, 1, 3, 1, 2, 1, 4, 2, 0, 3 };
        private int[] testHall10 = { 2, 2, 3, 2, 0, 1, 5, 0, 0, 3 };
        private int[] testHall11 = { 1, 1, 5, 0, 2, 1, 4, 0, 1, 5 };
        private int[] testHall12 = { 2, 1, 3, 0, 2, 1, 4, 2, 1, 5 };
        private int[] testHall13 = { 2, 1, 3, 2, 0, 1, 5, 0, 0, 5 };

        private int[] testHall0c = { 1, 1, 11, 0, 2, 1, 7, 0, 0, 3 };
        private int[] testHall1c = { 0, 1, 12, 0, 0, 2, 4, 2, 0, 13 };
        private int[] testHall2c = { 0, 1, 13, 1, 0, 1, 8, 0, 2, 14 };
        private int[] testHall3c = { 0, 0, 14, 0, 2, 1, 5, 2, 2, 11 };
        private int[] testHall4c = { 1, 1, 3, 2, 0, 1, 9, 2, 1, 4 };
        private int[] testHall5c = { 0, 1, 11, 2, 0, 1, 10, 0, 1, 5 };
        private int[] testHall6c = { 0, 1, 12, 0, 2, 1, 6, 2, 2, 6 };
        private int[] testHall7c = { 1, 1, 3, 2, 0, 1, 7, 2, 2, 11 };
        private int[] testHall8c = { 2, 1, 4, 2, 2, 0, 12, 0, 2, 5 };
        private int[] testHall9c = { 2, 1, 5, 1, 2, 1, 8, 2, 0, 6 };
        private int[] testHall10c = { 2, 2, 6, 2, 0, 1, 13, 0, 0, 3 };
        private int[] testHall11c = { 1, 1, 11, 0, 2, 1, 9, 0, 1, 12 };
        private int[] testHall12c = { 2, 1, 3, 0, 2, 1, 10, 2, 1, 13 };
        private int[] testHall13c = { 2, 1, 4, 2, 0, 1, 14, 0, 0, 14 };
        
        private int[] testHallExperiment;
        private int indexTestHall;

        
        #endregion;

        #region;

        public byte Input {
            set => _action = value;
        }

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

        public void Reset() { //reset all values, delete all hallwaypieces and rebuild a new starting hallway
            updateLightsNow = false;
            _reward = 0;
            _end = 0;
            currentZ = 0;
            currentZEmpty = 0;
            currentIndex = 0;
            currentIndexEmpty = 0;
            currentZLights = -12f;
            currentLightsIndex = 0;



            if (hallWayMade)
                destroyHallway();
            if (testing) {
                makeTestHallwayArray();
                indexTestHall = 0;
                makeStartTestHallway();
            } else {
                makeStartHallway();
            }
           _imgSynthesis.OnSceneChange();

        }

        private void Start() {
            testing = false;
            currentZ = 0;
            currentZEmpty = 0;
            hallWayMade = false;
            currentIndex = 0;
            currentIndexEmpty = 0;
            nrOfEmptyPieces = 2;
            currentZLights = -12f; //offset starting light behind the agent so the beginning is not to dark in comparison to the rest
            currentLightsIndex = 0;
            nrOfLights = (int)(nrOfCurrentPieces/2) +4; // On average 1 light every 2 pieces with 4 lights as buffer behind him
            currentPieces = new GameObject[nrOfCurrentPieces];
            currentPiecesEmpty = new GameObject[nrOfEmptyPieces];
            currentLights = new GameObject[nrOfLights];
            indexTestHall = 0;
            if (Camera.main != null) {
                _targetTexture = Camera.main.targetTexture;
                _height = _targetTexture.height;
                _width = _targetTexture.width;
                _imgSynthesis = Camera.main.GetComponent<ImgSynthesis>();


            }
            _data = new byte[2 + 16 * _width * _height];      
            
        }

        public void setHallwayType(bool complex) {
            complexHall = complex;
        }
        public void setTrainingOrTesting(bool test) {
            testing = test;
        }

        public void setReward(byte rewardChange) {
            _reward = rewardChange;
        }
        public void setEnd(byte endChange) {
            _end = endChange;
        }
        #endregion;

        public void destroyHallway()
        {
            for(int i = 0; i < nrOfEmptyPieces; i++)
                Destroy(currentPiecesEmpty[i]);
            for (int i = 0; i < nrOfCurrentPieces; i++) 
                Destroy(currentPieces[i]);
            for (int i = 0; i < nrOfLights; i++)
                Destroy(currentLights[i]);
            hallWayMade = false;
        }

        public void makeStartHallway() { //make the beginning hallway consisting of nrOfPieces amount of hallwaypieces
            hallWayMade = true;
            if (complexHall) {
                for (int i = 0; i < nrOfEmptyPieces; i++) {
                    currentPiecesEmpty[i] = Instantiate(emptyHallWaysComplex[0], new Vector3(0, 0, currentZ - 1 - (i * 2)), Quaternion.Euler(0, 0, 0));
                }
                for (int i = 0; i < nrOfCurrentPieces; i++) {               
                    if (currentZ % 8 == 0 && currentZ != 0) {
                        int randomPieceIndex = Random.Range(0, 12);
                        currentPieces[i] = Instantiate(bigBoxPiecesComplex[randomPieceIndex], new Vector3(0, 0, currentZ + 5), Quaternion.Euler(0, 0, 0));
                    } else {
                        int randomPieceIndex = Random.Range(0, 3);
                        currentPieces[i] = Instantiate(smallBoxPiecesComplex[randomPieceIndex], new Vector3(0, 0, currentZ + 5), Quaternion.Euler(0, 0, 0));
                    }
                    currentZ += 2;
                }
            } else {
                for (int i = 0; i < nrOfEmptyPieces; i++) {
                    currentPiecesEmpty[i] = Instantiate(emptyHallWays[0], new Vector3(0, 0, currentZ - 1 - (i * 2)), Quaternion.Euler(0, 0, 0));
                }
                for (int i = 0; i < nrOfCurrentPieces; i++) {
                    int randomPieceIndex = Random.Range(0, 3);
                    if (currentZ % 8 == 0 && currentZ != 0) {
                        currentPieces[i] = Instantiate(bigBoxPieces[randomPieceIndex], new Vector3(0, 0, currentZ + 5), Quaternion.Euler(0, 0, 0));
                    } else {
                        currentPieces[i] = Instantiate(smallBoxPieces[randomPieceIndex], new Vector3(0, 0, currentZ + 5), Quaternion.Euler(0, 0, 0));

                    }
                    currentZ += 2;
                }
            }
            for (int i = 0; i < nrOfLights; i++) { 
                float randomZVariation = Random.Range(-1f, 1f); //add random amount of variation to the z position
                float randomIntensity = Random.Range(lightIntensityLowerBound, lightIntensityUpperBound); //vary the intensity of the pointlight
                currentZLights += 4;
                currentLights[i] = Instantiate(pointLight, new Vector3(0, 2.4f, currentZLights + randomZVariation), Quaternion.Euler(0, 0, 0));
                currentLights[i].GetComponent<Light>().intensity = randomIntensity;
            }
        }

        public void addEmpty() { //add empty hallway pieces behind the agent to keep the light from the back and sometimes the vision of the agent to the side consistent
            if (complexHall) {
                Destroy(currentPiecesEmpty[currentIndexEmpty]);
                currentPiecesEmpty[currentIndexEmpty] = Instantiate(emptyHallWaysComplex[0], new Vector3(0, 0, currentZEmpty + 5), Quaternion.Euler(0, 0, 0));
                currentZEmpty += 2;
                if (currentIndexEmpty < nrOfEmptyPieces - 1) {
                    currentIndexEmpty++;
                } else {
                    currentIndexEmpty = 0;
                }
            } else {
                Destroy(currentPiecesEmpty[currentIndexEmpty]);
                currentPiecesEmpty[currentIndexEmpty] = Instantiate(emptyHallWays[0], new Vector3(0, 0, currentZEmpty + 5), Quaternion.Euler(0, 0, 0));
                currentZEmpty += 2;
                if (currentIndexEmpty < nrOfEmptyPieces - 1) {
                    currentIndexEmpty++;
                } else {
                    currentIndexEmpty = 0;
                }
            }
        }

        public void updateHallway() { //function to update the hallway by removing the piece in the back and spawning in a piece in front
            addEmpty();
            Destroy(currentPieces[currentIndex]);
            if (complexHall) {                
                if (currentZ % 8 == 0) {
                    int randomPieceIndex = Random.Range(0, 12);
                    currentPieces[currentIndex] = Instantiate(bigBoxPiecesComplex[randomPieceIndex], new Vector3(0, 0, currentZ + 5), Quaternion.Euler(0, 0, 0));
                    currentPieces[currentIndex].GetComponentInChildren<MeshFilter>().mesh.RecalculateNormals();
                } else {
                    int randomPieceIndex = Random.Range(0, 3);
                    currentPieces[currentIndex] = Instantiate(smallBoxPiecesComplex[randomPieceIndex], new Vector3(0, 0, currentZ + 5), Quaternion.Euler(0, 0, 0));
                    currentPieces[currentIndex].GetComponentInChildren<MeshFilter>().mesh.RecalculateNormals();
                }
            } else {
                int randomPieceIndex = Random.Range(0, 3);
                if (currentZ % 8 == 0) {
                    currentPieces[currentIndex] = Instantiate(bigBoxPieces[randomPieceIndex], new Vector3(0, 0, currentZ + 5), Quaternion.Euler(0, 0, 0));
                    currentPieces[currentIndex].GetComponentInChildren<MeshFilter>().mesh.RecalculateNormals();
                } else {
                    currentPieces[currentIndex] = Instantiate(smallBoxPieces[randomPieceIndex], new Vector3(0, 0, currentZ + 5), Quaternion.Euler(0, 0, 0));
                    currentPieces[currentIndex].GetComponentInChildren<MeshFilter>().mesh.RecalculateNormals();
                }
            }
            currentZ +=2;
            if (currentIndex < nrOfCurrentPieces -1) {
                currentIndex++;
            } else {
                currentIndex = 0;
            }
            if (updateLightsNow) {
                updateLights();
                updateLightsNow = false;
            } else {
                updateLightsNow = true;
            }
            
        }
        public void updateLights() { //function to add a new light in front and remove one in the back
            float randomZVariation = Random.Range(-1f, 1f); //add random amount of variation to the z position
            float randomIntensity = Random.Range(lightIntensityLowerBound, lightIntensityUpperBound); //vary the intensity of the pointlight
            currentZLights += 4;
            Destroy(currentLights[currentLightsIndex]);
            currentLights[currentLightsIndex] = Instantiate(pointLight, new Vector3(0, 2.4f, currentZLights + randomZVariation), Quaternion.Euler(0, 0, 0));
            currentLights[currentLightsIndex].GetComponent<Light>().intensity = randomIntensity;
            if (currentLightsIndex < nrOfLights - 1) {
                currentLightsIndex++;
            } else {
                currentLightsIndex = 0;
            }
        }
        public void makeTestHallwayArray() { //make the test hallway index arrays into one big list
            List<int> listTestHall = new List<int>();
            if (complexHall) {
                listTestHall.AddRange(testHall0c);
                listTestHall.AddRange(testHall1c);
                listTestHall.AddRange(testHall2c);
                listTestHall.AddRange(testHall3c);
                listTestHall.AddRange(testHall4c);
                listTestHall.AddRange(testHall5c);
                listTestHall.AddRange(testHall6c);
                listTestHall.AddRange(testHall7c);
                listTestHall.AddRange(testHall8c);
                listTestHall.AddRange(testHall9c);
                listTestHall.AddRange(testHall10c);
                listTestHall.AddRange(testHall11c);
                listTestHall.AddRange(testHall12c);
                listTestHall.AddRange(testHall13c);
                //add 2 hallways to the end so the hallway does not just abruptly end when the test is over
                listTestHall.AddRange(testHall0c);
                listTestHall.AddRange(testHall0c);
                testHallExperiment = listTestHall.ToArray();
            } else {
                listTestHall.AddRange(testHall0);
                listTestHall.AddRange(testHall1);
                listTestHall.AddRange(testHall2);
                listTestHall.AddRange(testHall3);
                listTestHall.AddRange(testHall4);
                listTestHall.AddRange(testHall5);
                listTestHall.AddRange(testHall6);
                listTestHall.AddRange(testHall7);
                listTestHall.AddRange(testHall8);
                listTestHall.AddRange(testHall9);
                listTestHall.AddRange(testHall10);
                listTestHall.AddRange(testHall11);
                listTestHall.AddRange(testHall12);
                listTestHall.AddRange(testHall13);
                //add 2 hallways to the end so the hallway does not just abruptly end when the test is over
                listTestHall.AddRange(testHall0);
                listTestHall.AddRange(testHall0);
                testHallExperiment = listTestHall.ToArray();
            }
            

        }
        public void makeStartTestHallway() { //Initiate the first nrAmountOfPieces for the test hallway
            hallWayMade = true;
            if (complexHall) {
                for (int i = 0; i < nrOfEmptyPieces; i++)
                    currentPiecesEmpty[i] = Instantiate(emptyHallWaysComplex[0], new Vector3(0, 0, currentZ - 1 - (i * 2)), Quaternion.Euler(0, 0, 0));

                for (int i = 0; i < nrOfCurrentPieces; i++) 
                {
                    if (testHallExperiment[indexTestHall] > 2) {
                        currentPieces[i] = Instantiate(bigBoxPiecesComplex[testHallExperiment[indexTestHall]-3], new Vector3(0, 0, currentZ + 5), Quaternion.Euler(0, 0, 0));
                    } else {
                        currentPieces[i] = Instantiate(smallBoxPiecesComplex[testHallExperiment[indexTestHall]], new Vector3(0, 0, currentZ + 5), Quaternion.Euler(0, 0, 0));
                    }
                    currentZ += 2;
                    indexTestHall += 1;
                }

            } else {
                for (int i = 0; i < nrOfEmptyPieces; i++) {
                    currentPiecesEmpty[i] = Instantiate(emptyHallWays[0], new Vector3(0, 0, currentZ - 1 - (i * 2)), Quaternion.Euler(0, 0, 0));
                }
                for (int i = 0; i < nrOfCurrentPieces; i++) {
                    if (testHallExperiment[indexTestHall] > 2) {
                        currentPieces[i] = Instantiate(bigBoxPieces[testHallExperiment[indexTestHall] - 3], new Vector3(0, 0, currentZ + 5), Quaternion.Euler(0, 0, 0));
                    } else {
                        currentPieces[i] = Instantiate(smallBoxPieces[testHallExperiment[indexTestHall]], new Vector3(0, 0, currentZ + 5), Quaternion.Euler(0, 0, 0));

                    }
                    currentZ += 2;
                    indexTestHall += 1;
                }
            }
            for (int i = 0; i < nrOfLights; i++) { 
                float randomZVariation = Random.Range(-1f, 1f); //add random amount of variation to the z position
                float randomIntensity = Random.Range(lightIntensityLowerBound, lightIntensityUpperBound); //vary the intensity of the pointlight
                currentZLights += 4;
                currentLights[i] = Instantiate(pointLight, new Vector3(0, 2.4f, currentZLights + randomZVariation), Quaternion.Euler(0, 0, 0));
                currentLights[i].GetComponent<Light>().intensity = randomIntensity;
            }
        }
        public void updateTestHallway() { //function to update the testhall by removing the piece in the back and spawning in a piece in front
            addEmpty();
            Destroy(currentPieces[currentIndex]);
            if (complexHall) {
                if (testHallExperiment[indexTestHall] > 2) {
                    currentPieces[currentIndex] = Instantiate(bigBoxPiecesComplex[testHallExperiment[indexTestHall] - 3], new Vector3(0, 0, currentZ + 5), Quaternion.Euler(0, 0, 0));
                } else {
                    currentPieces[currentIndex] = Instantiate(smallBoxPiecesComplex[testHallExperiment[indexTestHall]], new Vector3(0, 0, currentZ + 5), Quaternion.Euler(0, 0, 0));
                }
            } else {
                if (testHallExperiment[indexTestHall] >2) {
                    currentPieces[currentIndex] = Instantiate(bigBoxPieces[testHallExperiment[indexTestHall]-3], new Vector3(0, 0, currentZ + 5), Quaternion.Euler(0, 0, 0));
                } else {
                    currentPieces[currentIndex] = Instantiate(smallBoxPieces[testHallExperiment[indexTestHall]], new Vector3(0, 0, currentZ + 5), Quaternion.Euler(0, 0, 0));
                }
            }
            currentZ += 2;
            indexTestHall += 1;
            if (currentIndex < nrOfCurrentPieces - 1) {
                currentIndex++;
            } else {
                currentIndex = 0;
            }
            if (updateLightsNow) {
                updateLights();
                updateLightsNow = false;
            } else {
                updateLightsNow = true;
            }
        }
    }
}