using UnityEngine;

namespace indoorMobility.Scripts.Utils
{
    [CreateAssetMenu(menuName = "Neuromatics/AppData")]
    public class AppData : ScriptableObject
    {

        [SerializeField] private string _ipAddress;
        [SerializeField] private int _port;
        [SerializeField] private byte _fwd;
        [SerializeField] private byte _side;
        [SerializeField] private byte _box;
        [SerializeField] private byte _wall;
        [SerializeField] private bool _complex;


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

        public byte Forward
        {
            get => _fwd;
            set => _fwd = value;
        }

        public byte Side
        {
            get => _side;
            set => _side = value;
        }

        public byte Box
        {
            get => _box;
            set => _box = value;
        }

        public byte Wall
        {
            get => _wall;
            set => _wall = value;
        }

        public bool Complex
        {
            get => _complex;
            set => _complex = value;
        }

        private void Reset()
        {

            _ipAddress = "127.0.0.1";
            _port =  13000;
            _fwd = (byte) 10;
            _side = (byte) 101;
            _box = (byte) 120;
            _wall = (byte) 110;
            _complex = true;

        }
    }
}