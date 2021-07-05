using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;

namespace umu7.Neuromatics.Scripts.Utils
{




    public class GUIHandler : MonoBehaviour
    {
        [SerializeField] private AppData appData;
        [SerializeField] private Text ipField;
        [SerializeField] private Text portField;
        [SerializeField] private Text fwdField;
        [SerializeField] private Text sideField;
        [SerializeField] private Text boxField;
        [SerializeField] private Text wallField;

        [SerializeField] private Toggle complexToggle;



        public void ChangeIP ()
        {
            appData.IpAddress = ipField.text;
        }

        public void ChangePort()
        {
            appData.Port = int.Parse(portField.text);
        }

        public void ChangeForward()
        {
            appData.Forward = (byte) int.Parse(fwdField.text);
        }

        public void ChangeSide()
        {
            appData.Side = (byte) int.Parse(sideField.text);
        }

        public void ChangeBox()
        {
            Debug.Log(boxField.text);
            appData.Box = (byte) int.Parse(boxField.text);
        }

        public void ChangeWall()
        {
            appData.Wall = (byte) int.Parse(wallField.text);
        }



        public void StartServer()
        {
            SceneManager.LoadScene(1);
        }

        public void QuitApplication()
        {
            Application.Quit();
            SceneManager.LoadScene(0);
        }

    }
}