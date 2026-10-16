using System;
using System.Collections.Generic;
using System.IO;
using AutoPlayQA.Samples.PhysicsKeyDoor;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace AutoPlayQA.Editor
{
    /// <summary>Batch-mode entry points for the PhysicsKeyDoor sample.</summary>
    public static class PhysicsKeyDoorBuild
    {
        public const string BootstrapScenePath =
            "Assets/AutoPlayQA/PhysicsKeyDoorBootstrap.unity";

        /// <summary>
        /// Quick compile/contract validation. Reaching this method already
        /// proves that Unity compiled both package assemblies.
        /// </summary>
        public static void Validate()
        {
            if (AutoPlayQAProtocol.Version != "autoplay-qa/1")
            {
                throw new BuildFailedException(
                    "PhysicsKeyDoor requires the autoplay-qa/1 protocol.");
            }

            if (!typeof(IAutoPlayQAGame).IsAssignableFrom(
                typeof(PhysicsKeyDoorGameAdapter)))
            {
                throw new BuildFailedException(
                    "PhysicsKeyDoorGameAdapter does not implement IAutoPlayQAGame.");
            }

            if (!BuildPipeline.IsBuildTargetSupported(
                BuildTargetGroup.Standalone,
                BuildTarget.StandaloneOSX))
            {
                throw new BuildFailedException(
                    "This Unity installation does not support macOS standalone builds.");
            }

            Debug.Log(
                "AutoPlay QA PhysicsKeyDoor validation passed for " +
                Application.unityVersion + ".");
        }

        /// <summary>
        /// Create or repair the generated bootstrap scene, put it first in
        /// build settings, and build a macOS standalone player.
        /// </summary>
        public static void BuildMacOS()
        {
            Validate();
            EnsureBootstrapScene();
            EnsureSceneInBuildSettings();

            var projectRoot = Directory.GetParent(Application.dataPath).FullName;
            var outputPath = Path.Combine(
                projectRoot,
                "Builds",
                "AutoPlayQA-Physics.app");
            Directory.CreateDirectory(Path.GetDirectoryName(outputPath));

            var previousProductName = PlayerSettings.productName;
            BuildReport report;
            try
            {
                // This also fixes the name of the executable inside the app
                // bundle, making launch commands independent of project name.
                PlayerSettings.productName = "AutoPlayQA-Physics";
                report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
                {
                    scenes = new[] { BootstrapScenePath },
                    locationPathName = outputPath,
                    target = BuildTarget.StandaloneOSX,
                    options = BuildOptions.None
                });
            }
            finally
            {
                PlayerSettings.productName = previousProductName;
                AssetDatabase.SaveAssets();
            }

            if (report.summary.result != BuildResult.Succeeded)
            {
                throw new BuildFailedException(
                    "PhysicsKeyDoor macOS build failed with result " +
                    report.summary.result + ".");
            }

            Debug.Log(
                "Built AutoPlay QA PhysicsKeyDoor player at " + outputPath +
                " (" + report.summary.totalSize + " bytes).");
        }

        private static void EnsureBootstrapScene()
        {
            EnsureAssetFolder("Assets/AutoPlayQA");

            Scene scene;
            if (File.Exists(Path.Combine(
                Directory.GetParent(Application.dataPath).FullName,
                BootstrapScenePath)))
            {
                scene = EditorSceneManager.OpenScene(
                    BootstrapScenePath,
                    OpenSceneMode.Single);
            }
            else
            {
                scene = EditorSceneManager.NewScene(
                    NewSceneSetup.EmptyScene,
                    NewSceneMode.Single);
            }

            PhysicsKeyDoorBootstrap bootstrap = null;
            var roots = scene.GetRootGameObjects();
            for (var index = 0; index < roots.Length && bootstrap == null; index++)
            {
                bootstrap = roots[index].GetComponentInChildren<
                    PhysicsKeyDoorBootstrap>(true);
            }

            if (bootstrap == null)
            {
                var bootstrapObject = new GameObject(
                    "AutoPlay QA PhysicsKeyDoor Bootstrap");
                SceneManager.MoveGameObjectToScene(bootstrapObject, scene);
                bootstrap = bootstrapObject.AddComponent<PhysicsKeyDoorBootstrap>();
            }

            // Repair an existing generated scene that was accidentally saved
            // with the bootstrap or one of its parents disabled.
            bootstrap.enabled = true;
            var current = bootstrap.transform;
            while (current != null)
            {
                current.gameObject.SetActive(true);
                current = current.parent;
            }

            EditorUtility.SetDirty(bootstrap);

            if (scene.path != BootstrapScenePath || scene.isDirty)
            {
                if (!EditorSceneManager.SaveScene(scene, BootstrapScenePath))
                {
                    throw new BuildFailedException(
                        "Could not save the PhysicsKeyDoor bootstrap scene.");
                }
            }
        }

        private static void EnsureSceneInBuildSettings()
        {
            var scenes = new List<EditorBuildSettingsScene>();
            var existing = EditorBuildSettings.scenes;
            for (var index = 0; index < existing.Length; index++)
            {
                if (!string.Equals(
                    existing[index].path,
                    BootstrapScenePath,
                    StringComparison.Ordinal))
                {
                    scenes.Add(existing[index]);
                }
            }

            scenes.Insert(0, new EditorBuildSettingsScene(BootstrapScenePath, true));
            EditorBuildSettings.scenes = scenes.ToArray();
            AssetDatabase.SaveAssets();
        }

        private static void EnsureAssetFolder(string folderPath)
        {
            if (AssetDatabase.IsValidFolder(folderPath))
            {
                return;
            }

            var parent = Path.GetDirectoryName(folderPath).Replace('\\', '/');
            var name = Path.GetFileName(folderPath);
            EnsureAssetFolder(parent);
            AssetDatabase.CreateFolder(parent, name);
        }
    }
}
