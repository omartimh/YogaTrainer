let video;
let poseNet;
let pose;
let skeleton;
let brain;
let poseLabel = "";
let state = 'waiting';
let targetLabel;

var poseImage;
var level;
var checkCounter = "";

function preload() {
   poseImage = loadImage('images/1.jpg');
  level = 1;
}

function setup() {
  createCanvas(640, 480);
  video = createCapture(VIDEO);
  video.hide();
  poseNet = ml5.poseNet(video, poseNetLoaded);
  poseNet.on('pose', detectPose);
  let options = {
    inputs: 34,
    outputs: 5,
    task: 'classification',
    debug: true
  }
  brain = ml5.neuralNetwork(options);
  
  if (level == 1) {
    // LOAD PRETRAINED MODEL
    const modelInfo = {
      model: 'model1/model.json',
      metadata: 'model1/model_meta.json',
      weights: 'model1/model.weights.bin',
    };
    brain.load(modelInfo, neuralNetworkLoaded);
    
    // LOAD TRAINING DATA
    brain.loadData('train/train1.json', trainData);
  }
}

function guide() {
  console.log("~ Press any key (label) to start collecting poses");
  console.log(`~ Press "s" key to save collected data to a JSON file`);
  console.log(`~ Press "t" key to train collected data (2 Minimum)`);
  console.log(" ");
} 

function poseNetLoaded() {
  console.log('PoseNet is Ready!');
  console.log(" ");
  //guide();
}

function detectPose(poses) {
  if (poses.length > 0) {
    pose = poses[0].pose;
    skeleton = poses[0].skeleton;
    if (state == 'collecting') {
      let inputs = [];
      for (let i = 0; i < pose.keypoints.length; i++) {
        let x = pose.keypoints[i].position.x;
        let y = pose.keypoints[i].position.y;
        inputs.push(x);
        inputs.push(y);
      }
      let target = [targetLabel];
      brain.addData(inputs, target);
    }
  }
}

function neuralNetworkLoaded() {
  console.log('Pose Classification is Ready !');
  console.log('Classifying ...');
  state = "classifying";

  classifyPose();
}

function classifyPose() {
  if (pose) {
    let inputs = [];
    for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      inputs.push(x);
      inputs.push(y);
    }
    brain.classify(inputs, classifiedData);
  } else {
    setTimeout(classifyPose, 150);
  }
}

function trainData() {
  brain.normalizeData();
  /*brain.train({
    epochs: 150
  }, saveData);*/
}

function saveData() {
  console.log('Model is Trained !');
  brain.save();
  classifyPose();
}

function classifiedData(error, results) {  
  if (state == "classifying" && results[0].confidence > 0.75) {
    poseLabel = results[0].label.toUpperCase();
    if (poseLabel == 'R') {
      poseLabel = "Right"
    } else if (poseLabel == "W") {
      poseLabel = "Wrong";           
    }
  }

  classifyPose();
}

var check = setInterval(function() {
  if (state == "classifying")
    checkPoseCounter();
}, 1000);

function checkPoseCounter() {
  checkCounter = 3;
  if (poseLabel == "Right") {
    var checkPose = setInterval(function() {
      if (checkCounter > 0) {
        checkCounter -= 1;
      } else {
        checkCounter = "";
        state = "waiting";
        poseLabel = "Good Job!";
        clearInterval(checkPose);
        setTimeout(function() {
          level += 1;
          if (level == 2 && state == "waiting") {
            state = "classifying";
            poseImage = loadImage('images/2.png');
            
            // LOAD PRETRAINED MODEL
            const modelInfo = {
              model: 'model2/model.json',
              metadata: 'model2/model_meta.json',
              weights: 'model2/model.weights.bin',
            };
            brain.load(modelInfo, neuralNetworkLoaded);

            // LOAD TRAINING DATA
            brain.loadData('train/train2.json', trainData);
            
            level = 2;
            return;
          }
          if (level > 3 && state == "waiting") {
            state = "classifying";
            poseImage = loadImage('images/3.png');
            
            // LOAD PRETRAINED MODEL
            const modelInfo = {
              model: 'model3/model.json',
              metadata: 'model3/model_meta.json',
              weights: 'model3/model.weights.bin',
            };
            brain.load(modelInfo, neuralNetworkLoaded);

            // LOAD TRAINING DATA
            brain.loadData('train/train3.json', trainData);
            
            return;
          }
        }, 3000)
      }
    }, 1000);
  } else if (poseLabel == "Wrong") {
    checkCounter = 3;
  }
}

function keyPressed() {
  // Train
  if (key == 't') {
    console.log("Training");
    console.log(" ");
    brain.normalizeData();
    brain.train({epochs: 150}, trainData); 
    guide();
    
    // Save
  } else if (key == 's') {
    console.log("Saving");
    console.log(" ");
    brain.saveData();
    guide();
    
    // Collect (Any Key Pressed)
  } else {
    var duration = 30000;
    var timeout = duration / 1000;
    targetLabel = key.toUpperCase();
    console.log("Choosen Label: " + targetLabel);
    var timer = 3;
    var f = false;
    var countdown = setInterval(function(){
      if (timer > 0) {
        console.log(timer);
        timer -= 1;
      } else if (timer == 0 && !f) {
        f = true;
        console.log("Collecting Started (" + timeout + " seconds)");
        state = 'collecting';
        setTimeout(function() {
          console.log(">>> Collecting Complete <<<");
          console.log(" ");
          state = 'waiting';
          guide();
        }, duration);
      }
    }, 1000);
  }
}

function draw() {
  push();
  translate(video.width, 0);
  scale(-1, 1);
  image(video, 0, 0, video.width, video.height);

  if (pose) {
    for (let i = 0; i < skeleton.length; i++) {
      let a = skeleton[i][0];
      let b = skeleton[i][1];
      strokeWeight(2);
      stroke(0);
      line(a.position.x, a.position.y, b.position.x, b.position.y);
    }
    
    for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      fill(0);
      stroke(255);
      ellipse(x, y, 16, 16);
    }
  }
  
  pop();
  noStroke();
  textSize(60);
  if (poseLabel == "Right") {
    fill(0, 255, 0);
    text(poseLabel, 480, 50);
  } else if (poseLabel == "Wrong") {
    fill(255, 0, 0);
    text(poseLabel, 450, 50);
  } else {
    fill(0);
    text(poseLabel, 350, 50);
  }
  
  image(poseImage, 0, 0, 100, 160);
  
  fill(255);
  noStroke();
  textSize(50);
  text(checkCounter, 550, 420);
}