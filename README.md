# BlinkRest

BlinkRest is a macOS computer-vision application that monitors whether a user is continuously focused on the screen and launches a guided eye-rest break after a selected amount of screen-focus time.

The app uses webcam-based eye, face, and gaze measurements to estimate whether the user is looking at the screen, looking away, or closing their eyes.

## Purpose

People often spend long periods looking at a computer without taking proper visual breaks.

BlinkRest is designed to encourage healthier screen habits by:

- Tracking continuous screen-focus time
- Starting a break after the focus threshold is reached
- Guiding the user through an eye-rest period
- Pausing or adjusting the break when the expected behavior is not detected
- Running in the background while the user works

BlinkRest is an educational wellness project and is not a medical device.

## Features

- Webcam-based screen-focus detection
- Detects whether the user is looking at the screen
- Detects looking away
- Detects eye closure
- Configurable focus duration
- Guided break timer
- Background monitoring
- Circular visual break timer
- Menu-bar or tray controls
- Start and stop controls
- Optional hotkey support
- Machine-learning gaze classifier
- Locally stored trained model
- Local camera processing
- No intentional uploading of camera footage

## How It Works

BlinkRest uses OpenCV and MediaPipe to detect facial and eye landmarks from the webcam.

The application may calculate features such as:

- Eye aspect ratio
- Iris position
- Eye openness
- Horizontal gaze position
- Vertical gaze position
- Face orientation
- Relative landmark distances

A trained machine-learning model classifies the user’s current state into categories such as:

```text
screen
away
closed
