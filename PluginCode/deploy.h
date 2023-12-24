
#pragma once

#include "Source/DeploymentThreads/DeploymentThread.h"

class PluginDeploymentThread: public DeploymentThread {
public:
    // initialize your deployment thread here
    PluginDeploymentThread():DeploymentThread() {

    }

    // this method runs on a per-event basis.
    // the majority of the deployment will be done here!
    std::pair<bool, bool> deploy (
        std::optional<MidiFileEvent> & new_midi_event_dragdrop,
        std::optional<EventFromHost> & new_event_from_host,
        bool gui_params_changed_since_last_call,
        bool new_preset_loaded_since_last_call,
        bool new_midi_file_dropped_on_visualizers,
        bool new_audio_file_dropped_on_visualizers) override {


        // Try loading the model if it hasn't been loaded yet
        if (!isModelLoaded) {
            load("drumLoopVAE.pt");
        }

        // Check if voice map should be updated
        bool voiceMapChanged = false;
        if (gui_params_changed_since_last_call) {
            voiceMapChanged = updateVoiceMap();
        }

        // generate a new pattern on button request
        bool generatedNewPattern = false;
        if (gui_params.wasButtonClicked("Randomize")) {
            if (isModelLoaded)
            {
                // Generate a random pattern
                generateRandomPattern();
                generatedNewPattern = true;
            }
        }

        // if the voice map has changed, or a new pattern has been generated,
        // prepare the playback sequence
        if ((voiceMapChanged || generatedNewPattern) && isModelLoaded) {
            preparePlaybackSequence();
            preparePlaybackPolicy();
            return {true, true};
        }

        // your implementation goes here
        return {false, false};
    }

private:
    // add any member variables or methods you need here
    torch::Tensor latentVector;
    torch::Tensor voice_thresholds = torch::ones({ 9 }, torch::kFloat32) * 0.5f;
    torch::Tensor max_counts_allowed = torch::ones({ 9 }, torch::kFloat32) * 32;
    int sampling_mode = 0;
    float temperature = 1.0f;
    std::map<int, int> voiceMap;

    torch::Tensor hits;
    torch::Tensor velocities;
    torch::Tensor offsets;

    // checks if any of the voice map parameters have been updated and updates the voice map
    // returns true if the voice map has been updated
    bool updateVoiceMap() {
        bool voiceMapChanged = false;
        if (gui_params.wasParamUpdated("Kick")) {
            voiceMap[0] = int(gui_params.getValueFor("Kick"));
            voiceMapChanged = true;
        }
        if (gui_params.wasParamUpdated("Snare")) {
            voiceMap[1] = int(gui_params.getValueFor("Snare"));
            voiceMapChanged = true;
        }
        if (gui_params.wasParamUpdated("ClosedHat")) {
            voiceMap[2] = int(gui_params.getValueFor("ClosedHat"));
            voiceMapChanged = true;
        }
        if (gui_params.wasParamUpdated("OpenHat")) {
            voiceMap[3] = int(gui_params.getValueFor("OpenHat"));
            voiceMapChanged = true;
        }
        if (gui_params.wasParamUpdated("LowTom")) {
            voiceMap[4] = int(gui_params.getValueFor("LowTom"));
            voiceMapChanged = true;
        }
        if (gui_params.wasParamUpdated("MidTom")) {
            voiceMap[5] = int(gui_params.getValueFor("MidTom"));
            voiceMapChanged = true;
        }
        if (gui_params.wasParamUpdated("HighTom")) {
            voiceMap[6] = int(gui_params.getValueFor("HighTom"));
            voiceMapChanged = true;
        }
        if (gui_params.wasParamUpdated("Crash")) {
            voiceMap[7] = int(gui_params.getValueFor("Crash"));
            voiceMapChanged = true;
        }
        if (gui_params.wasParamUpdated("Ride")) {
            voiceMap[8] = int(gui_params.getValueFor("Ride"));
            voiceMapChanged = true;
        }
        return voiceMapChanged;
    }

    void generateRandomPattern() {
        PrintMessage("Generating new sequence...");

        // Generate a random latent vector
        latentVector = torch::randn({ 1, 128});

        // Prepare above for inference
        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back(latentVector);
        inputs.emplace_back(voice_thresholds);
        inputs.emplace_back(max_counts_allowed);
        inputs.emplace_back(sampling_mode);
        inputs.emplace_back(temperature);

        // Get the scripted method
        auto sample_method = model.get_method("sample");

        // Run inference
        auto output = sample_method(inputs);

        // Extract the generated tensors from the output
        hits = output.toTuple()->elements()[0].toTensor();
        velocities = output.toTuple()->elements()[1].toTensor();
        offsets = output.toTuple()->elements()[2].toTensor();
    }

    // extracts the generated pattern into a PlaybackSequence
    void preparePlaybackSequence() {
        if (!hits.sizes().empty()) // check if any hits are available
        {
            // clear playback sequence
            playbackSequence.clear();

            // iterate through all voices, and time steps
            int batch_ix = 0;
            for (int step_ix = 0; step_ix < 32; step_ix++)
            {
                for (int voice_ix = 0; voice_ix < 9; voice_ix++)
                {

                    // check if the voice is active at this time step
                    if (hits[batch_ix][step_ix][voice_ix].item<float>() > 0.5)
                    {
                        auto midi_num = voiceMap[voice_ix];
                        auto velocity = velocities[batch_ix][step_ix][voice_ix].item<float>();
                        auto offset = offsets[batch_ix][step_ix][voice_ix].item<float>();
                        // we are going to convert the onset time to a ratio of quarter notes
                        auto time = (step_ix + offset) * 0.25f;

                        playbackSequence.addNoteWithDuration(
                            0, midi_num, velocity, time, 0.1f);

                    }
                }
            }
        }
    }

    // prepares the playback policy
    void preparePlaybackPolicy() {
        // Specify the playback policy
        playbackPolicy.SetPlaybackPolicy_RelativeToAbsoluteZero();
        playbackPolicy.SetTimeUnitIsPPQ();
        playbackPolicy.SetOverwritePolicy_DeleteAllEventsInPreviousStreamAndUseNewStream(true);
        playbackPolicy.ActivateLooping(8);
    }

};