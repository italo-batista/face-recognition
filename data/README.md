## Expected folders strucutre

data /
  - **faces/**: imgs of users to recognize using `face-recognition.livecam`
  - **test/**: extracted from vggface2 data
    - label1/
    - label2/
    - ...
    - labelX/
      - img1.jpg
      - img859053.jpg
  - **train/**: extracted from vggface2 data
    - label1453/
    - label25345/
    - ...
    - labelX/
      - img15345.jpg
      - img8546653.jpg
  - **aligned-images**/
    - test/: test data after preprocessing* 
      - label1/
      - label2/
      - ...
      - labelX/
        - img1.jpg
        - img859053.jpg    
    - train/: train data after preprocessing*
      - label1453/
      - label25345/
      - ...
      - labelX/
        - img15345.jpg
        - img8546653.jpg

* last step at [step3](https://github.com/italo-batista/ia-ia-oh/blob/master/steps/step3.md)
