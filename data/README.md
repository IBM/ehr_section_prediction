# EHR Section Prediction Data

The following folder contains the data and vocabulary for the Medical Literature data from Wikipedia.

This data is licensed under the Wikipedia CC-BY-SA license. For more information please visit: https://creativecommons.org/licenses/by-sa/3.0/

Please send any questions/concerns to Sara Rosenthal at sjrosenthal@us.ibm.com

### Data Format

There are two sets of data, sentence level and section level. Each contains the same data but they are divided up differently.

```
sentence level: by sentence (The header is also a sentence)
section level: by paragraph (The header is included)
```

The data format is as follows, each pkl file contains the training/development data with one instance per line.

Each line has the following format:

```
doc_id.paragraph_num, text, labels, header, source, paragraph line in source

example:
0 = '4690.0'
1 = 'in an acute context hypoxemia can cause symptoms such as those in respiratory distress'
2 = 'Chief Complaint'
3 = 'signs and symptoms'
4 = 'WikipediaMedical'
5 = '2'
```
