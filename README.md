# AI Particle Art

Ένα πειραματικό project Computer Vision και Digital Art που μετατρέπει την ανθρώπινη κίνηση σε ένα δυναμικό νεφέλωμα από σωματίδια. Η εφαρμογή χρησιμοποιεί τεχνητή νοημοσύνη για να ανιχνεύσει το σώμα και να δημιουργήσει ένα εφέ "αποσύνθεσης" (disintegration/shedding) που εκτοξεύεται από τον χρήστη στο χώρο.

## Χαρακτηριστικά

    Real-time Human Segmentation: Χρήση του MediaPipe Selfie Segmenter για απομόνωση της σιλουέτας με υψηλή ακρίβεια.

    Dynamic Particle System: Διαχείριση 80.000+ σωματιδίων ταυτόχρονα μέσω NumPy Vectorization.

    Canvas Warping Engine: Εφαρμογή γεωμετρικών μετασχηματισμών (warpAffine) στον καμβά για τη δημιουργία αίσθησης συνεχούς ροής.

    Motion Persistence: Σύστημα "μνήμης" καμβά (Feedback Loop) που επιτρέπει στην ενέργεια να παραμένει ορατή καθώς απομακρύνεται από την πηγή.

## Τεχνολογικό Stack

    Python 3.13

    OpenCV: Rendering, Gaussian Blurring, και Affine Transformations.

    MediaPipe (Google): * Pose Landmarker (για τον εντοπισμό του κέντρου μάζας).

        Selfie Segmenter (για τη δημιουργία της μάσκας σώματος).

    NumPy: Μαθηματικοί υπολογισμοί πινάκων για μέγιστη ταχύτητα.

## Πώς Λειτουργεί

Το πρόγραμμα δεν μετακινεί τα σωματίδια ένα-προς-ένα. Αντίθετα, χρησιμοποιεί μια τεχνική Image Feedback:

    Warp Stage: Σε κάθε frame, ο καμβάς υφίσταται ένα Zoom-in (κλίμακα ~1.03) με κέντρο τους ώμους του χρήστη. Αυτό σπρώχνει τα "παλιά" pixels προς τις άκρες της οθόνης.

    Decay Stage: Ο καμβάς πολλαπλασιάζεται με έναν συντελεστή (π.χ. 0.94), κάνοντας τα παλαιότερα σωματίδια να σβήνουν σταδιακά καθώς απομακρύνονται.

    Injection Stage: Νέα λευκά σωματίδια "ψεκάζονται" τυχαία πάνω στην τρέχουσα σιλουέτα του χρήστη.

    Diffusion Stage: Ένα Gaussian Blur εφαρμόζεται στο τέλος για να ενώσει τις σπίθες σε μια ενιαία "αύρα" ενέργειας.

## Εγκατάσταση και Χρήση
1. Κλωνοποίηση
```
Bash

git clone https://github.com/jetr00/Particle-Art-Maker.git
cd Particle-Art-Maker
```

2. Εγκατάσταση Εξαρτήσεων
```
Bash

pip install opencv-python mediapipe numpy
```
3. Απαραίτητα Μοντέλα (AI Models)

Για να λειτουργήσει το project, πρέπει να κατεβάσετε τα παρακάτω αρχεία από το επίσημο MediaPipe Solutions guide και να τα τοποθετήσετε στον φάκελο του project:

    pose_landmarker_lite.task/pose_landmarker_full.task/pose_landmarker_heavy.task

    selfie_segmenter.tflite/selfie_segmenter_landscape.tflite

4. Εκτέλεση
```
Bash

python ParticleAIArtMaker.py
```
Χειρισμός

    'q' ή ';': Τερματισμός της εφαρμογής.

    Η εφαρμογή ανοίγει αυτόματα ένα upscaled παράθυρο (1280x720) για καλύτερη οπτική εμπειρία.

Developed by `Ioannis Choriatellis`
