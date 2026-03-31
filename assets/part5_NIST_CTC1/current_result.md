This file contains the result of the latest run of part 5's revision drawing pairs. Each characteristic will be marked as correct, partially correct, or incorrect (with correction). Please review this result and investigate the issues and fix.

The changes made in rev B involve shifting the location of the characteristic annotations on the drawing PDF compared to their former locations on rev A drawing, as well as the addition and removal of characteristics.

- Char 1: unchanged (rev A: Ø35 +0.2/-0.2 mm, rev B: 35.2) <- Partially correct. The unchanged status is correctly classified, but rev B should be "Ø35.2 / Ø34.8" or "Ø35 +0.2/-0.2" just like rev A. It's the same in both revisions. It's presented as a limit tolerance with the upper limit written above the lower limit.
- Char 2: removed (rev A: Ø35 +0.2/-0.2 mm, rev B: doesn't exist) <- Incorrect. The correct status should be unchanged. Rev B should be the "Ø35.2 / Ø34.8" or "Ø35 +0.2/-0.2" just like rev A. It's the same in both revision, and it's another case of limit tolerance.
- Char 3: unchanged (rev A: Ø35 +0.2/-0 mm, rev B: Ø35) <- Partially correct. The unchanged status is correctly classified, but rev B should be "Ø35 +0.2/-0" just like rev A.
- Char 4: removed (rev A: Ø35 +0/-0.2 mm, rev B: doesn't exist) <- Correct
- Char 5: unchanged (rev A: Ø25 +0.15/-0.15 mm, rev B: Ø25±0.15) <- Correct
- Char 6: unchanged (rev A: 60° +0.5°/-0.5° deg, rev B: 60°±0.5°) <- Correct
- Char 7: removed (rev A: Ø20 +0.05/-0.1 mm, rev B: doesn't exist) <- Incorrect. The correct status should be unchanged. Rev B should be "Ø20 +0.05/-0.1" just like rev A. In the drawing, the annotation is moved diagonally upward to the right compared to its old location on rev A drawing.
- Char 8: removed (rev A: Ø20 +0.1/-0.05 mm, rev B: doesn't exist) <- Incorrect. The correct status should be unchanged. Rev B should be "Ø20 +0.1/-0.05" just like rev A. It's the same case as char 7.
- Char 9: unchanged (rev A: ⏥0.2, rev B: ⏥ 0.2) <- Correct
- Char 10: removed (rev A: ⌓1.25ABC, rev B: doesn't exist) <- Correct
- Char 11: removed (rev A: ⌓0.5A, rev B: doesn't exist) <- Incorrect. The correct status should be unchanged. Rev B should be "⌓ 0.5A" just like rev A.
- Char 12: removed (rev A: ⟂1.5A, rev B: doesn't exist) <- Incorrect. The correct status should be unchanged. Rev B should be "⟂ 1.5A" just like rev A.
- Char 13: added (rev A: doesn't exist, rev B: -0.1) <- Incorrect. This is an excessive characteristic and shouldn't exist. The program mistakenly read the "-0.1" on rev B from the unchanged char 7 that is "Ø20 +0.05/-0.1".
- Char 14: added (rev A: doesn't exist, rev B: -0) <- Incorrect. This is an excessive characteristic and shouldn't exist. The program mistakenly read the "-0" on rev B from the unchanged char 3 that is "Ø35 +0.2/-0".
- Char 15: added (rev A: doesn't exist, rev B: ⏥ 0.2) <- Incorrect. This is an excessive characteristic and shouldn't exist. The program double-read the unchanged char 9.
- Char 16: added (rev A: doesn't exist, rev B: 35.2) <- Incorrect. This is an excessive characteristic and shouldn't exist. The program mistakenly read the "35.2" on rev B from the unchanged char 2 that is "Ø35.2 / Ø34.8" or "Ø35 +0.2/-0.2".
- Char 17: added (rev A: doesn't exist, rev B: Ø20) <- Incorrect. This is an excessive characteristic and shouldn't exist. The program mistakenly read the "Ø20" on rev B from the unchanged char 7 that is "Ø20 +0.05/-0.1".
- Char 18: added (rev A: doesn't exist, rev B: Ø20) <- Incorrect. This is an excessive characteristic and shouldn't exist. The program mistakenly read the "Ø20" on rev B from the unchanged char 8 that is "Ø20 +0.1/-0.05".
- Char 19: added (rev A: doesn't exist, rev B: 3X Ø18 30) <- Partially correct. The added status is correctly classified, but rev B should be "3X Ø18 ↧30 M20x2.5 − 6H ↧6", or "3X Ø18 ↧30" and "M20x2.5 − 6H ↧6" as two separate added characteristics.

The program has missed these added characteristics in rev B (in no particular order):

- Char 20: added (rev A: doesn't exist, rev B: 800) <- This is a general length, added to the front view
- Char 21: added (rev A: doesn't exist, rev B: 155) <- This is part of a series of ordinate dimensions added to the new left view of the part
- Char 22: added (rev A: doesn't exist, rev B: 225) <- This is part of a series of ordinate dimensions added to the new left view of the part
- Char 23: added (rev A: doesn't exist, rev B: 295) <- This is part of a series of ordinate dimensions added to the new left view of the part
- Char 24: added (rev A: doesn't exist, rev B: 450) <- This is part of a series of ordinate dimensions added to the new left view of the part
