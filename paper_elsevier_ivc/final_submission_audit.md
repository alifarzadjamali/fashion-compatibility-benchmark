# Final submission audit — Image and Vision Computing

Audit date: 15 September 2026.

## Official requirements checked

- The current [Image and Vision Computing aims and scope](https://www.sciencedirect.com/journal/image-and-vision-computing/about/aims-and-scope) and [Guide for Authors](https://www.sciencedirect.com/journal/image-and-vision-computing/publish/guide-for-authors) were checked for article fit, initial-submission format, references, declarations, supplementary material, highlights, graphical abstracts, authorship, and data/code statements. The paper fits the journal's computer-vision remit through controlled quantitative comparison, performance evaluation, relational visual reasoning, and benchmark validity.
- The project uses the official Elsevier `elsarticle` class, version 3.5 (9 January 2026), in the clean `preprint,12pt` review layout. Numbered citations use `elsarticle-num`, matching the journal's sequential bracketed reference presentation. Journal-specific settings are isolated near the top of `main.tex` so the scientific body does not need to change for another Elsevier venue.
- Elsevier's current [LaTeX instructions](https://www.elsevier.com/researcher/author/policies-and-guidelines/latex-instructions) were followed. The source bundle includes the class, bibliography style, bibliography, tables, and figures needed for an independent build.
- `highlights.txt` contains five highlights; the longest is 79 characters including its bullet marker, within Elsevier's current limit of 85 characters. Elsevier's [highlights guidance](https://www.elsevier.com/researcher/author/tools-and-resources/highlights) states that highlights are submitted separately and are not part of editorial consideration. No graphical abstract was created because one was not identified as mandatory for this initial IVC submission; the live submission portal should be checked once more immediately before upload in case journal requirements change.
- Elsevier-style CRediT, competing-interest, funding, ethics, data-availability, code/reproducibility, acknowledgement, and supplementary-material statements are present in the manuscript, with separate editable declaration files where useful for upload.

## Journal-specific changes made

- Preserved `paper/` byte-for-byte as `paper_springer_mva/` before deriving this independent folder. A recursive SHA-256 comparison found 45 files in each folder and zero differences; the original `paper/` was not modified.
- Replaced Springer front matter and declaration headings with Elsevier front matter while preserving the title, abstract, keywords, five-author order, University of Salford affiliation, and contribution balance. Ali Jamali remains first and corresponding author (`a.jamali2@salford.ac.uk`); Taha Mansouri remains last author.
- Retained the four Gap–Objective–RQ–Contribution–Evidence paths and all four explicit Discussion answers. Two small journal-fit edits connect fashion compatibility to representation transfer and relational reasoning in computer vision; no scientific claim, method, result, or limitation was changed.
- Converted “Online Resource 1” to independently compiled Elsevier supplementary material with S-prefixed sections, tables, and figures. All content and all numerical values were retained.
- Reflowed floats for the single-column review layout. Vector PDF figures remain vector; wide tables were adjusted without changing values. Page-by-page inspection found no clipping, overlap, unreadable legend, broken caption, or material whitespace defect.
- Retained all 81 cited works. The Elsevier bibliography has 81 unique keys, all cited and none missing. Seven records were normalized to eliminate BibTeX warnings and correct publication metadata, including the author list of *The Efficiency Misnomer*, AAAI article pagination/types, NeurIPS pagination, and non-paginated ICLR/workshop status. No cited work or citation meaning was removed or changed.

## Scientific-content and build verification

- Frozen results, model values, confidence intervals, protocol hashes, dataset counts, A100 values, LookBench values, calibration measures, efficiency measures, and robustness results were not altered. After excluding LaTeX layout commands, the ordered sequence of all 821 numerical tokens in the main scientific body is identical to the preserved Springer version; all 568 supplementary numerical tokens are also identical.
- All 18 figure files are SHA-256-identical between the Springer and Elsevier folders. The bibliography contains the same 81 works, with only the metadata corrections described above.
- Both journal folders compile independently from clean temporary staging directories. The Springer main manuscript and supplement and the Elsevier main manuscript and supplement all exit successfully with no LaTeX errors, unresolved citations or references, `??`, missing characters, overfull boxes, or BibTeX warnings.
- The Elsevier PDF has 43 pages and the supplementary PDF has 9 pages. Both were inspected page by page. Author order, corresponding-author marking, figures, tables, equations, citations, cross-references, declarations, and the complete reference list render correctly.

## Remaining manual tasks before submission

- Create the promised tagged public archive (for example, GitHub plus Zenodo) and replace future-release wording with the permanent URL/identifier only after the archive actually exists. No DOI or identifier has been invented.
- Recheck the live IVC Editorial Manager checklist on the upload date, particularly any graphical-abstract field. If highlights are requested as a Word file, convert `highlights.txt` without altering the five approved lines.
- Editorial Manager does not process LaTeX subfolders. For upload, make a flat submission archive or adjust copied figure/table paths in an upload-only bundle; keep this structured, independently compiling source folder as the archival master.
- Obtain final approval of the manuscript, CRediT statement, declarations, and correspondence details from all five authors.

## Portability to JVCIR

To retarget this version to the *Journal of Visual Communication and Image Representation*, change `\TargetJournal` and then verify JVCIR's current Guide for Authors, article type, reference style, highlights, graphical-abstract, and declaration requirements. The `elsarticle` source, scientific body, supplementary material, figures, and frozen results require no substantive change unless JVCIR's current instructions explicitly demand one.

## Verdict

The two independent journal versions are complete and clean. `paper_springer_mva/` is an exact preservation copy of the finished MVA package, and `paper_elsevier_ivc/` is a submission-ready IVC-format version with unchanged science and validated standalone builds. Only the listed author-approval, archival-release, and upload-portal tasks remain.
