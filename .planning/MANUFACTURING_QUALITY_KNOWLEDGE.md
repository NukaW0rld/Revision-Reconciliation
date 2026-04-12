# Manufacturing & Quality Domain Knowledge

---

## 1. GD&T: Feature Control Frames

### What is a feature control frame (FCF)?

A feature control frame (FCF) is the rectangular box on an engineering drawing that contains a GD&T callout. It is read left to right and has 2–5 compartments:
1. The geometric characteristic symbol
2. The tolerance value with any diameter prefix or material-condition modifier
3. One to three datum references (primary, secondary, tertiary)

Every GD&T callout on a drawing appears inside an FCF. FCF literacy is foundational because it tells you the control type, tolerance zone, and datum reference frame for that callout.

### How to read an FCF left to right

- **Compartment 1:** Geometric characteristic symbol (position ⊕, flatness ⏥, perpendicularity ⊥, etc.) — identifies the type of control applied
- **Compartment 2:** Tolerance value. A ⌀ prefix means the zone is cylindrical. A modifier like Ⓜ (MMC) or Ⓛ (LMC) after the value enables bonus tolerance.
- **Compartments 3–5:** Datum references (A, B, C) in order of precedence. Primary datum constrains 3 degrees of freedom (DOF), secondary constrains 2, tertiary constrains 1.

### Material Condition Modifiers

**Ⓜ (MMC — Maximum Material Condition):** Tolerance applies at maximum material size (smallest hole or largest pin). As the feature departs from MMC, tolerance increases as "bonus tolerance." Use MMC when assembly clearance is the concern (bolt patterns, pin-in-hole fits). Enables functional go/no-go gaging.

**Ⓛ (LMC — Least Material Condition):** Tolerance applies at least material size (smallest pin or largest hole). As the feature departs from LMC, tolerance also increases. Use LMC when wall-thickness preservation is the concern.

**RFS (Regardless of Feature Size):** Default under ASME Y14.5-2018 when no modifier appears. No bonus tolerance.

### FCF Compartment Count

- **2 compartments (minimum):** Form controls (flatness, straightness, circularity, cylindricity) — no datums required because they describe a single feature in isolation.
- **3–5 compartments:** Controls referencing datums. Datum compartments may also include modifiers (e.g., "B Ⓜ" for datum B at MMC).

### Which controls require datums?

- **Never require datums:** Flatness, straightness, circularity, cylindricity (form controls)
- **May or may not require datums:** Profile controls (without datums = form only; with datums = form + orientation + location)
- **Always require at least one datum:** All orientation controls (parallelism, perpendicularity, angularity), all location controls (position, concentricity, symmetry), all runout controls (circular runout, total runout)

### GD&T Symbols Reference

| Symbol | Control | Type | Requires Datum |
|--------|---------|------|----------------|
| ⏥ | Flatness | Form | No |
| ○ | Circularity | Form | No |
| ⌀ (straightness) | Straightness | Form | No |
| ⌀⌀ | Cylindricity | Form | No |
| ∥ | Parallelism | Orientation | Yes |
| ⊥ | Perpendicularity | Orientation | Yes |
| ∠ | Angularity | Orientation | Yes |
| ⊕ | True Position | Location | Yes |
| ◎ | Concentricity | Location | Yes |
| = | Symmetry | Location | Yes |
| ↗ | Circular Runout | Runout | Yes |
| ↗↗ | Total Runout | Runout | Yes |
| ⌒ | Profile of a Line | Profile | Optional |
| ⌒⌒ | Profile of a Surface | Profile | Optional |

### Datum Reference Frame (3-2-1 Rule)

The primary datum constrains 3 degrees of freedom (DOF), the secondary constrains 2, and the tertiary constrains 1. In machining, datum order maps directly to fixture design:
- **Datum A:** Primary locating surface
- **Datum B:** Secondary locator
- **Datum C:** Tertiary locator

This datum order is also critical for inspection fixturing on CMMs — the part must be set up in the same datum reference frame as specified on the drawing.

### How a machinist (and inspector) uses an FCF

The FCF determines: which features need geometric control beyond title-block tolerances, which datum features must be established first, how tight the tolerance is, and whether MMC is specified. If MMC is specified, the shop may use functional gaging (go/no-go) for production inspection instead of CMM measurement.

---

## 2. Fit Tolerances (Press Fit / Slip Fit)

Fit designations per ISO 286-1:2010 and ASME B4.2. Values shown for Ø18–30 mm range.

### Fit Class Comparison Table

| Fit | Type | Clearance / Interference | Assembly | Typical Use |
|-----|------|--------------------------|----------|-------------|
| H7/f7 | Free-running | +20 to +62 µm | Hand | Journal bearings, pump shafts |
| H7/g6 | Close-running | +7 to +41 µm | Hand / light push | Locating pins, reamer bores |
| H7/h6 | Sliding | 0 to +34 µm | Hand | Spigot joints, tooling keys |
| H7/k6 | Transition | −15 to +19 µm | Mallet / light press | Gears with key, hubs |
| H7/n6 | Transition (tight) | −28 to +6 µm | Hydraulic press | Pulleys with set screw retention |
| H7/p6 | Light press | −1 to −35 µm | Hydraulic press | Bearing races, bronze bushings |
| H7/s6 | Medium press | −14 to −48 µm | Press or heat hub | Keyless gear hubs, couplings |
| H7/u6 | Heavy press | −27 to −61 µm | Thermal (≥150°C) | Heavy-duty drive hubs |

*Clearance/interference for Ø18–30 mm range, ISO 286-1:2010.*

### How to Call Out Fits on a Drawing

**Format 1 — ISO fit designation (preferred):**
- Assembly view: Ø25 H7/p6
- Bore detail: Ø25 H7 (= Ø25.000 / Ø25.021)
- Shaft detail: Ø25 p6 (= Ø25.022 / Ø25.035)

**Format 2 — Explicit limits:**
- Bore: Ø25.000 / Ø25.021 or Ø25 +0.021/+0.000
- Shaft: Ø25.022 / Ø25.035 or Ø25 +0.035/+0.022

Always add: surface finish requirement (Ra 1.6 µm on mating surfaces) and lubrication instruction if applicable. The fit code alone does not specify these.

### Transition Fit

A transition fit occupies the zone between clearance and interference — the assembly may result in either a small clearance or a small interference depending on where individual parts land within their tolerance bands. Common examples: H7/k6 and H7/n6. Use when accurate location is needed but disassembly is also required. Requires positive retention (set screw, circlip) since transition fits do not reliably transmit torque by friction alone.

---

## 3. Surface Finish (Ra) Reference

Ra (arithmetic average roughness) per ISO 4287 — the mean deviation of the surface profile from the centerline, measured in microinches (µin.) or micrometers (µm). Lower Ra = smoother surface.

### Ra Reference Table

| Ra (µin.) | Ra (µm) | Typical Process | Application |
|-----------|---------|-----------------|-------------|
| 250 | 6.3 | Rough machining, sawing | Non-functional hidden surfaces |
| 125 | 3.2 | Standard CNC milling | General as-machined default |
| 63 | 1.6 | Fine milling, standard turning | Mating surfaces, bearing housings |
| 32 | 0.8 | Fine turning, finish milling | O-ring grooves, sealing faces |
| 16 | 0.4 | Grinding, fine boring | Precision bearing journals |
| 8 | 0.2 | Lapping, honing, polishing | Optical, gauge faces |
| 4 | 0.1 | Superfinishing, electropolishing | Medical implants, mirror finish |

### How Surface Finish Is Specified on Drawings

- Surface finish symbol (per ASME Y14.36 or ISO 1302) placed on the surface with a leader line
- The maximum Ra value is written in the symbol (e.g., "32" for Ra 32 µin., or "0.8" for Ra 0.8 µm)
- Surfaces without a callout default to the general note (typically "125 µin. Ra UOS" — Unless Otherwise Specified)
- Secondary finishes (anodize, passivation, etc.) called out in general notes with full spec: "FINISH: TYPE II ANODIZE, BLACK, PER MIL-A-8625 TYPE II CLASS 2, 0.7 MIL NOM"
- Masking requirements noted when certain features must remain uncoated (threads, press-fit bores)

### Inspection Impact

Ra is measured with a contact profilometer (diamond stylus) or optical profilometer. Surface finish callouts on drawings must be verified during inspection on applicable surfaces.

---

## 4. Quality & Inspection Terminology

### Inspection Documents

**Certificate of Conformance (CoC):** A signed document from the manufacturer certifying that parts meet all specified drawing requirements — dimensions, material, finish, and special processing. Included with production shipments.

**First Article Inspection (FAI):** A complete, documented inspection of the first production part against all drawing dimensions and specifications. Verifies that the manufacturing process can consistently produce conforming parts. Standard practice for any new production program. Includes a balloon drawing (each dimension numbered) and a dimensional report mapping every balloon to a measured value, stated tolerance, and pass/fail status.

**AS9102 FAIR (First Article Inspection Report):** The aerospace-specific FAI standard. AS9102 defines three forms:
- **Form 1:** Design documentation (drawing revision, spec list, etc.)
- **Form 2:** Material and process documentation
- **Form 3:** Characteristic accountability — the complete list of all drawing characteristics (dimensions, GD&T callouts, notes) with measured values, tolerances, and conformance status. This is the core inspection record for each characteristic on the drawing.

**Balloon Drawing:** An engineering drawing with each dimension and GD&T callout numbered (ballooned) sequentially. The FAI report uses these balloon numbers as the identifier for each characteristic. Balloon numbering must be consistent between drawing revisions for traceability.

**CMM (Coordinate Measuring Machine) Report:** Output from a coordinate measuring machine that measures the geometry of physical objects by sensing discrete points on the surface with a probe. CMM systems typically operate in the low-micron range (±0.002–0.005 mm depending on machine volume and probing strategy). CMM reports provide dimensional data for FAI documentation and are used to verify GD&T callouts.

**Material Test Report (MTR) / Mill Test Report:** A document from the material supplier certifying the chemical composition and mechanical properties of a specific heat/lot of metal. Provides raw material traceability from the mill to the finished part.

**Gage R&R (Repeatability and Reproducibility):** A measurement system analysis study that verifies the shop's inspection equipment is capable. Typically <10% GRR is acceptable for critical features. Values >30% indicate a measurement system problem.

**Non-Conformance Report (NCR):** Documentation of any part or lot that does not meet specifications. Triggers CAPA (Corrective and Preventive Action) process.

**CAPA (Corrective and Preventive Action):** Formal process for investigating root cause of a non-conformance and implementing corrective actions to prevent recurrence. Required by AS9100 and ISO 13485.

### Sampling and Statistical Concepts

**AQL (Acceptable Quality Level):** The maximum defect rate considered acceptable for a sampling plan. For CNC production inspection: Level II general inspection, 1.0% for critical dimensions, 2.5% for major dimensions. Defined per ANSI/ASQ Z1.4.

**Cpk (Process Capability Index):** A statistical measure of how well a manufacturing process can produce parts within tolerance limits, accounting for both process centering and spread.
- **Cpk ≥ 1.33:** Production-capable process (≤63 defects per million opportunities). Target for production.
- **Cpk ≥ 1.67:** Highly capable (≤0.6 DPM). Required for some aerospace applications.
- **Cpk < 1.0:** Process is not capable — producing out-of-spec parts.

**SPC (Statistical Process Control):** In-process monitoring of critical dimensions using control charts to detect drift before it produces out-of-spec parts. Monitors Cpk and triggers corrective action when capability drops.

**PPM (Parts Per Million):** Defect rate metric. Target for qualified CNC suppliers: <1,000 PPM. Action trigger: >2,500 PPM on any single lot.

### Inspection Methods

**Go/No-Go Gaging:** Functional gauging that checks whether a feature is within tolerance limits. The "go" gauge must pass (part is not too small/tight) and the "no-go" gauge must not pass (part is not too large/loose). Commonly used for threaded features, holes, and shaft diameters. Enabled by MMC modifier in GD&T callouts.

**Thread Gauging:** Class 3A/3B threads (critical connections) are inspected with calibrated thread gages — go/no-go plus thread ring and plug gages.

**Profilometer:** Instrument for measuring surface roughness (Ra). Contact type uses a diamond stylus; optical type uses light scattering. Required for verifying Ra callouts on drawings.

**Balloon Drawing (as inspection tool):** The numbered drawing used by the inspector to record measured values against each characteristic. Each balloon number corresponds to one row in the FAI Form 3.

### Tolerances and Standard Values

**Standard CNC Tolerance:** ±0.005 in. (±0.13 mm) for machined features in aluminum and steel. This is the shop default when no tolerance is explicitly called out on a feature.

**Precision Tolerance:** ±0.001–0.002 in. (±0.025–0.05 mm) for mating surfaces, press fits, and slip fits.

**Bearing Bore:** ±0.0005 in. (±0.013 mm) typical for press-fit bearing housings.

**Title Block Tolerance:** The general tolerance stated in the drawing title block, applied to all dimensions that do not have an explicit callout. Typically ±0.005 in. for machined features, ±0.010 in. for sheet metal, ±0.030 in. for cast/molded features.

---

## 5. Manufacturing Glossary (Inspection-Relevant Terms)

**ASME Y14.5:** The ANSI/ASME standard for Geometric Dimensioning and Tolerancing (GD&T). The 2018 revision is the current edition. ISO 1101 is the international equivalent.

**ASME Y14.36:** The standard for surface texture symbols on engineering drawings.

**ISO 4287:** The international standard for surface roughness parameters, including Ra (arithmetic average roughness).

**ISO 286-1:2010:** The international standard for limits and fits for cylindrical features. Defines the system of fundamental deviations and tolerance grades (H7/p6, etc.).

**ECO (Engineering Change Order):** A formal document that authorizes a change to an engineering drawing, specification, or process. ECOs identify what changed, why, and on which drawing revision. The AS9102 FAIR process requires reconciliation of characteristics when an ECO changes a drawing.

**Revision (Rev):** A letter or number suffix on a drawing that identifies the version. Rev A is the initial release; subsequent changes increment to Rev B, Rev C, etc. Each new revision may add, modify, or delete characteristics. The AS9102 Form 3 must reflect the current revision's characteristics.

**Characteristic:** Any dimension, GD&T callout, note, or specification on an engineering drawing that can be measured or verified. Each characteristic on a drawing must be accounted for on AS9102 Form 3. Characteristics are identified by balloon number on the drawing.

**Critical Characteristic:** A characteristic whose nonconformance could cause a safety issue or failure to meet a primary functional requirement. Often designated with a special symbol or note on the drawing (e.g., a diamond ◆ or the word "CRITICAL"). Critical characteristics require 100% inspection or process control with documented Cpk.

**Key Characteristic:** Similar to critical characteristic; a characteristic that significantly affects fit, form, function, performance, or service life. Tracked with SPC in AS9100 programs.

**Datum:** A theoretically exact point, axis, or plane from which measurements are made. Datums are established from datum features on the physical part. The datum reference frame (DRF) is established by clamping/fixtuing the part against the primary, secondary, and tertiary datum features in order.

**True Position:** The theoretically exact location of a feature as defined by basic dimensions on the drawing. The position tolerance (a GD&T control using ⊕) specifies the allowable deviation from true position.

**Basic Dimension:** A theoretically exact dimension used to define the true position, true profile, or true orientation of a feature. Basic dimensions are enclosed in a rectangle (box) on the drawing and have no tolerance of their own — the tolerance is given by the associated GD&T callout.

**Title Block:** The area of an engineering drawing (typically lower right corner) containing the drawing number, revision, title, material, finish, tolerances, and other metadata. Critical for parsing drawing identity and revision.

**Notes Block:** The section of a drawing containing general notes that apply to the entire drawing — material callout, finish specification, general tolerances, and special requirements.

**ITAR (International Traffic in Arms Regulations):** US regulations controlling the export of defense-related materials, including most aerospace manufacturing drawings. Production aerospace drawings are restricted under ITAR, which is why publicly available aerospace drawing datasets are extremely limited.

**FAIR (First Article Inspection Report):** See AS9102 FAIR above. The complete package documenting that the first article of a part conforms to all drawing requirements.

**AS9100:** The quality management system standard for the aviation, space, and defense industry. Based on ISO 9001 with additional aerospace-specific requirements. Required for suppliers to major aerospace OEMs (Boeing, Lockheed, Northrop Grumman, etc.).

**AS9102:** The aerospace standard specifically governing First Article Inspection. Defines the required forms, data, and process for conducting and documenting FAIs. The three forms are: Form 1 (design documentation), Form 2 (material and process), Form 3 (characteristic accountability).

**AS9102 Form 3 (Characteristic Accountability):** The record of every characteristic on the drawing — each numbered, with the ballooned drawing number, characteristic description, nominal value, tolerance, measured value, and conformance status. When a drawing is revised, the Form 3 must be updated to reflect added, changed, or deleted characteristics.

**PPAP (Production Part Approval Process):** An automotive-industry quality framework (AIAG) equivalent to FAI for automotive supply chains. Includes dimensional results, material certs, process flow diagrams, control plans, and capability studies. Less common in aerospace but conceptually parallel to AS9102.

**DFM (Design for Manufacturing):** The practice of designing parts so they can be manufactured efficiently and cost-effectively. DFM review catches features that would be difficult or expensive to machine before drawings are released.

**Drawing Revision Reconciliation:** The process of comparing two revisions of an engineering drawing to identify which characteristics were added, changed, or deleted. In AS9102 FAIR workflows, a drawing revision requires the supplier to update the Form 3 to account for all changes. This reconciliation is the core problem that James's software addresses — tracking which inspection characteristics carry over vs. are new or modified across revision changes.

---

## 6. Aerospace Quality Documentation Concepts

### AS9102 FAIR Workflow Overview

1. **Engineering releases drawing** at a given revision (e.g., Rev A)
2. **Supplier creates balloon drawing** — numbers every characteristic on the drawing
3. **Supplier manufactures first article(s)**
4. **Supplier completes FAI package:**
   - Form 1: Design documentation (drawing number, revision, applicable specs)
   - Form 2: Material and process certifications (MTRs, process certs)
   - Form 3: Characteristic accountability — one row per characteristic with measured values
5. **Customer reviews and approves (or rejects) FAI**
6. **Drawing changes (new revision, e.g., Rev B)**
7. **Supplier must reconcile Form 3:** Identify which characteristics changed, which are new, which were deleted, and which are unchanged. Only changed/new characteristics require re-inspection; unchanged ones carry forward from the previous FAI.

### Characteristic Identity Across Revisions

The central challenge in revision reconciliation is establishing **identity** of characteristics across drawing revisions:

- **Unchanged characteristic:** Same dimension/callout at same location with same tolerance. The prior measured value and FAI data carry forward.
- **Modified characteristic:** The nominal value, tolerance, or GD&T control changed. Requires re-measurement.
- **Added characteristic:** A new dimension/callout appears in the new revision. Requires first-time inspection.
- **Deleted characteristic:** A dimension/callout was removed. Removed from Form 3.

The ECO (Engineering Change Order) accompanying a drawing revision should document these changes, but ECOs are frequently incomplete — engineers may change more than what is documented. This "ECO trust problem" means that software reconciling characteristics must compare the drawings directly rather than relying solely on ECO documentation.

### Balloon Numbering and Characteristic Identity

Balloon numbers are assigned by the shop/supplier, not the engineer, and they are not persistent across revisions. A characteristic ballooned as #14 on Rev A may receive a different balloon number on Rev B if characteristics were added or re-organized. Characteristic identity must therefore be established by comparing the content and location of characteristics on the drawing, not by balloon number alone.

### Key Documents in an AS9102 Package

| Document | Produced By | Content |
|----------|------------|---------|
| Engineering Drawing (current rev) | Customer/OEM | Defines all characteristics |
| AS9102 Form 1 | Supplier | Design documentation, drawing ref |
| AS9102 Form 2 | Supplier | Material certs, process certs |
| AS9102 Form 3 | Supplier | Characteristic accountability (all dimensions) |
| Balloon Drawing | Supplier | Drawing with all characteristics numbered |
| CMM Report | Supplier | Measured data for inspected features |
| MTR / Mill Cert | Material supplier | Raw material traceability |
| CoC | Supplier | Conformance declaration |

### Quality System Documentation (AS9100 Context)

- **DHF (Design History File):** Contains design inputs, outputs, verification, and validation records. Machine shop FAI documentation feeds into the DHF.
- **DMR (Device/Drawing Master Record):** The manufacturing specifications and process instructions the shop must follow. Drawing changes require ECO approval — the shop cannot deviate without written authorization.
- **DHR (Device/Drawing History Record):** The as-built record for each production lot — material lot numbers, inspection results, process parameters, operator sign-offs.
- **NCR (Non-Conformance Report):** Triggered when a part or lot does not conform to drawing requirements.
- **CAPA:** Root cause analysis and corrective action for nonconformances.

### Inspection Characteristic Types on Aerospace Drawings

**Dimensional characteristics:**
- Linear dimensions (lengths, depths, widths) with bilateral or unilateral tolerances
- Diameters (OD, ID, bore) with tolerances or ISO fit designations (H7, etc.)
- Angular dimensions

**GD&T characteristics:**
- True position (most common on hole patterns, fastener locations)
- Profile of a surface (complex contours, airfoil shapes)
- Flatness, parallelism, perpendicularity (datum surfaces, mating surfaces)
- Circular runout, total runout (rotating features, shafts)
- Concentricity (coaxial features)

**Surface finish characteristics:**
- Ra callouts on functional surfaces
- Specified by the surface finish symbol (ASME Y14.36)

**Thread characteristics:**
- Thread form, size, pitch, class (e.g., 1/4-20 UNC-2B, Class 3A)
- Thread depth (blind vs. through)

**Material/process characteristics (Form 2 scope):**
- Material alloy and temper (e.g., Al 7075-T6 per AMS-QQ-A-250/12)
- Heat treat specification
- Surface treatment (anodize, passivation, plating) with governing spec

**Special/critical characteristics:**
- Designated with a special symbol on the drawing
- Require 100% inspection or process control with Cpk documentation

# AS9102C Official Guide from SAE International

## Page 1

AEROSPACE
STANDARD

AS9102™
REV. C
Issued
2000-08
Revised
2023-06

Superseding AS9102B
Technically equivalent writings
published in all IAQG sectors
(R) Aerospace Series - First Article Inspection Requirements
RATIONALE
This standard was revised to emphasize and enhance the First Article Inspection (FAI) planning, evaluation, and
re-accomplishment activities; aligning requirements to the 9100 standard. Additional changes to the standard requirements,
definitions, and associated notes were incorporated in response to stakeholder needs.
FOREWORD
To assure customer satisfaction, the aviation, space, and defense industry organizations must produce and continually
improve safe, reliable products that meet or exceed customer and regulatory requirements. The globalization of the industry
and the resulting diversity of regional/national requirements and expectations have complicated this objective. End-product
organizations face the challenge of assuring the quality and integration of products purchased from suppliers throughout
the world and at all levels of the supply chain. Industry suppliers face the challenge of delivering products to multiple
customers having varying quality requirements and expectations.
The aviation, space, and defense industry established the International Aerospace Quality Group (IAQG) for the purpose of
achieving significant improvements in quality, delivery, safety, and reductions in cost throughout the value stream. This
organization includes representation from companies in the Americas, Asia/Pacific, and Europe.
This document standardizes FAI process requirements to the greatest extent possible. While primarily developed for the
aviation, space, and defense industry, this standard can also be used in other industry sectors where a standardized FAI
process is needed.

---

## Page 3

1. SCOPE
1.1
General
This standard establishes the requirements for performing and documenting FAI. It is emphasized the requirements
specified in this standard are complementary (not alternative) to customer and applicable statutory and regulatory
requirements.
1.2
Purpose
The primary purpose of FAI is to verify and validate product realization processes are capable of producing characteristics
that meet engineering and design requirements. A well-planned and executed FAI by a multi-disciplinary team (e.g.,
members from responsible functions) provides objective evidence the manufacturer’s processes can produce compliant
product; having effectively understood and incorporated the associated requirements.
NOTE:  A FAI is not a product acceptance document. While interrelated, FAI and product acceptance are separate activities.
The focus of FAI is verification of production processes via assessment of product. FAI and supporting
documentation does not provide assurance regarding conformance for product acceptance purposes; neither does
the lack of a FAI necessarily imply product is nonconforming to engineering and design requirements.
FAI will:
•
Provide confidence, through objective evidence, the product realization processes are capable of producing conforming
product.
•
Demonstrate the manufacturers and processors of the product have an understanding of the associated requirements.
•
Provide assurance of product conformance at the start of production and after changes, as outlined in this standard.
A FAI is intended to:
•
Mitigate risks associated with production startup and process changes.
•
Reduce future escapes.
•
Help ensure product safety.
•
Improve quality, delivery, and customer satisfaction.
•
Reduce costs and production delays associated with product nonconformances.
•
Identify product realization processes not capable of producing conforming characteristics, and initiate and/or validate
associated corrective actions.
1.3
Application
This standard applies to organizations and their suppliers responsible for product realization processes that produce the
design characteristics of the product. The organization shall flow down the requirements of this standard to suppliers who
produce design characteristics.
This standard also applies to suppliers performing special process(es). A Certificate of Conformance (CoC) provided by
processors attests to satisfying the requirements. External suppliers providing special process(es) can satisfy this standard’s
requirements by either:
•
Documenting the design characteristics and associated results on a First Article Inspection Report (FAIR).
•
Documenting the design characteristics and associated results on a detailed CoC.

---

## Page 4

This standard applies to assemblies, sub-assemblies, and detail parts including castings, forgings, and modifications to
standard catalogue or  Commercial-Off-the-Shelf (COTS) items. Each of these items shall have a separate FAI.
Unless contractually required, this standard does not apply to:
•
Development and prototype parts that are not considered as part of the first production run.
•
Procured standard catalogue item, COTS, or deliverable software. When these items are included in an assembly, they
shall be documented in the index of part numbers in an assembly FAIR.
1.4
Informative
If there is a conflict between the requirements of this standard, and customer or applicable statutory/regulatory requirements,
the latter shall take precedence.
In this standard, the following verbal forms are used:
•
 “Shall” indicates a requirement.
•
 “Should” indicates a recommendation.
•
 “May” indicates a permission.
•
 “Can” indicates a possibility or a capability.
Information marked as “NOTE” is for guidance in understanding or clarifying the associated requirement.
2. APPLICABLE DOCUMENTS
The following referenced documents support the application/use of this standard. For dated references, only the edition
cited applies. For undated references, the latest edition of the referenced document (including any amendments) applies.
When a conflict in requirements between this standard and the referenced documents below exist, the requirements of this
document shall take precedence.
9100
Quality Management Systems - Requirements for Aviation, Space, and Defense Organizations
9103
Aerospace Series - Quality Management Systems - Variation Management of Key Characteristics
As developed under the auspices of the IAQG and published by various standards bodies [e.g., ASD-STAN, SAE
International, European Committee for Standardization (CEN), Japanese Standards Association (JSA)/Society of Japanese
Aerospace Companies (SJAC), Brazilian Association for Technical Norms (ABNT)].
ASME Y14.41 Digital Product Definition Data Practices
ISO 9000
Quality management systems - Fundamentals and vocabulary
ISO 16792
Digital Product Definition Data Practices

---

## Page 5

3. TERMS AND DEFINITIONS
Definitions for general terms can be found in ISO 9000 and the IAQG International Dictionary (located on the IAQG website).
An acronym log for this standard is presented in Appendix A. For the purpose of this standard, the following definitions
apply.
3.1
ASSEMBLY
A product that is produced by joining two or more detail parts, COTS, standard catalogue item, or sub-assemblies into one
item.
3.2
ATTRIBUTE DATA
A result from a characteristic or property that is appraised only as to whether it does or does not conform to a given
requirement (e.g., go/no-go, accept/reject, pass/fail).
3.3
BALLOONED DESIGN CHARACTERISTIC
Clear and uniquely identified design characteristic indicated on a ballooned document. The unique identifier may be circled
or highlighted for easy visual identification.
3.4
BALLOONED DOCUMENT
An aid used in FAI to identify all the design characteristics, including all documents—e.g., drawings, purchase order, Digital
Product Definition (DPD)—typically sequentially numbering the design characteristics and putting a circle around or
highlighting the numbered design characteristics.
3.5
BASELINE PART NUMBER
This refers to a part number from the previous FAI or approved configuration, including revision level, to which a partial FAI
is performed. An example of an approved configuration is a part produced and verified as conforming product prior to the
requirements of this standard.
3.6
COMMERCIAL-OFF-THE-SHELF (COTS) ITEM
Commercially available item intended by design to be procured and utilized without modification (e.g., common electronic
components). Any item or assembly meeting all of the following requirements:
a. Defined by industry, manufacturer, military, or recognized specifications or standards.
b. Without design modification, specifically for a customer.
c. Customarily used by the public or industries.
d. Offered for sale to the public, through catalogues, price list, brochures, stores, or websites.
3.7
DELIVERABLE SOFTWARE
Embedded or loadable airborne, spaceborne, or ground support software or firmware components which are part of an
aircraft type design, weapon system, missile, or spacecraft.

---

## Page 6

3.8
DESIGN CHARACTERISTIC
Dimensional, visual, functional, mechanical, and material features or properties, which describe and constitute the design
of the product. These characteristics can be measured, inspected, tested, or verified to determine conformance to the design
requirements as specified on the parts list, purchasing document, drawing, or DPD, to which the product is to be produced.
•
Dimensional design characteristics include in-process locating features (e.g., additive manufacturing, target-machined
or forged/cast dimensions on forgings and castings, weld/braze joint preparation necessary for acceptance of finished
joint).
•
Material design characteristics include processing output variable (e.g., plating or coating thickness/runout, material
hardness/conductivity). These provide assurance of intended characteristics that could not be otherwise defined.
3.9
DESIGNED TOOLING
Product specific tooling [e.g., check fixtures, Coordinate Measurement Machine (CMM) program] specifically made to
validate the design characteristics of a product.
3.10 DETAIL PART
Article/part produced to engineering definition that does not include assembly processes (i.e., processes that join two or
more parts together). Detail parts may include processing, finishes, and/or special process(es).
3.11 DIGITAL PRODUCT DEFINITION (DPD)
Digital data file(s) that disclose, directly or by reference, the physical or functional requirements, including data files that
disclose the design or acceptance criteria of a product. Examples of DPD include the following:
•
Digital data file(s) and fully dimensioned two-dimensional (2D) drawing sheets.
•
Three-dimensional (3D) data model, and simplified or reduced content 2D drawing sheets.
•
3D data model with design characteristics displayed as text.
•
Any other data files containing design characteristics that define a product in its entirety.
3.12 FIRST ARTICLE INSPECTION (FAI)
A planned, complete, independent, and documented inspection and verification process to ensure that prescribed
production processes have produced an item conforming to engineering drawings, DPD, planning, purchase order,
engineering specifications, and/or other applicable design documents.
NOTE:  The intent of independent as referenced above is to mitigate the effect of measurement error. This includes ensuring
the person that verifies the characteristic for the first article not be the same person that generated the characteristic.
Self-inspection (i.e., operator self-verification) is not considered independent. The equipment used to verify the
characteristic should be different from the equipment used to produce the characteristic.
3.13 FIRST ARTICLE INSPECTION REPORT (FAIR)
Comprised of the forms identified in Appendix B, all ballooned design characteristics, and the supporting documentation
determined by FAI planning for a part number (e.g., detail part, sub-assembly, or assembly).
3.14 FIRST PRODUCTION RUN
The initial group of one or more parts that are the result of a planned process designed to be used for production of these
same parts.

---

## Page 7

3.15 MODIFIED COMMERCIAL-OFF-THE-SHELF (COTS)/STANDARD CATALOGUE ITEM
A COTS or Standard Catalogue item that has a change made to it from its original designed configuration.
NOTE:  Once modified, these items are categorized as detail parts for the purpose of assembly.
3.16 MULTIPLE CHARACTERISTICS
Identical characteristics that occur at more than one location (e.g., four places), but are identified by a single set of drawing
or DPD requirements (e.g., rivet hole size, dovetail slots, corner radii, chemical milling pocket thickness).
3.17 PRODUCT
Any intended output resulting from the product realization process, which in the context of this standard includes finished
detail parts, sub-assemblies, assemblies, forgings, and castings.
3.18 QUALIFIED TOOLING
Universal (not part specific) calibrated monitoring and measuring equipment (e.g., go/no go gauges, thread gauges, radius
gauges) used to validate product design characteristics using attribute data.
3.19 REFERENCE CHARACTERISTIC
Characteristic (including reference and basic dimensions) that are used for “information only” or to show relationship; these
are dimensions without tolerances and refer to other dimensions on the drawing or in the DPD.
3.20 SPECIAL PROCESS
Any process for production and service provision where the resulting output cannot be verified by subsequent monitoring
or measurement and, as a consequence, deficiencies become apparent only after the product is in use or the service has
been delivered.
3.21 STANDARD CATALOGUE ITEM
A part or material that conforms to an established industry or national authority published specification, having all
characteristics identified by written description or an industry/national/military standard drawing.
3.22 VARIABLE DATA
Quantitative measurements taken on a continuous scale (e.g., the diameter of a cylinder, the gap between mating parts).

---

## Page 8

4. REQUIREMENTS
4.1
First Article Inspection Planning
a. The organization shall have a documented process to plan for FAI. This process shall identify the responsible functions
and address the activities to be performed, prior to the first production run.
b. The organization shall verify the revision for embedded or deliverable software as defined by the Bill of Materials (BOM),
drawing/DPD, specification, or purchase order requirements.
c. The organization shall consider the following activities, during FAI planning and, if required by contract, coordinate
planning with the customer:
1. Determine design characteristic inspection and sequencing for inspection of characteristics not measurable in the
final product and provisions to carry out those activities at the appropriate stage of the manufacturing process.
2. Evaluate DPD design characteristics required for product realization which are not fully defined on 2D drawings,
including tolerances for nominal dimensions.
3. Determine the required objective evidence to be included in the FAIR for each design characteristic, including
supporting documentation.
NOTE:  This includes bubbled or ballooned document(s), and may also include certifications, inspection reports, test
reports, manufacturing plans, purchase orders, etc.
4. Identify the approved special process, laboratory, material, and customer required sources, as applicable, and
confirm the manufacturing planning, routing, and purchase document identify the correct specification and relevant
sources.
5. Identify key characteristic and critical item requirements, as applicable (refer to IAQG standards 9100 and 9103 for
supporting guidance/direction).
6. Determine suitable monitoring and measuring equipment of appropriate resolution and accuracy. Ensure part
specific gauges and tooling are identified, qualified, and traceable.
NOTE:  Metrology principles (e.g., accuracy ratio, measurement uncertainty) should be taken into consideration when
selecting a measurement method.
7. Coordination of customer FAI review(s) at any stage.
8. Identify events requiring an updated FAI (see 4.6).
d. The organization shall verify FAI planning activities have been completed.
4.2
Part Requirements
a. The organization shall perform a FAI on new product representative of the first production run. The first production
product delivery requires a FAI.
b. The organization shall use one or more representative items from the first production run of a new product to verify that
the production processes, production documentation, and tooling have the ability to produce products that meet
established requirements.
c. For assemblies, the assembly level FAI shall be performed on those characteristics specified on the assembly drawing
or DPD.
d. Detail part characteristics created or modified during assembly may be accounted for at the assembly level FAI, all other
detail part characteristics shall be accounted for on the detail part FAI.

---

## Page 9

4.3
Digital Product Definition Requirements
a. When design requirements are in a DPD format and traditional 2D drawing information is not available for all applicable
design requirements, DPD design characteristics required for product realization shall be extracted, verified, and
included in the FAIR.
b. The organization shall:
1. Establish a process to extract the applicable DPD design characteristics.
2. Extract the DPD design characteristics required for product realization.
3. Ensure the production, inspection, and operations requiring verification have been completed as planned to achieve
DPD design characteristics.
NOTE: For additional information on DPD, refer to ASME Y14.41 and/or ISO 16792.
4.4
Evaluation Activities
The organization shall conduct the following activities during product realization in support of FAI to ensure conformance
with design characteristics:
a. Review the manufacturing process documentation (e.g., routing sheets, risk analysis, manufacturing or quality plans,
manufacturing work instructions) to ensure all operations are complete as planned and call out the correct specification,
material types, conditions, and approvals.
b. Review supporting documentation for completeness.
c. Verify the raw material and special process certifications (e.g., CofC, special process completion certification, raw
material test report number, modified standard catalogue item compliance report number, traceability number) call out
the correct specification, material types, conditions, and approvals.
d. Verify that required customer approved sources are utilized (e.g., directed source, approved suppliers list).
e. Review nonconformance documentation for completeness.
f.
Verify that required designed tooling (e.g., part specific gauges) is used.
g. Verify that every design characteristic requirement, including DPD characteristics as required per 4.3.b, is accounted
for, uniquely identified, and has inspection results traceable to each unique identifier (e.g., ballooned design
characteristic).
h. Verify the design characteristics resulting from the output of the manufacturing process are measured, inspected, tested,
or verified to determine conformance, including DPD characteristics as required per 4.3.b.
i.
Verify part marking has met defined requirements, such as legibility (i.e., human/machine readable), method, material,
content, size, and location.

---

## Page 10

4.5
Nonconformance Handling
a. When processing a FAIR with documented nonconformances the organization shall:
1. Record the nonconforming design characteristics on Form 3, “Characteristic Accountability, Verification, and
Compatibility Evaluation.”
2. Record the nonconformance document reference number on Form 3 (see field 11).
3. Check the “Yes” box on Form 1 [see field 19 - “Does FAIR Contain a Documented Nonconformance(s)”].
NOTE: This standard does not address disposition of the nonconformance.
b. The organization shall implement corrective action(s) to correct the product realization process until it delivers the
intended output (conforming design characteristics). This process may be subject to multiple iterations and needs to be
managed through the organization’s quality management system within the context of the corrective action process.
c. Upon implemented corrective action, the organization shall conduct a partial/full FAI, and at a minimum document the
corrected nonconforming characteristics and any other characteristics affected by the corrective action.
d. Once all nonconformances have been corrected, check the “No” box on Form 1 [see field 19 - “Does FAIR Contain a
Documented Nonconformance(s)”].
NOTE:  A full FAI may be performed in lieu of a partial FAI (see 4.6).
4.6
Partial or Re-Accomplishment of First Article Inspection
a. The original FAI requirement shall continue to apply after initial compliance.
b. The FAI shall be repeated when changes occur that invalidate or are not represented in the original results, as
determined by a multi-disciplinary team (e.g., members from responsible functions).
c. The FAI requirements shall be satisfied by a FAI that addresses the changes from a baseline part number provided all
other characteristics were conforming on the previous FAI and are produced by the original production processes.
NOTE 1:  This is referred to as a partial FAI.
NOTE 2:  A full FAI may be completed in place of a partial FAI.
d. When performing a partial FAI, the organization shall complete the affected fields in the FAI forms.
e. When performing a partial FAI, the organization shall record the “Baseline Part Number,” including the revision level
and reason for the partial FAI on Form 1 (see field 14).
NOTE 1: If a nonconformance is detected during FAI, the design characteristics not affected by the nonconformance
are still valid, regardless of the product nonconformance disposition (e.g., scrap).
NOTE 2: FAI requirements on a previously approved FAI performed on identical characteristics of similar parts
produced by identical means are valid. FAI requirements may be satisfied in this manner. For similar parts
made using the same processes (e.g., identical means) except for a few characteristics, a complete FAI can
be done on one part and for the similar parts account for only the unique characteristics. On Form 3 for the
similar parts, record the unique characteristics. This provides objective evidence and traceability for all
applicable design characteristics.

---

## Page 11

f.
The organization shall have a documented process to evaluate any changes to product realization processes or
engineering/design requirements (see supporting sub-sections 1-6) that invalidate or are not represented in the previous
FAI and then perform a full or partial FAI, as determined by the evaluation. The organization shall perform the evaluation
when any of the following occurs:
1. A change in engineering definition affecting design characteristics.
2. A change in manufacturing source(s), process(es), inspection method(s), tooling, materials/alternate materials, or
location of manufacture.
3. A change in the numerical control program or translation to another media.
4. A natural or man-made event, which can adversely affect the manufacturing process.
5. An implementation of corrective action required to complete a previous FAI, as defined in 4.5.
6. A lapse in production for two years for any characteristics that may be impacted. This lapse is from the completion
of the last production operation to the actual restart of production.
4.7
Documentation
4.7.1
Forms
a. Appendix B contains forms that comply with the documentation requirements of this standard. Forms other than those
depicted in Appendix B may be used; however, they shall contain all “Required” and “Conditionally Required” information
and have the same field reference numbers.
1. (R) - Required: This is mandatory information.
2. (CR) - Conditionally Required: This field shall be completed, when applicable to the product (e.g., serial number
shall be entered when the product has an associated serial number). When not applicable may be left blank.
3. (O) - Optional: This field is provided for convenience; the field may be left blank.
NOTE: Continuation sheets and insertion of additional rows are acceptable.
b. All forms shall be completed either electronically or in permanent ink.
c. All forms shall be completed in English or in a language specified by the customer.
4.7.2
Characteristic Accountability
a. The organization shall verify every design characteristic, during the FAI. Every design characteristic shall have its own
unique characteristic number.
b. Reference characteristics may be omitted from the FAI.
c. More than one line may be used, if needed, for any characteristic.
d. Characteristics not measurable in the final product shall be verified during the manufacturing process, as long as they
are not affected by subsequent operations or by destructive means.
e. Characteristics verified on detail parts may be referenced in the assembly-level FAIR.

---

## Page 12

4.7.3
Recording Results
a. The organization shall record the requirements and results in the primary units (e.g., metric, imperial systems) as
specified on the drawing or DPD, unless otherwise approved by the customer.
b. Results from inspection of design characteristics shall be expressed in quantitative terms (i.e., variable data), to the
level of accuracy including tolerance of measurement (i.e., number of decimal places) when a design characteristic is
expressed by numerical limits.
1. Attribute data (e.g., pass/fail) may be used in lieu of variable data when no inspection technique resulting in variable
data is feasible. Designed tooling or qualified tooling is consistently used as a check feature and a go/no-go feature
has been established for the specific characteristic. When qualified tooling (e.g., radius gauges, comparator, mylar,
loft dimensions) are used as a go/no-go gauge, record the gauge value or range (e.g., minimum/maximum value),
as applicable.
c. Attribute data shall be used, when the design characteristic does not specify numerical limits (e.g., break all sharp
edges).
4.8
Retained Documented Information
a. FAI documentation required by this standard shall be considered a quality record. The organization shall retain the FAIR
while the product is being produced and, at a minimum, retain it according to applicable customer or regulatory
requirements; whichever is longer.
b. The reviewed and verified quality records shall be retained in accordance with the producer’s record retention
requirements. The recording of the verification in the FAIR provides evidence of compliance for those records and
artifacts.
5. NOTES
5.1
Revision Indicator
A change bar (I) located in the left margin is for the convenience of the user in locating areas where technical revisions, not
editorial changes, have been made to the previous issue of this document. An (R) symbol to the left of the document title
indicates a complete revision of the document, including technical revisions. Change bars and (R) are not used in original
publications, nor in documents that contain editorial changes only.
PREPARED BY THE G-14 AMERICAS AEROSPACE QUALITY STANDARDS COMMITTEE (AAQSC)

---

## Page 13

APPENDIX A - ACRONYM LOG
2D
Two-Dimensional
3D
Three-Dimensional
ABNT
Brazilian Association for Technical Norms
ASD-STAN
Aero Space and Defense Industries Association of Europe - Standardization
BOM
Bill of Materials
CEN
European Committee for Standardization
CMM
Coordinate Measurement Machine
CMS
Coordinate Measurement System
CoC
Certificate of Conformance (also known as Certificate of Conformity)
COTS
Commercial-off-the-Shelf
DPD
Digital Product Definition
FAI
First Article Inspection
FAIR
First Article Inspection Report
IAQG
International Aerospace Quality Group
JSA
Japanese Standards Association
SJAC
Society of Japanese Aerospace Companies

---

## Page 14

APPENDIX B - 9102 FORMS AND SUPPORTING FORM INSTRUCTIONS
FORM 1 - PART NUMBER ACCOUNTABILITY
FORM 2 - PRODUCT ACCOUNTABILITY - MATERIALS, SPECIAL PROCESSES, AND FUNCTIONAL TESTING
FORM 3 - CHARACTERISTIC ACCOUNTABILITY, VERIFICATION, AND COMPATIBILITY EVALUATION
This appendix provides the instructions to complete the associated 9102 forms. Each input field is identified as:
•
(R) Required - This is mandatory information.
NOTE: These fields are depicted in bold font.
•
(CR) Conditionally Required - This field shall be completed when applicable to the product (e.g., serial number shall
be entered when there is a serial number). When not applicable, may be left blank.
NOTE: These fields are depicted in bold italic font.
•
(O) Optional - This field is provided for convenience; the field may be left blank.
NOTE:  These fields are depicted in standard font.

---

## Page 15

B.1
FORM 1 - PART NUMBER ACCOUNTABILITY

Sheet ___ of ___
1.  Part Number:

2.  Part Name:

3.  Serial Number:

4.  FAIR Identifier:

5.  Part Revision Level:

6.  Drawing Number:

7.  Drawing Revision Level:

8.  Additional Changes:

9.  Manufacturing Process
Reference:

10.  Organization Name:

11.  Supplier Code:

12.  Purchase Order Number:

13.  Detail:

Assembly:

14.  Full FAI                      Partial FAI:
 Baseline Part Number (including revision level):
       Reason for Full / Partial FAI:
a)  If the part number above is a detail part only, go to field 19.
b)  If the part number above is an assembly, go to the "INDEX" section below.
INDEX of part numbers or sub-assembly numbers required to make the assembly noted above.
15.  Part Number:
16.  Part Name:
17.  Part Type:
18.  FAIR Identifier:
19.  Does FAIR Contain a Documented Nonconformance(s)?        Yes               No
20.  FAIR Verified By:
21.  Date:
22.  FAIR Reviewed/Approved By:
23.  Date:
24.  Customer Approval:
25.  Date:
26.  Comments:

---

## Page 16

B.2
FORM 1 - PART NUMBER ACCOUNTABILITY FORM INSTRUCTIONS
This form is used to identify the product that is having the First Article Inspection (FAI) conducted on (e.g., detail part, sub-
assembly, assembly); referred to as “FAI part.”
NOTE:  Data fields 1 thru 4 are repeated on all forms for convenience and traceability. Any subsequent changes to “data
fields” 1 thru 4 need to be made to all pages.
1. (R)
Part Number: Number of the FAI part [e.g., customer part number contained on the purchasing documents;
part number from the associated Bill of Materials (BOM); manufacturer part number for internal parts, when
customer part number is not available].
2. (R)
Part Name: Name of the FAI part.
3. (CR)
Serial Number: Serial number of the FAI part; unique identifier assigned to a detail part, sub-assembly, or
assembly by the organization or customer.
4. (R)
FAIR Identifier: Identifier for the First Article Inspection Report (FAIR).
5. (CR)
Part Revision Level: The revision level of the FAI part being inspected. When the part is controlled by a part
revision and the part has not been revised, indicate as such (e.g., N/C, No Change).
NOTE 1: The latest drawing or DPD revision (see field 7) does not always affect all parts contained on a
drawing or DPD.
NOTE 2:  This is the revision level that is identified on the part. Not all organizations use a part revision level
for tracking configuration.
6. (CR)
Drawing Number: Drawing and/or DPD number associated with the FAI part; drawing may be from customer,
internal system, or design definition.
NOTE:  This field identifies all the drawings (including parts list), that contain design characteristics needed for
product realization. There may be more than one drawing listed in this field.
7. (CR)
Drawing Revision Level: The revision level of the drawing or DPD associated with the FAI part. If the drawing
has not been revised, indicate as such (e.g., N/C, No Change).
NOTE:  This field identifies the revision levels of the drawings or DPD sets listed in field 6. When there is more
than one entry in field 6, the entries in this field need to correspond to the entries presented in field 6.
8. (CR)
Additional Changes: Provide reference numbers of any changes that are incorporated in the product, but not
reflected in referenced drawing/part revision level (e.g., change in design, engineering changes, manufacturing
changes, deviation or exclusion from certain drawing or DPD requirements).
9. (R)
Manufacturing Process Reference: Reference number that provides traceability to the manufacturing record
of the FAI part (e.g., router number, manufacturing plan number). Additional information such as lot number,
batch number, date code, revision level, or line number may be included, as needed, to provide traceability to
the specific manufacturing lot.
10. (R)
Organization Name: Name of the organization responsible for producing the design characteristics of the
product and performing the FAI.
11. (O)
Supplier Code: A unique number given by customer to the organization; sometimes referred to as a Vendor
Code, Vendor Identification Number, or Supplier Number.
12. (O)
Purchase Order Number: Customer purchase order number, if applicable.
13. (R)
Detail/Assembly: Type of FAI; check, as appropriate.

---

## Page 17

14. (R)
Full FAI/Partial FAI: Check the appropriate box (Full FAI or Partial FAI).
For a partial FAI, provide the previous part number, including revision level. For partial FAIs based on similar
parts (see 4.6), provide the approved configuration or FAI part number, including revision level.
Baseline Part Number (including revision level): For a partial FAI, provide the previous FAI part number or
approved configuration (including revision level).
Reason for Full/Partial FAI: Describe the reason [e.g., new part number; lapse in production; changes in
design, process, or manufacturing location (see 4.6)] for the full or partial FAI.
Data Fields 15, 16, 17, and 18: This section is only required if the part number identified in field 1 is an assembly. All BOM
parts (e.g., detail parts, sub-assemblies, COTS) that are part of the assembly, identified in field 1, shall be listed in this
section.
15. (CR)
Part Number: Part number included in the assembly and items from the engineering and/or manufacturing
BOM included in the drawing, DPD, or next level assembly. Typically, these are the part numbers, standard
catalogue item numbers, deliverable or embedded software identification, or sub-assembly numbers required
to complete the product noted in field 1.
NOTE 1: Include revision level for software listed on the BOM.
NOTE 2: Materials and processes listed on Form 2 do not need to be restated on Form 1.
16. (CR)
Part Name: Name or description of the part number entered in field 15 that is installed in the assembly.
17. (CR)
Part Type: Enter whether the part is a detail part, sub-assembly, software, standard catalogue item, or COTS
(or equivalent).
18. (CR)
FAIR Identifier: FAIR identifier (e.g., software generated FAIR identification or number, part number, individual
organizational FAIR identification naming conventions) for the detail parts and associated assemblies. If no
FAIR identifier is available, input the organization’s identifier for the FAI or approved configuration.
19. (R)
Does FAIR Contain a Documented Nonconformance(s)?: When a nonconformance(s) has been
documented in the FAIR, check “Yes” (see 4.5).
20. (R)
FAIR Verified By: Legible identification of the person verifying the evaluation activities in 4.4 were completed.
NOTE:  Electronic identification is acceptable.
21. (R)
Date: Date when field 20 was populated.
22. (R)
FAIR Reviewed/Approved By: Legible identification of the person from the organization who reviewed and
approved the FAIR. Should not be the same individual identified in field 20.
NOTE:  Electronic identification is acceptable.
23. (R)
Date: Date when field 22 was populated.
24. (CR)
Customer Approval: Used by customer to record approval.
NOTE:  Electronic identification is acceptable.
25. (CR)
Date: Date when field 24 was populated.
26. (O)
Comments: Provide any supporting comments (e.g., associated nonconformance information, identification of
associated documentation).

---

## Page 18

B.3
FORM 2 - PRODUCT ACCOUNTABILITY - MATERIALS, SPECIAL PROCESSES, AND FUNCTIONAL TESTING

Sheet ___ of ___
1.  Part Number:
2.  Part Name:
3.  Serial Number:
4.  FAIR Identifier:
5.  Material or Process
Name:
6.  Specification
Number:
7.  Code:
8.  Supplier:
9.  Customer
Approval
Verification:
10.  Certificate of
Conformance
Number:
11.  Functional Test Procedure Number:
12.  Acceptance Report Number:
13.  Comments
B.4
FORM 2 - PRODUCT ACCOUNTABILITY - MATERIALS, SPECIAL PROCESSES, AND FUNCTIONAL TESTING
FORM INSTRUCTIONS
This form is used if any materials, special processes, or functional testing are defined as a design characteristic.
NOTE:  Data fields 1 thru 4 are repeated on all forms for convenience and traceability. Any subsequent changes to “data
fields” 1 thru 4 need to be made to all pages.

---

## Page 19

1. (R)
Part Number: Number of the FAI part [e.g., customer part number contained on the purchasing documents;
part number from the associated Bill of Materials (BOM); manufacturer part number for internal parts, when
customer part number is not available].
2. (R)
Part Name: Name of the FAI part.
3. (CR)
Serial Number: Serial number of the FAI part; unique identifier assigned to a detail part, sub-assembly, or
assembly by the organization or customer.
4. (R)
FAIR Identifier: Identifier or identification number for the First Article Inspection Report (FAIR).
5. (CR)
Material or Process Name: Name of materials (e.g., raw materials, paint, primer adhesives, weld filler) or
special processes.
6. (CR)
Specification Number: Provide the following information:
•
Material specifications and material form (e.g., sheet, bar) for all materials incorporated into the FAI part
(e.g., weld, braze filler).

•
Special process specifications; including class, if applicable, and permitted substitutions.

•
If Commercial-Off-the-Shelf (COTS)/standard catalogue items are modified, then list the non-modified
standard hardware or COTS item part number.

NOTE:  Non-modified standard catalogue item(s), when part of an assembly, are listed on Form 1, “Part
Number Accountability.”
7. (O)
Code: Any code specified for the material or process.
8. (CR)
Supplier: Identify organization (internal or external) performing special process(es) or supplying material.
•
Name.
•
Address.
•
Code (when available).
9. (CR)
Customer Approval Verification: Indicate if the special process(es) or material sources are approved by the
customer. Enter “Yes” if approved; “No” if approval is required, but process source is not approved; or “NA” if
customer approval is not required.
NOTE: A “No” would be handled in accordance with 4.5.
10. (CR)
Certificate of Conformance Number: The applicable certificate number (e.g., special process completion
certification, raw material test report number, modified standard catalogue item compliance report number,
traceability number).
11. (CR)
Functional Test Procedure Number: Functional Test Procedure number identified as a design characteristic.
12. (CR)
Acceptance Report Number: The functional test certification indicating that test requirements have been met.
 NOTE: When software is uploaded as part of a test procedure, record the software and revision level and
acceptance report number.
13. (O)
Comments: Provide supporting comments, as applicable.

---

## Page 20

B.5
FORM 3 - CHARACTERISTIC ACCOUNTABILITY, VERIFICATION, AND COMPATIBILITY EVALUATION

| 1. Part Number | 2. Part Name | 3. Serial Number | 4. FAIR Identifier |
|---|---|---|---|
|  |  |  |  |

| 5. Char. No. | 6. Reference Location | 7. Characteristic Designator | 8. Requirement | 9. Results | 10. Designed/Qualified Tooling | 11. Nonconformance Number | 12. Additional Data / Comments |
|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |

---

## Page 21

B.6
FORM 3 - CHARACTERISTIC ACCOUNTABILITY, VERIFICATION, AND COMPATIBILITY EVALUATION FORM
INSTRUCTIONS
This form is used to record inspection results for the design characteristics and to document any applicable
nonconformances (see 4.5).
NOTE:  Data fields 1 thru 4 are repeated on all forms for convenience and traceability. Any subsequent changes to “data
fields” 1 thru 4 need to be made to all pages.
1. (R)
Part Number: Number of the FAI part [e.g., customer part number contained on the purchasing documents;
part number from the associated Bill of Materials (BOM); manufacturer part number for internal parts, when
customer part number is not available].
2. (R)
Part Name: Name of the FAI part.
3. (CR)
Serial Number: Serial number of the FAI part; unique identifier assigned to a detail part, sub-assembly, or
assembly by the organization or customer.
4. (R)
FAIR Identifier: Identifier or identification number for the First Article Inspection Report (FAIR).
5. (R)
Char. No.: Unique assigned number for each design characteristic.
•
The ballooned design characteristic shall clearly be traceable to the characteristic number listed in field 5.
•
Automated inspection methods/tooling measurement report/results, shall all be clearly linked to the
characteristic number in field 5, ballooned drawing, and associated measurement report/results.
NOTE:  A single design callout that applies to multiple characteristics (see 3.16) may be recorded as one
characteristic.
6. (CR)
Reference Location: Location of the design characteristic [e.g., drawing zone (page number and section),
Digital Product Definition (DPD) model location callout].
7. (CR)
Characteristic Designator: As applicable, a unique identification for special requirements [e.g., Key
Characteristic (KC), Critical Item (CI), items requiring additional design or process control] defined by customer
(reference 9100 and 9103).
NOTE: See 4.1.c.5.
8. (R)
Requirement: Specified requirement for the design characteristic (e.g., drawing or DPD dimensional
characteristic with associated nominal dimension and tolerances, drawing notes, requirements).
•
The organization shall record the requirements in the units (e.g., metric, imperial systems) specified on the
drawing or DPD, unless otherwise approved by the customer (see 4.7.3.a).
•
The organization shall record the software revision for embedded or deliverable software.
9. (R)
Results: List measurement(s) obtained for the design characteristics.
The organization shall record the results in the units (e.g., metric, imperial systems) specified on the drawing,
DPD, unless otherwise approved by the customer (see 4.7.3.a).
•
For multiple characteristics list each characteristic as individual values or list once with the minimum and
maximum of measured values attained. If a characteristic is found to be nonconforming, then that
characteristic shall be listed separately with the measured value noted.
•
When qualified tooling (e.g., radius gauges) is used as a go/no-go gauge (see4.7.3.b), record the results
as an attribute (e.g., pass/fail).

---

## Page 22

•
When automated inspection equipment produces measurement results, those results may be referenced
on Form 3 identified as pass/fail and attached only when:
−
The characteristic numbers are clearly linked in the attached report [e.g., characteristic identification on
Coordinate Measurement System (CMS) report is the same as on this form].
−
The results in the attached reports are clearly traceable to the characteristic numbers.
−
The results are directly comparable to the design characteristic.
•
A CMS report only depicting deviation from nominal in multiple axes is not acceptable; the report shall
reflect an actual geometric value.
•
If a design requirement requires verification testing, record the actual results on the form. If a laboratory
report or certificate of test is included in the FAIR, the results may be recorded as an attribute (e.g., pass/fail)
and the test reference number recorded on the form. The laboratory report or certificate of test shall show
specific values for requirements and actual results.
•
For characteristics with visual verification requirements that are rated against standard photographs/master
samples/standards; list the unique identifier of the closest comparison. A statement of conformance is
acceptable; record the reference number on the forms.
•
For processes that require verification per design characteristics, include a statement of conformance (e.g.,
certification of conformance, verification indicator - accept).
•
For characteristics verified by attribute inspection, include statement of conformance (e.g., accept).
10. (CR)
Designed/Qualified Tooling: When design tooling or specially designed tooling, including Numerically
Controlled (NC) programming as a media of inspection, is used for attribute acceptance of the characteristic;
record the tool identification number. When qualified tooling is used for attribute acceptance, record the gauge
value or range (e.g., minimum/maximum value), as applicable.
11. (CR)
Nonconformance Number: If the characteristic is found to be nonconforming, record a nonconformance
document reference number.
12. (O)
Additional Data/Comments: This area is reserved for optional fields; add additional columns, as required, by
the organization or customer.
