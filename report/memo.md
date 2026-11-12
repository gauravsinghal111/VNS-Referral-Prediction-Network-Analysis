# Memo: Prioritising Texas Neurologists for VNS Therapy Outreach

**To:** Commercial Strategy & Analytics
**Re:** Which neurologists are most likely to send patients toward epilepsy surgery and VNS evaluation
**Basis:** Public, aggregated CMS Medicare data, Texas, 2013–2015. Portfolio analysis, not a commercial deployment.

---

## The short version

1. **A ranked list of 666 Texas neurologists is ready.** Each one has a score, the three reasons for it,
   and a flag showing whether they are still eligible to refer in Medicare.
2. **The list works.** We tested it the way it would be used: build the list from one year's data,
   then check the following year. Of the top 100 neurologists, **66** went on to share patients with an
   epileptologist, neurosurgeon or epilepsy unit. Ranking by patient volume alone found 34. A random
   100 would find about 14.
3. **Most of that value comes from one fact:** neurologists who already send patients down this
   pathway keep doing so. A simple "already refers" rule found 68 of 100, as good as the model. The
   model's added value is the reasons it gives for each score, plus a modest edge in finding new referrers.
4. **New referrers are rare and hard to predict.** Of 538 neurologists with no current tie to the
   pathway, only 15 formed one the next year. The model found 6 of them in its top 50, compared with 4
   for volume ranking. That is promising but not yet proven.

## What we measured

- **Who:** Texas neurologists (including subspecialties) active in Medicare patient-sharing data,
  about 700 per year.
- **Outcome:** a Medicare **patient-sharing tie** to an epileptologist, neurosurgeon or hospital
  epilepsy unit in the following year. A tie means the neurologist saw a patient and the specialist saw
  the same patient within 30 days, for at least 11 patients. **This is a proxy for referral, not a
  recorded referral.**
- **Signals used:**
  - how many patients the doctor shares and with how many doctors
  - whether they already send patients to this pathway
  - their position in the neurology network (influence and "bridge" roles)
  - how many pathway specialists practise nearby
  - specialty, years in practice, and metro vs non-metro
- **Fair test:** the model learned from 2013→2014 and was graded on 2014→2015, so it never saw the
  answer year.

| Approach | Future referrers in top 100 | Overall ranking quality (ROC-AUC) |
|---|---|---|
| Random list | ~14 | 0.50 |
| Rank by patient volume | 34 | 0.79 |
| "Already refers" rule | 68 | 0.92 |
| Logistic regression model | 65 | 0.93 |
| XGBoost model | 66 | 0.92 |

## What drives a high score

1. **Patients sent onward to other doctors.** Neurologists who actively route patients onward are the ones who reach the surgical pathway.
2. **Patients already sent to epilepsy/neurosurgery specialists.** Past behaviour is the best predictor of future behaviour.
3. **An existing tie to a pathway specialist.** One relationship tends to persist.

**Where the model is wrong.** The neurologists it over-rates look like real referrers on volume, but they
practise in very dense markets: a median of 81 pathway specialists in their ZIP area, against 43 for the
ones it gets right. In places like the Texas Medical Center, patients have many routes and a single
relationship matters less. **Only 8% of Houston neurologists had a pathway tie the following year, against
26% outside the big metros.**

## Recommendations for the commercial team

1. **Protect current referrers first.** 79% of the top-100 list already shares patients with the
   pathway. These relationships are the most reliable source of next year's referrals, and they are
   where clinical education on VNS candidacy has the clearest path to an implant.
2. **Treat new-referrer growth as a pilot, not a forecast.** Take the top 50 "new opportunity"
   neurologists (use the dashboard filter). Visit half of them, hold the other half back, and compare
   referral rates after 6–12 months. That turns a correlation into evidence of what outreach actually
   changes.
3. **Look outside Houston.** Non-metro neurologists were three times as likely as Houston neurologists
   to have a pathway tie the following year. Field coverage and CORE-VNS education could be weighted toward them.
4. **Recalibrate before using scores as probabilities.** On the test year the scores ran high, because
   the overall rate fell from 24% to 14%. Use them to **rank**, not as literal percentages.

## Limitations: read before acting

- **Proxy, not referral.** Sharing a patient within 30 days is not a written referral order.
- **Medicare only.** Many VNS candidates are children and younger adults, who are under-represented
  here. Only 7 child neurologists appear in the 2015 data.
- **Data age.** 2015 is the latest free public year, and its file is about 37% smaller than 2014's.
  Specialty and address come from today's NPI registry and are self-reported.
- **Correlation, not influence.** A high score says *where* to look; it does not prove a visit causes a referral.

## What would make this production-ready

- Replace the public Medicare data with LivaNova's internal sources: CRM visit history, implant and
  referral records, and licensed all-payer claims. That would cover pediatric patients and current years.
- Add outcome tracking from the pilot in recommendation 2, so the model learns which visits made a difference.
- Refresh quarterly and recalibrate on the latest period before each sales cycle.

*Full methods, code and tables: README.md, report/metrics.md, report/error_analysis.md. Interactive
list: `streamlit run app.py`.*
