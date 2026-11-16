"""The event logs of the censoring study and, for each, the forecasting task and the experiment grid.

Everything here was fixed from the logs' own shape (describe_log.py: activities, time span, how long outcomes take)
before any model was trained on them. BPI Challenge 2012 keeps the settings of the earlier pilot.

  outcomes      activity -> outcome name. The first such event of a case is its outcome; a case with none stays open.
  lifecycle     activities whose name starts with this get their lifecycle state appended (start, complete, ...).
  amount        where the one numeric case feature comes from: ('case', key), ('first_event', key) or None.
  deadlines     days: "has an outcome happened within d days of now?"
  history_days  lengths of the training window: cases that started in the last so-many days before the cutoff.
  test_days     test cases are those that start within this many days after the cutoff.
  cutoffs       the dates that stand in for "today". They leave room for the longest history before them and for the
                test window plus the longest deadline after them.

Licences: the BPI Challenge 2020 log is CC BY-NC 4.0 (no commercial use); the others are under the 4TU General
Terms of Use. All are for research here; check the terms before any other use.
"""
import os

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
FILES = 'https://data.4tu.nl/file/'

DATASETS = {
    'bpi12': dict(
        title='BPI Challenge 2012: loan applications, Dutch bank',
        doi='10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f',
        file='bpi12_raw/BPI_Challenge_2012.xes.gz', md5='74c7ba9aba85bfcb181a22c9d565e5b5',
        url=FILES + '533f66a4-8911-4ac7-8612-1235d65d1f37/3276db7f-8bee-4f2b-88ee-92dbffb5a893',
        outcomes={'A_APPROVED': 'approved', 'A_DECLINED': 'declined', 'A_CANCELLED': 'cancelled'},
        lifecycle='W_', amount=('case', 'AMOUNT_REQ'),
        deadlines=[1, 3, 7, 14, 30], history_days=[14, 28, 56, 91], test_days=21,
        cutoffs=['2011-12-11', '2012-01-01', '2012-01-15']),
    'bpi17': dict(
        title='BPI Challenge 2017: loan applications, same bank, new system',
        doi='10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b',
        file='bpi17_raw/BPI_Challenge_2017.xes.gz', md5='10b37a2f78e870d78406198403ff13d2',
        url=FILES + '34c3f44b-3101-4ea9-8281-e38905c68b8d/f3aec4f7-d52c-4217-82f4-57d719a8298c',
        outcomes={'A_Pending': 'approved', 'A_Denied': 'declined', 'A_Cancelled': 'cancelled'},
        lifecycle='W_', amount=('case', 'RequestedAmount'),
        deadlines=[1, 3, 7, 14, 30], history_days=[14, 28, 56, 91], test_days=21,
        cutoffs=['2016-04-01', '2016-05-14', '2016-06-26', '2016-08-08', '2016-09-20', '2016-11-01']),
    'traffic': dict(
        title='Road traffic fines, Italian local police',
        doi='10.4121/uuid:270fd440-1057-4fb9-89a9-b699b47990f5',
        file='traffic_raw/Road_Traffic_Fine_Management_Process.xes.gz', md5='dc4cae9a592938af92932ca6c043baeb',
        url=FILES + '806acd1a-2bf2-4e39-be21-69b8cad10909/b234b06c-4d4f-4055-9f14-6218e3906d82',
        outcomes={'Payment': 'first payment', 'Send for Credit Collection': 'credit collection'},
        lifecycle=None, amount=('first_event', 'amount'),
        deadlines=[7, 30, 90, 180, 365], history_days=[182, 365, 730, 1095], test_days=30,
        # no cutoff after 2010: the log thins out in 2012 and 2013
        cutoffs=['2003-01-01', '2004-07-26', '2006-02-18', '2007-09-13', '2009-04-07', '2010-11-01']),
    'bpi20': dict(
        title='BPI Challenge 2020: domestic travel expense claims, Dutch university',
        doi='10.4121/uuid:3f422315-ed9d-4882-891f-e180b5b4feb5',
        file='bpi20_domestic_raw/DomesticDeclarations.xes.gz', md5='6a78c39491498363ce4788e0e8ca75ef',
        url=FILES + '6a0a26d2-82d0-4018-b1cd-89afb0e8627f/6eeb0328-f991-48c7-95f2-35033504036e',
        outcomes={'Payment Handled': 'paid'},
        lifecycle=None, amount=('case', 'Amount'),
        deadlines=[3, 5, 7, 10, 14], history_days=[7, 14, 28, 42], test_days=21,
        # 2018 only: in 2017 the process ran as a trial in two departments
        cutoffs=['2018-02-12', '2018-04-11', '2018-06-09', '2018-08-06', '2018-10-04', '2018-12-01']),
    'helpdesk': dict(
        title='Help desk tickets, Italian software company',
        doi='10.4121/uuid:0c60edf1-6f83-4e75-9367-4c63b3e9d5bb',
        file='helpdesk_raw/finale.csv', md5='a91d74377c55be5cbdfda05dae69db30',
        url=FILES + '94ee26c8-78f6-4387-b32b-f028f2103a2c/697ff9af-fca0-4363-bd84-e81735d27e9f',
        outcomes={'Resolve ticket': 'resolved'},
        lifecycle=None, amount=None,
        deadlines=[1, 3, 7, 14, 30], history_days=[14, 28, 56, 91], test_days=60,
        cutoffs=['2010-07-01', '2011-02-11', '2011-09-24', '2012-05-06', '2012-12-17', '2013-08-01']),
}

# ---------------------------------------------------------------------------------------------------------------------
# Phase B. Outcome events were chosen from the activity lists; everything else is the output of design.py
# (PROTOCOL_B.md, section 2), computed on 2026-11-14 before any model was trained on these logs.
PERMIT = {'01_HOOFD_510_2': 'decision sent', '01_HOOFD_510_3': 'decision sent', '01_HOOFD_510_4': 'decision sent'}
# "enter senddate decision (environmental permit)": the day the decision on the permit was sent to the applicant


def permit(m, cases, file_id, md5, doi, deadlines, history, cutoffs):
    return dict(
        title=f'BPI Challenge 2015: building permits, Dutch municipality {m}', doi=doi, org=f'municipality {m}', phase='B',
        file=f'bpi15_{m}_raw/BPIC15_{m}.xes', md5=md5, url=FILES + file_id,
        outcomes=PERMIT, lifecycle=None, amount=None, vocab='frequency',
        deadlines=deadlines, history_days=history, test_days=90, cutoffs=cutoffs)


DATASETS.update({
    'bpi14': dict(
        title='BPI Challenge 2014: IT incidents, Dutch bank service desk', org='service desk', phase='B',
        doi='10.4121/uuid:86977bac-f874-49cf-8337-80f26bf5d2ef',
        file='bpi14_raw/Detail_Incident_Activity.csv', md5='afecd95a464271b97c745f361d3516af',
        url=FILES + '657fb1d6-b4c2-4adc-ba48-ed25bf313025/bd6cfa31-44f8-4542-9bad-f1f70c894728',
        csv=dict(case='Incident ID', activity='IncidentActivity_Type', time='DateStamp', time_format='%d-%m-%Y %H:%M:%S',
                 delimiter=';'),
        outcomes={'Closed': 'closed'},  # the incident is closed (it can be reopened later; the first closure counts)
        lifecycle=None, amount=None,
        deadlines=[1, 5, 14], history_days=[7, 14, 28, 42], test_days=21,
        cutoffs=['2013-10-29', '2013-11-19', '2013-12-11', '2014-01-01', '2014-01-22', '2014-02-12']),
    'bpi20r': dict(
        title='BPI Challenge 2020: requests for payment, Dutch university', org='university', phase='B',
        doi='10.4121/uuid:895b26fb-6f25-46eb-9e48-0dca26fcd030',
        file='bpi20_request_raw/RequestForPayment.xes.gz', md5='2eb4dd20e70b8de4e32cc3c239bde7f2',
        url=FILES + 'a6f651a7-5ce0-4bc6-8be1-a7747effa1cc/7b1f2e56-e4a8-43ee-9a09-6e64f45a1a98',
        outcomes={'Payment Handled': 'paid'}, lifecycle=None, amount=('case', 'RequestedAmount'),
        not_before='2018-01-01',  # 2017 was a trial in two departments, as for bpi20
        deadlines=[3, 5, 7, 10, 21], history_days=[10, 21, 42, 63], test_days=21,
        cutoffs=['2018-03-13', '2018-04-30', '2018-06-17', '2018-08-03', '2018-09-20', '2018-11-07']),
    'bpi15_1': permit(1, 1199, '6f35269e-4ce7-4bc4-9abb-b3cea04cad00/2c8d5827-3e08-471d-98e2-6ffdec92f958',
                      '6c8a04d566eef571cf0aaf7b6f70310b', '10.4121/uuid:a0addfda-2044-4541-a450-fdcc9fe16d17',
                      [1, 7, 21, 45, 90], [45, 90, 180, 270],
                      ['2011-07-21', '2012-02-01', '2012-08-13', '2013-02-24', '2013-09-06', '2014-03-20']),
    'bpi15_3': permit(3, 1409, 'd6741425-5f62-4a59-92c5-08bae64b4611/21b574ab-02ba-4dfb-badc-bb46ce0edc44',
                      '7fee0bd0bfb8169cb2f91f2c77d2598e', '10.4121/uuid:ed445cdd-27d5-4d77-a1f7-59fe7360cfbe',
                      [1, 2, 7, 30, 60], [30, 60, 120, 180],
                      ['2011-04-17', '2011-12-05', '2012-07-24', '2013-03-13', '2013-10-31', '2014-06-20']),
    'bpi15_4': permit(4, 1053, '34216d8a-f054-46d4-bf03-d9352f90967e/68923819-b085-43be-abe2-e084a0f1381f',
                      'fac8a0d1d7ee074c2780bc75b9154c6a', '10.4121/uuid:679b11cf-47cd-459e-a6de-9ca614e25985',
                      [1, 5, 14, 30, 60], [30, 60, 120, 180],
                      ['2011-04-20', '2011-12-07', '2012-07-26', '2013-03-15', '2013-11-01', '2014-06-21']),
    'bpi15_5': permit(5, 1156, '32b70553-0765-4808-b155-aa5319802c8a/d39e1365-e4b8-4cb8-83d3-0b01cbf6f8c2',
                      '25fbd274fa1cd1c447bab98e22816359', '10.4121/uuid:b32c6fe5-f212-4286-9774-58dd53511cf8',
                      [2, 5, 10, 45, 60], [30, 60, 120, 180],
                      ['2011-04-18', '2011-12-06', '2012-07-26', '2013-03-15', '2013-11-02', '2014-06-22']),
})
for _name, _org in (('bpi12', 'bank'), ('bpi17', 'bank'), ('traffic', 'police'), ('bpi20', 'university'), ('helpdesk', 'software company')):
    DATASETS[_name].setdefault('org', _org)   # the organisation, for the pooled statistics
    DATASETS[_name].setdefault('phase', '1')

# Examined for phase B and left out, with the criterion of PROTOCOL_B.md that failed.
REJECTED = {
    'BPI Challenge 2013, incidents': 'grid (already in version 1): 99% of the outcomes fall within two months',
    'BPI Challenge 2013, closed problems': '3: an extract of closed cases only',
    'BPI Challenge 2015, municipality 2': '4: 46 test cases per cutoff, fewer than 50',
    'BPI Challenge 2018, subsidy applications': '4: no usable period (cases last most of a year; longest history 810 days in a log of three years)',
    'BPI Challenge 2019, purchase orders': '4: no usable period (one year of orders; longest history 360 days)',
    'BPI Challenge 2020, international declarations': '4: no usable period (longest history 270 days, 2018 only)',
    'BPI Challenge 2020, prepaid travel costs': '4: two cutoff dates, fewer than three',
    'BPI Challenge 2020, travel permits': '5: the same trips as the international declarations and requests',
    'CoSeLoG permit logs, municipalities 1 to 5, and the receipt phase': '5: the same cases as BPI Challenge 2015 (identical counts per year)',
}


def derived(name):
    return os.path.join(DATA, 'derived', name + '.pkl')
