import psycopg2 as pg2
import random
import math
import datetime

def connect(host='113.198.224.50', port='5432', database='medibot', user='jjh', password='skwoghd1'):
    conn = False
    try:
        conn = pg2.connect(host=host, port=port, database=database, user=user, password=password)
        print("데이터베이스에 연결 하였습니다.")
    except Exception as e:
        print('데이터베이스에 연결 할 수 없습니다.')
        print('에러메세지: ', e)
    return conn


def select_top_bmi_report(name):
    bmi = None
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT bmi FROM medibot_bmireport WHERE name = %s ORDER BY pub_date LIMIT 1;", (name,))
            result = cur.fetchall()
            # result is empty
            if not result:
                bmi = 0.0
            else:
                bmi = result[0]
    return bmi


def select_bmi_report(sql):
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM medibot_bmireport")
            rows = cur.fetchall()
    return rows


def select_bmi_report_pub_date(sdate, edate):
    rows = None
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT stature, weight, bmi FROM medibot_bmireport "
                        "WHERE TO_DATE(to_char(pub_date,'YYYY-MM-DD'),'YYYY-MM-DD') >= to_timestamp('" + sdate + "', 'YYYY-MM-DD')"
                        "AND TO_DATE(to_char(pub_date,'YYYY-MM-DD'),'YYYY-MM-DD') <= to_timestamp('" + edate + "', 'YYYY-MM-DD')"
                        "AND uid=6")
            rows = cur.fetchall()
    return rows


def select_whr_report(date):
    rows = None
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM medibot_chatvocab")
            rows = cur.fetchall()
        conn.close()
    print("복부비만도(whr) 데이터를 읽어옵니다....")
    return rows


def select_energy_report(date):
    rows = None
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM medibot_themetype")
            rows = cur.fetchall()
        conn.close()
    print("기초대사량 데이터를 읽어옵니다....")
    return rows


def insert_bmi(data):
    with connect() as conn:
        with conn.cursor() as cur:

            sql = "INSERT INTO public.medibot_bmireport(stature, weight, bmi, pub_date, uid) VALUES (%s, %s, %s, %s, 6)"
            cur.execute(sql, data)
    print("bmi 데이터를 저장했습니다.")


def insert_whr():
    with connect() as conn:
        with conn.cursor() as cur:
            sql = '''
                        INSERT INTO medibot_chatvocab (vocab, morpheme)
                        VALUES (?, ? , ?, ?, ?)
                    '''
            cur.execute(sql, locals())
        conn.close()
    print("bmi 데이터를 저장했습니다.")


def insert_energy():
    with connect() as conn:
        with conn.cursor() as cur:
            sql = '''
                        INSERT INTO medibot_chatvocab (vocab, morpheme)
                        VALUES (?, ? , ?, ?, ?)
                    '''
            cur.execute(sql, locals())
        conn.close()
    print("bmi 데이터를 저장했습니다.")



for i in range(1000, -1, -1):
    stature = 174.8
    weight = random.randrange(60.0, 70.0)
    stature_pow = math.pow(stature / 100, 2)
    bmi_result = weight / stature_pow
    bmi_result = round(bmi_result, 3)
    pub_date = datetime.datetime.now() + datetime.timedelta(days=-i)
    data = (stature, weight, bmi_result, pub_date)
    insert_bmi(data)




