import psycopg2 as pg2

def connect(host='113.198.224.50', port='5432', database='medibot', user='jjh', password='skwoghd1'):
    conn = False
    try:
        conn = pg2.connect(host=host, port=port, database=database, user=user, password=password)
        print("데이터베이스에 연결 하였습니다.")
    except Exception as e:
        print('데이터베이스에 연결 할 수 없습니다.')
        print('에러메세지: ', e)
    return conn


def select_chat_sequence():
    rows = None
    sequence_data = []
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM medibot_chatsequence")
            rows = cur.fetchall()

    for row in rows:
        sequence_data.append([row[1], row[2]])
    print("질의응답 데이터를 읽어옵니다...")
    return sequence_data


def select_chat_vocab():
    rows = None
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM medibot_vocabdict")
            rows = cur.fetchall()
    print("어휘 사전 데이터를 읽어옵니다...")
    return rows


def select_chat_report():
    rows = None
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM medibot_themetype")
            rows = cur.fetchall()
    return rows


def delete_and_insert_vocab_list(vocab_dic):
    with connect() as conn:
        with conn.cursor() as cur:
            sql = "delete from medibot_vocabdict "
            cur.execute(sql)

            sql = '''
                INSERT INTO medibot_vocabdict (idx, vocab, morpheme)
                SELECT  unnest(%(idx)s),
                        unnest(%(vocab)s),
                        unnest(%(morpheme)s)
            '''
            idx = [r['idx'] for r in vocab_dic]
            vocab = [r['vocab'] for r in vocab_dic]
            morpheme = [r['morpheme'] for r in vocab_dic]
            cur.execute(sql, locals())
    print("어휘 사전을 데이터베이스에 저장했습니다...")















