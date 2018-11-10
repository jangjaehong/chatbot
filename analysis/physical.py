import tensorflow as tf
import analysis.db as db


class Physical:
    def __init__(self):
        self.physical_data = []
        self.bmi_data = []
        self.data_size = 0

    def load_data(self, sdate, edate):
        bmireport = db.select_bmi_report_pub_date(sdate, edate)
        if bmireport is not None:
            self.data_size = 2
            for rows in bmireport:
                x_data = [rows[0], rows[1]]
                y_data = [rows[2]]
                self.physical_data.append(x_data)
                self.bmi_data.append(y_data)
        else:
            print("데이터가 없습니다.")


def main(_):
    print("예측모델 가동중입니다...\n")

if __name__ == "__main__":
    tf.app.run()