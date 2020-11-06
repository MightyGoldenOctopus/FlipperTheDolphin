from currency_converter import CurrencyConverter
from datetime import date, datetime
import requests
import urllib3
import json
import os

urllib3.disable_warnings()


class Jump:
    def __init__(self):
        self.user = os.getenv('API_USER')
        self.password = os.getenv('API_PASSWORD')
        self.url = "https://dolphin.jump-technology.com:8443/api/v1/"
        self.c = CurrencyConverter(fallback_on_missing_rate=True)

    def __get_data(self, endpoint, args, date=None, full_response=False):
        payload = {
            "date": date,
            "fullResponse": full_response
        }
        args_string = "?" + "".join([f"{'columns' if 'columns' in key else key}={value}&"
                                     for key, value in args.items()])[:-1]
        res = requests.get(
            self.url + endpoint + args_string,
            params=payload,
            auth=(self.user, self.password),
            verify=False
        )
        return res.content.decode("utf-8")

    def __put_data(self, endpoint, data):
        res = requests.put(
            self.url + endpoint,
            data=data
        )
        return res

    def __post_data(self, endpoint, data):
        res = requests.post(
            self.url + endpoint,
            data=data
        )
        return res

    def get_assets(self, date="2020-09-30"):
        asset_dict = {}
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        res = json.loads(
            self.__get_data(
                endpoint="asset",
                args={
                    "columns0": "ASSET_DATABASE_ID",
                    "columns1": "LABEL",
                    "columns2": "TYPE",
                    "columns3": "LAST_CLOSE_VALUE_IN_CURR",
                    "date": date
                }
            )
        )
        for asset in res:
            id = asset["ASSET_DATABASE_ID"]["value"]
            type = asset["TYPE"]["value"]
            converted = False
            if "LAST_CLOSE_VALUE_IN_CURR" not in asset:
                continue
            value, currency = asset["LAST_CLOSE_VALUE_IN_CURR"]["value"].split(" ")
            value, value_usd = float(value.replace(",", ".")), float(value.replace(",", "."))
            if currency != "USD":
                converted = True
                value_usd = self.c.convert(value, currency, "USD", date=date_obj)
            asset_dict[id] = {
                "type": type,
                "last_close": value,
                "last_close_usd": value_usd,
                "original_currency": currency,
                "converted": converted,
                "date": date
            }
        return asset_dict

    def get_asset(self, asset_id, start_date="2016-06-01", end_date="2020-09-30"):
        values_list = []
        res = json.loads(
            self.__get_data(
                endpoint=f"asset/{asset_id}/quote",
                args={
                    "start_date": start_date,
                    "end_date": end_date
                }
            )
        )
        # Get asset info for currency conversion
        currency = self.get_assets()[f"{asset_id}"]["original_currency"]
        for value in res:
            date = value["date"]["value"]
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            nav = nav_usd = float(value["nav"]["value"].replace(",", "."))
            gross = gross_usd = float(value["gross"]["value"].replace(",", "."))
            pl = float(value["pl"]["value"].replace(",", "."))
            close = close_usd = float(value["real_close_price"]["value"].replace(",", "."))
            ret = float(value["return"]["value"].replace(",", "."))
            converted = False
            if currency != "USD":
                converted = True
                nav_usd = self.c.convert(nav_usd, currency, "USD", date=date_obj)
                gross_usd = self.c.convert(gross_usd, currency, "USD", date=date_obj)
                close_usd = self.c.convert(close_usd, currency, "USD", date=date_obj)
            values_list.append({
                "date": date,
                "pl": pl,
                "return": ret,
                "original_currency": currency,
                "converted": converted,
                "values": {
                    "nav": nav,
                    "gross": gross,
                    "close": close
                },
                "values_usd": {
                    "nav": nav_usd,
                    "gross": gross_usd,
                    "close": close_usd
                }
            })
        return values_list

    def get_ratio(self):
        res = json.loads(
            self.__get_data(
                endpoint=f"ratio",
                args={}
            )
        )
        return res

    def get_portfolio(self, portfolio_id):
        res = json.loads(
            self.__get_data(
                endpoint=f"portfolio/{portfolio_id}/dyn_amount_compo",
                args={}
            )
        )
        return res

    def update_portfolio(self, portfolio_id, portfolio_obj):
        res = self.__put_data(
            endpoint=f"portfolio/{portfolio_id}/dyn_amount_compo",
            data=json.dumps(portfolio_obj)
        )
        return res

    def calculate_ratio(self, ratio_ids, asset_ids, start_date="2016-06-01", end_date="2020-09-30"):
        payload = json.dumps({
            "_ratio": ratio_ids,
            "_asset": asset_ids,
            "_bench": None,
            "_startDate": start_date,
            "_endDate": end_date,
            "_frequency": None
        })
        res = self.__post_data(
            endpoint="ratio/invoke",
            data=payload
        )
        return res

    def currency_conversion(self, origin_curr, target_curr):
        origin_curr, target_curr = origin_curr.upper(), target_curr.upper()
        res = json.loads(
            self.__get_data(
                endpoint=f"currency/rate/{origin_curr}/to/{target_curr}",
                args={}
            )
        )
        return res
