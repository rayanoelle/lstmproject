import http from 'k6/http';

import { check, sleep } from 'k6';


export const options = {

  stages: [
    { duration: '5m', target: 100 },
    { duration: '5m', target: 100 },
    { duration: '5m', target: 0 },
    { duration: '5m', target: 150 },
    { duration: '5m', target: 150 },
    { duration: '5m', target: 0 },
    

  ],

};


export default function () {
  // let value = Math.floor(Math.random() * 35);

  // const res = http.get('http://192.168.39.122:31101/'+ value);

  const res = http.get('http://192.168.39.122:31101/30');
  check(res, { 'status was 200': (r) => r.status == 200 });

  sleep(1);

}