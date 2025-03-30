import fs from 'fs';
import request from "request";

import { S3 } from "@aws-sdk/client-s3";
import { Cloud } from "laf-client-sdk";
// import COS from 'cos-nodejs-sdk-v5';


export function stdlog(content: string, color: string = 'default') {
    // 定义颜色
    const colorsDict: { [key: string]: string } = {
        black: '\x1B[30m',
        red: '\x1B[31m',
        green: '\x1B[32m',
        yellow: '\x1B[33m',
        blue: '\x1B[34m',
        magenta: '\x1B[35m',
        cyan: '\x1B[36m',
        white: '\x1B[37m',
        default: '',
    };

    // 输出
    process.stdout.write(`${colorsDict[color]}${content}\x1B[0m`);
}

export async function getCos() {
    // var cos = new COS({
    //     Domain:"https://oss.laf.run/{Bucket}",
    //     getAuthorization: async function (options: Object, callback: Function) {
    //         // 初始化时不会调用，只有调用 cos 方法（例如 cos.putObject）时才会进入
    //         var cosTemp = config.COS_KEY;
    //         if (!cosTemp || !cosTemp.Credentials) {
    //             console.error("无效cosTemp信息: ", cosTemp)
    //             callback({
    //                 TmpSecretId: "empty",
    //                 TmpSecretKey: "empty",
    //                 SecurityToken: "empty",
    //                 ExpiredTime: "1111111111",
    //             });
    //             return;
    //         }
    //
    //         callback({
    //             TmpSecretId: cosTemp.Credentials.TmpSecretId,        // 临时密钥的 tmpSecretId
    //             TmpSecretKey: cosTemp.Credentials.TmpSecretKey,      // 临时密钥的 tmpSecretKey
    //             SecurityToken: cosTemp.Credentials.Token,            // 临时密钥的 sessionToken
    //             ExpiredTime: cosTemp.ExpiredTime,                    // 临时密钥失效时间戳，是申请临时密钥时，时间戳加 durationSeconds
    //         });
    //     }
    // });
    //
    // return cos;


// @ts-ignore
    const cloud = new Cloud({
  baseUrl: `https://mgcwld.laf.run`
});
// 获取云存储临时令
const { credentials, endpoint, region } =await cloud.invokeFunction(
  "get-oss-sts",
  {}
);
    //stdlog(JSON.stringify(res),"green")
    const s3 = new S3({
      endpoint: endpoint,
      region: region,
      credentials: {
        accessKeyId: credentials.AccessKeyId,
        secretAccessKey: credentials.SecretAccessKey,
        sessionToken: credentials.SessionToken,
        expiration: credentials.Expiration,
      },
      forcePathStyle: true,
    });
    //stdlog(String(s3),"green")
    return s3;
}




export async function downloadCosPath(cos: S3, path: string, localPath: string) {
    // const pathObj = _getRegionBucketPath(path);
        // bucket name prefixed with appid
    // const cmd = new GetObjectCommand({
    //   Bucket: pathObj.bucket,
    //   Key: pathObj.filePath,
    // });
    return new Promise((resolve, reject) => {
        var writeStream = fs.createWriteStream(localPath);
        path=encodeURI(path)
        request(path).on('response', (response) => {
            if (!/^image\//.test(response.headers['content-type'] as string)) {
                writeStream.close();
                fs.unlinkSync(localPath);
                reject(new Error('Download failed: Unauthorized'));
            }
        }).pipe(writeStream)
            .on('finish', resolve)
            .on('error', (error) => {
                writeStream.close();
                fs.unlinkSync(localPath);
                reject(error);
            });
    })
}


