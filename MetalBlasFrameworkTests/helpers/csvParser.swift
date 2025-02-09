//
//  csvParser.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-08.
//

import Foundation

class csvParser
{
    var filename : URL

//    init()
//    {
//        filename = "data.csv"
//    }
//
//    init(file: String)
//    {
//        filename = file
//    }

    init(file: URL)
    {
        filename = file
    }

    func readFile() -> String
    {
        var data : String
        do
        {
            data = try String(contentsOf: filename, encoding: .utf8)
        }
        catch
        {
            print(error)
            data = ""
        }

        return data
    }

    func parse() -> [TestParams]
    {
        var allParams : [TestParams] = []
        let data = readFile()

        var rows = data.components(separatedBy: "\n")

        let headerRow = rows[0]
        rows.removeFirst()

        let headers = headerRow.components(separatedBy: ",")

        for i in 0..<rows.count
        {
            let row = rows[i]
            if row.isEmpty || (row.starts(with: "//"))
            {
                // dummy param for now
                allParams.append(TestParams())
                continue
            }

            let cols = row.components(separatedBy: ",")
            if cols.count != headers.count
            {
                print("Error in file ", filename, ", row #", i, " has ", cols.count, " cols, while header has ", headers.count, " cols. Skipping.")
                continue
            }

            var param = TestParams()

            for i in 0 ..< headers.count
            {
                param.set(headers[i], cols[i])
            }

            allParams.append(param)
        }

        return allParams
    }
}
