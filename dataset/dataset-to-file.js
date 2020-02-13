const dateFormat = require("./date_format");
const { generateOrderedDates } = require("./generator");
const fs = require("fs").promises;

async function generator(minYear = 1950, maxYear = 2050) {
  // Generate ordered data sets
  const dateTuples = generateOrderedDates(minYear, maxYear);

  const ins = dateFormat.INPUT_FNS.map(fn =>
    dateTuples.map(tuple => {
      return {
        input: fn(tuple),
        output: dateFormat.dateTupleToYYYYDashMMDashDD(tuple)
      };
    })
  );

  await fs.writeFile(
    "./out.json",
    JSON.stringify(ins, { replacer: null, spaces: 2 })
  );
}

generator().catch(err => console.log(err));
