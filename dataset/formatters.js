function toTwoDigitString(num) {
  return num < 10 ? `0${num}` : `${num}`;
}

/** Date format such as 01202019. */
function dateTupleToDDMMMYYYY(dateTuple) {
  const monthStr = MONTH_NAMES_3LETTER[dateTuple[1] - 1];
  const dayStr = toTwoDigitString(dateTuple[2]);
  return `${dayStr}${monthStr}${dateTuple[0]}`;
}

/** Date format such as 01/20/2019. */
function dateTupleToMMSlashDDSlashYYYY(dateTuple) {
  const monthStr = toTwoDigitString(dateTuple[1]);
  const dayStr = toTwoDigitString(dateTuple[2]);
  return `${monthStr}/${dayStr}/${dateTuple[0]}`;
}

/** Date format such as 1/20/2019. */
function dateTupleToMSlashDSlashYYYY(dateTuple) {
  return `${dateTuple[1]}/${dateTuple[2]}/${dateTuple[0]}`;
}

/** Date format such as 01/20/19. */
function dateTupleToMMSlashDDSlashYY(dateTuple) {
  const monthStr = toTwoDigitString(dateTuple[1]);
  const dayStr = toTwoDigitString(dateTuple[2]);
  const yearStr = `${dateTuple[0]}`.slice(2);
  return `${monthStr}/${dayStr}/${yearStr}`;
}

/** Date format such as 1/20/19. */
function dateTupleToMSlashDSlashYY(dateTuple) {
  const yearStr = `${dateTuple[0]}`.slice(2);
  return `${dateTuple[1]}/${dateTuple[2]}/${yearStr}`;
}

/** Date format such as 012019. */
function dateTupleToMMDDYY(dateTuple) {
  const monthStr = toTwoDigitString(dateTuple[1]);
  const dayStr = toTwoDigitString(dateTuple[2]);
  const yearStr = `${dateTuple[0]}`.slice(2);
  return `${monthStr}${dayStr}${yearStr}`;
}

/** Date format such as JAN 20 19. */
function dateTupleToMMMSpaceDDSpaceYY(dateTuple) {
  const monthStr = MONTH_NAMES_3LETTER[dateTuple[1] - 1];
  const dayStr = toTwoDigitString(dateTuple[2]);
  const yearStr = `${dateTuple[0]}`.slice(2);
  return `${monthStr} ${dayStr} ${yearStr}`;
}

/** Date format such as JAN 20 2019. */
function dateTupleToMMMSpaceDDSpaceYYYY(dateTuple) {
  const monthStr = MONTH_NAMES_3LETTER[dateTuple[1] - 1];
  const dayStr = toTwoDigitString(dateTuple[2]);
  return `${monthStr} ${dayStr} ${dateTuple[0]}`;
}

/** Date format such as JAN 20, 19. */
function dateTupleToMMMSpaceDDCommaSpaceYY(dateTuple) {
  const monthStr = MONTH_NAMES_3LETTER[dateTuple[1] - 1];
  const dayStr = toTwoDigitString(dateTuple[2]);
  const yearStr = `${dateTuple[0]}`.slice(2);
  return `${monthStr} ${dayStr}, ${yearStr}`;
}

/** Date format such as JAN 20, 2019. */
function dateTupleToMMMSpaceDDCommaSpaceYYYY(dateTuple) {
  const monthStr = MONTH_NAMES_3LETTER[dateTuple[1] - 1];
  const dayStr = toTwoDigitString(dateTuple[2]);
  return `${monthStr} ${dayStr}, ${dateTuple[0]}`;
}

/** Date format such as 20-01-2019. */
function dateTupleToDDDashMMDashYYYY(dateTuple) {
  const monthStr = toTwoDigitString(dateTuple[1]);
  const dayStr = toTwoDigitString(dateTuple[2]);
  return `${dayStr}-${monthStr}-${dateTuple[0]}`;
}

/** Date format such as 20-1-2019. */
function dateTupleToDDashMDashYYYY(dateTuple) {
  return `${dateTuple[2]}-${dateTuple[1]}-${dateTuple[0]}`;
}

/** Date format such as 20.01.2019. */
function dateTupleToDDDotMMDotYYYY(dateTuple) {
  const monthStr = toTwoDigitString(dateTuple[1]);
  const dayStr = toTwoDigitString(dateTuple[2]);
  return `${dayStr}.${monthStr}.${dateTuple[0]}`;
}

/** Date format such as 20.1.2019. */
function dateTupleToDDotMDotYYYY(dateTuple) {
  return `${dateTuple[2]}.${dateTuple[1]}.${dateTuple[0]}`;
}

/** Date format such as 2019.01.20. */
function dateTupleToYYYYDotMMDotDD(dateTuple) {
  const monthStr = toTwoDigitString(dateTuple[1]);
  const dayStr = toTwoDigitString(dateTuple[2]);
  return `${dateTuple[0]}.${monthStr}.${dayStr}`;
}

/** Date format such as 2019.1.20. */
function dateTupleToYYYYDotMDotD(dateTuple) {
  return `${dateTuple[0]}.${dateTuple[1]}.${dateTuple[2]}`;
}

/** Date format such as 20190120. */
function dateTupleToYYYYMMDD(dateTuple) {
  const monthStr = toTwoDigitString(dateTuple[1]);
  const dayStr = toTwoDigitString(dateTuple[2]);
  return `${dateTuple[0]}${monthStr}${dayStr}`;
}

/** Date format such as 2019-1-20. */
function dateTupleToYYYYDashMDashD(dateTuple) {
  return `${dateTuple[0]}-${dateTuple[1]}-${dateTuple[2]}`;
}

/** Date format such as 20 JAN 2019. */
function dateTupleToDSpaceMMMSpaceYYYY(dateTuple) {
  const monthStr = MONTH_NAMES_3LETTER[dateTuple[1] - 1];
  return `${dateTuple[2]} ${monthStr} ${dateTuple[0]}`;
}

/**
 * Date format such as 2019-01-20
 * (i.e.,  the ISO format and the conversion target).
 * */
function dateTupleToYYYYDashMMDashDD(dateTuple) {
  const monthStr = toTwoDigitString(dateTuple[1]);
  const dayStr = toTwoDigitString(dateTuple[2]);
  return `${dateTuple[0]}-${monthStr}-${dayStr}`;
}

module.exports = {
  toTwoDigitString,
  dateTupleToDDMMMYYYY,
  dateTupleToMMSlashDDSlashYYYY,
  dateTupleToMSlashDSlashYYYY,
  dateTupleToMMSlashDDSlashYY,
  dateTupleToMSlashDSlashYY,
  dateTupleToMMDDYY,
  dateTupleToMMMSpaceDDSpaceYY,
  dateTupleToMMMSpaceDDSpaceYYYY,
  dateTupleToMMMSpaceDDCommaSpaceYY,
  dateTupleToMMMSpaceDDCommaSpaceYYYY,
  dateTupleToDDDashMMDashYYYY,
  dateTupleToDDashMDashYYYY,
  dateTupleToDDDotMMDotYYYY,
  dateTupleToDDotMDotYYYY,
  dateTupleToYYYYDotMMDotDD,
  dateTupleToYYYYDotMDotD,
  dateTupleToYYYYMMDD,
  dateTupleToYYYYDashMDashD,
  dateTupleToDSpaceMMMSpaceYYYY,
  dateTupleToYYYYDashMMDashDD
};
