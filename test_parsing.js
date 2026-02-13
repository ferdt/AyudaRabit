function parseTextToData(text) {
    const lines = text.split('\n');
    const data = [];

    const rowRegex = /(\d+[.,]?\d*)\s*(km|m)?\s*[\t| ]+\s*(\d+[.,]?\d*)\s*(h|min|s)?/gi;

    for (const line of lines) {
        let match;
        while ((match = rowRegex.exec(line)) !== null) {
            let distRaw = match[1].replace(',', '.');
            let distUnit = (match[2] || 'km').toLowerCase();

            let timeRaw = match[3].replace(',', '.');
            let timeUnit = (match[4] || 'h').toLowerCase();

            let distance = parseFloat(distRaw);
            let time = parseFloat(timeRaw);

            if (!isNaN(distance) && !isNaN(time) && time > 0) {
                if (distUnit === 'm') distance = distance / 1000;

                if (timeUnit === 'min') {
                    time = time / 60;
                } else if (timeUnit === 's') {
                    time = time / 3600;
                }

                let velocity = distance / time;

                data.push({
                    distance: distance,
                    time: time,
                    velocity: velocity
                });
            }
        }
    }
    return data;
}

// Test Cases
const tests = [
    { name: "Standard", input: "100 km 1 h", expected: { d: 100, t: 1, v: 100 } },
    { name: "Implicit Units", input: "100 1", expected: { d: 100, t: 1, v: 100 } },
    { name: "Meters and Minutes", input: "1000 m 30 min", expected: { d: 1, t: 0.5, v: 2 } },
    { name: "Decimals with Comma", input: "10,5 km 2,0 h", expected: { d: 10.5, t: 2, v: 5.25 } },
    { name: "Complex Layout", input: "Dist: 100 km Time: 1h", expected: { d: 100, t: 1, v: 100 } }, // Regex might rely on spacing, let's see
    { name: "Multi-column", input: "100 1 200 2", expectedCount: 2 }
];

console.log("Running Tests...");
let passed = 0;

tests.forEach(test => {
    const result = parseTextToData(test.input);

    if (test.expectedCount) {
        if (result.length === test.expectedCount) {
            console.log(`[PASS] ${test.name}`);
            passed++;
        } else {
            console.error(`[FAIL] ${test.name}. Expected ${test.expectedCount} items, got ${result.length}`);
            console.log(JSON.stringify(result));
        }
    } else {
        if (result.length > 0) {
            const item = result[0];
            const dOk = Math.abs(item.distance - test.expected.d) < 0.001;
            const tOk = Math.abs(item.time - test.expected.t) < 0.001;
            const vOk = Math.abs(item.velocity - test.expected.v) < 0.001;

            if (dOk && tOk && vOk) {
                console.log(`[PASS] ${test.name}`);
                passed++;
            } else {
                console.error(`[FAIL] ${test.name}. Expected ${JSON.stringify(test.expected)}, got ${JSON.stringify(item)}`);
            }
        } else {
            console.error(`[FAIL] ${test.name}. No data parsed.`);
        }
    }
});

console.log(`\nPassed ${passed} / ${tests.length} tests.`);
