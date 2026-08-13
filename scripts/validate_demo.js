#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const ROOT = path.resolve(__dirname, '..');
const FIXTURE_DIR = path.join(ROOT, 'showcase/demo_fixture');
const SCHEMA_PATH = path.join(ROOT, 'docs/demo/schema/demo_summary.schema.json');
const HTML_PATH = path.join(ROOT, 'docs/demo/index.html');

const results = [];
let failures = 0;

function assert(condition, message) {
  if (!condition) {
    failures++;
    results.push('FAIL: ' + message);
    throw new Error(message);
  }
  results.push('PASS: ' + message);
}

function sha256(filePath) {
  const content = fs.readFileSync(filePath);
  return crypto.createHash('sha256').update(content).digest('hex');
}

// --- lightweight JSON Schema draft-07 validator ---
function validateSchema(instance, schema, instanceName, errors) {
  if (schema.constructor !== undefined && schema.constructor.name !== 'Object') {
    return;
  }
  if (schema.type !== undefined && schema.type !== 'any') {
    const actualType = Array.isArray(instance) ? 'array' : (instance === null ? 'null' : typeof instance);
    if (actualType !== schema.type) {
      errors.push(instanceName + ': type must be ' + schema.type + ', got ' + actualType);
      return;
    }
  }
  if (schema.enum !== undefined) {
    if (!schema.enum.includes(instance)) {
      errors.push(instanceName + ': value must be one of ' + JSON.stringify(schema.enum));
    }
  }
  if (schema.const !== undefined && instance !== schema.const) {
    errors.push(instanceName + ': value must be ' + JSON.stringify(schema.const));
  }
  if (schema.minLength !== undefined && typeof instance === 'string' && instance.length < schema.minLength) {
    errors.push(instanceName + ': string length must be >= ' + schema.minLength);
  }
  if (schema.maxLength !== undefined && typeof instance === 'string' && instance.length > schema.maxLength) {
    errors.push(instanceName + ': string length must be <= ' + schema.maxLength);
  }
  if (schema.minProperties !== undefined && instance !== null && typeof instance === 'object' && !Array.isArray(instance)) {
    if (Object.keys(instance).length < schema.minProperties) {
      errors.push(instanceName + ': object must have >= ' + schema.minProperties + ' properties');
    }
  }
  if (schema.required !== undefined && instance !== null && typeof instance === 'object' && !Array.isArray(instance)) {
    for (const field of schema.required) {
      if (!(field in instance)) {
        errors.push(instanceName + ': missing required field ' + field);
      }
    }
  }
  if (schema.properties !== undefined && instance !== null && typeof instance === 'object' && !Array.isArray(instance)) {
    for (const [key, subSchema] of Object.entries(schema.properties)) {
      if (key in instance) {
        validateSchema(instance[key], subSchema, instanceName + '.' + key, errors);
      }
    }
  }
  if (schema.additionalProperties !== undefined && instance !== null && typeof instance === 'object' && !Array.isArray(instance)) {
    const allowed = schema.properties ? Object.keys(schema.properties) : [];
    for (const key of Object.keys(instance)) {
      if (!allowed.includes(key)) {
        errors.push(instanceName + ': additional property not allowed: ' + key);
      }
    }
  }
  if (schema.items !== undefined && Array.isArray(instance)) {
    instance.forEach((item, i) => validateSchema(item, schema.items, instanceName + '[' + i + ']', errors));
  }
  if (schema.if && schema.then) {
    let conditionMet = false;
    if (schema.if.properties && schema.if.properties.status && schema.if.properties.status.const !== undefined) {
      if (instance.status === schema.if.properties.status.const) {
        conditionMet = true;
      }
    }
    if (conditionMet) {
      const branchErrors = [];
      validateSchema(instance, schema.then, instanceName, branchErrors);
      errors.push(...branchErrors);
    } else if (schema.else) {
      if (schema.else.if && schema.else.then) {
        let nestedMet = false;
        if (schema.else.if.properties && schema.else.if.properties.status && schema.else.if.properties.status.const !== undefined) {
          if (instance.status === schema.else.if.properties.status.const) {
            nestedMet = true;
          }
        }
        if (nestedMet) {
          const nestedErrors = [];
          validateSchema(instance, schema.else.then, instanceName, nestedErrors);
          errors.push(...nestedErrors);
        } else if (schema.else.else) {
          if (schema.else.else.if && schema.else.else.then) {
            let deepestMet = false;
            if (schema.else.else.if.properties && schema.else.else.if.properties.status && schema.else.else.if.properties.status.const !== undefined) {
              if (instance.status === schema.else.else.if.properties.status.const) {
                deepestMet = true;
              }
            }
            if (deepestMet) {
              const deepestErrors = [];
              validateSchema(instance, schema.else.else.then, instanceName, deepestErrors);
              errors.push(...deepestErrors);
            }
          }
        }
      }
    }
  }
  if (schema.anyOf !== undefined && instance !== null && typeof instance === 'object' && !Array.isArray(instance)) {
    let anyValid = false;
    for (const subSchema of schema.anyOf) {
      const subErrors = [];
      validateSchema(instance, subSchema, instanceName, subErrors);
      if (subErrors.length === 0) {
        anyValid = true;
        break;
      }
    }
    if (!anyValid && schema.anyOf.length > 0) {
      errors.push(instanceName + ': must satisfy at least one of anyOf');
    }
  }
  if (schema.not !== undefined && instance !== null && typeof instance === 'object' && !Array.isArray(instance)) {
    const notErrors = [];
    validateSchema(instance, schema.not, instanceName, notErrors);
    if (notErrors.length === 0 && Object.keys(schema.not).length > 0) {
      errors.push(instanceName + ': must NOT satisfy the not schema');
    }
  }
}

try {
  // 1. JSON parse validation
  const jsonFiles = fs.readdirSync(FIXTURE_DIR).filter(f => f.endsWith('.json'));
  assert(jsonFiles.length >= 5, 'fixture directory has at least 5 JSON files');

  const fixtures = {};
  for (const file of jsonFiles) {
    const content = fs.readFileSync(path.join(FIXTURE_DIR, file), 'utf8');
    const parsed = JSON.parse(content);
    fixtures[file] = parsed;
    results.push('PASS: JSON parse ' + file);
  }

  const schema = JSON.parse(fs.readFileSync(SCHEMA_PATH, 'utf8'));
  const schemaContent = fs.readFileSync(SCHEMA_PATH, 'utf8');
  results.push('PASS: JSON parse demo_summary.schema.json');

  // 1a. JSON Schema validation (draft-07 lightweight validator) for model summary files
  const modelFiles = ['model_main.json', 'model_independent.json', 'model_unavailable.json'];
  for (const file of modelFiles) {
    const schemaErrors = [];
    validateSchema(fixtures[file], schema, file, schemaErrors);
    assert(schemaErrors.length === 0, file + ' passes JSON Schema validation (0 errors, got ' + schemaErrors.length + ')');
    for (const err of schemaErrors) {
      results.push('SCHEMA_ERR: ' + file + ' ' + err);
    }
  }
  results.push('PASS: All model summary files pass JSON Schema validation');

  // 1b. State exclusivity via Schema if/then/else
  for (const file of ['model_main.json', 'model_independent.json', 'model_unavailable.json']) {
    const model = fixtures[file];
    if (model.status === 'unavailable') {
      const hasMetrics = model.metrics && Object.keys(model.metrics).length > 0;
      const hasTerminal = !!model.terminal;
      assert(!hasMetrics && !hasTerminal, file + ': unavailable must not have metrics or terminal');
    }
  }
  results.push('PASS: State exclusivity enforced by Schema branches');

  // 1c. Schema structure: metrics must reject empty object
  assert(schema.properties.metrics.minProperties === 1, 'Schema enforces metrics minProperties=1');
  results.push('PASS: Schema rejects empty metrics object');

  // 1d. Schema structure: unavailable branch uses true exclusivity
  const unavailableBranch = schema.else.else.then;
  assert(
    unavailableBranch.not &&
    unavailableBranch.not.anyOf &&
    unavailableBranch.not.anyOf.length === 2,
    'Schema unavailable branch uses anyOf not: required for true exclusivity'
  );
  results.push('PASS: Schema unavailable branch has true exclusivity constraint');

  // 2. Schema required fields
  assert(schema.required.includes('prediction_commitment'), 'Schema requires prediction_commitment');
  assert(schema.required.includes('scored'), 'Schema requires scored');
  assert(schema.required.includes('comparison'), 'Schema requires comparison');

  // 3. Model validation against Schema constraints
  function validateModel(model, filename) {
    const validStatuses = ['scored', 'terminal_pre_prediction', 'unavailable'];
    assert(validStatuses.includes(model.status), filename + ': status is valid');

    if (model.status === 'scored') {
      assert(model.prediction_commitment === true, filename + ': scored must have prediction_commitment=true');
      assert(model.scored === true, filename + ': scored must have scored=true');
      assert(model.metrics && Object.keys(model.metrics).length > 0, filename + ': scored must have non-empty metrics');
      assert(!model.terminal, filename + ': scored must not have terminal');
    } else if (model.status === 'terminal_pre_prediction') {
      assert(model.prediction_commitment === false, filename + ': terminal_pre_prediction must have prediction_commitment=false');
      assert(model.scored === false, filename + ': terminal_pre_prediction must have scored=false');
      assert(!model.metrics || Object.keys(model.metrics).length === 0, filename + ': terminal_pre_prediction must not have metrics');
      assert(model.terminal, filename + ': terminal_pre_prediction must have terminal');
    } else if (model.status === 'unavailable') {
      assert(model.prediction_commitment === false, filename + ': unavailable must have prediction_commitment=false');
      assert(model.scored === false, filename + ': unavailable must have scored=false');
      assert(!model.metrics || Object.keys(model.metrics).length === 0, filename + ': unavailable must not have metrics');
      assert(!model.terminal, filename + ': unavailable must not have terminal');
      assert(!model.gages, filename + ': unavailable must not have gages');
      assert(!model.public_version, filename + ': unavailable must not have public_version');
      assert(!model.protocol_id, filename + ': unavailable must not have protocol_id');
    }

    // Comparison fields validation
    if (model.comparison) {
      const requiredFields = ['data_version', 'prefix_definition', 'prediction_range', 'partition_id', 'scoring_rule_id', 'metric_name', 'metric_unit', 'protocol_id', 'scored'];
      for (const field of requiredFields) {
        assert(field in model.comparison, filename + ': comparison must have ' + field);
      }
      // All string fields must be non-empty
      const stringFields = requiredFields.filter(f => f !== 'scored');
      for (const field of stringFields) {
        assert(typeof model.comparison[field] === 'string' && model.comparison[field].length > 0, filename + ': comparison.' + field + ' must be non-empty string');
      }
      assert(typeof model.comparison.scored === 'boolean', filename + ': comparison.scored must be boolean');
      assert(model.comparison.scored === model.scored, filename + ': comparison.scored must match top-level scored');
    }
  }

  validateModel(fixtures['model_main.json'], 'model_main.json');
  validateModel(fixtures['model_independent.json'], 'model_independent.json');
  validateModel(fixtures['model_unavailable.json'], 'model_unavailable.json');

  // 4. Comparison gate tests
  function compareModels(a, b) {
    const keys = ['data_version', 'prefix_definition', 'prediction_range', 'partition_id', 'scoring_rule_id', 'metric_name', 'metric_unit', 'protocol_id'];
    for (const key of keys) {
      const av = (a.comparison && a.comparison[key]) || '';
      const bv = (b.comparison && b.comparison[key]) || '';
      if (av !== bv) return false;
      if (!av || !bv) return false;
      if (typeof av !== 'string' || typeof bv !== 'string') return false;
    }
    if (!a.comparison || !b.comparison) return false;
    if (a.comparison.scored !== b.comparison.scored) return false;
    return a.scored === b.comparison.scored && b.scored === a.comparison.scored && a.scored === true && b.scored === true;
  }

  assert(compareModels(fixtures['model_main.json'], fixtures['model_independent.json']) === false, 'comparison gate: should not compare different models');
  results.push('PASS: Comparison gate rejects different models');

  // Same model with scored=true should compare
  const fakeModelScored = JSON.parse(JSON.stringify(fixtures['model_main.json']));
  fakeModelScored.scored = true;
  fakeModelScored.prediction_commitment = true;
  fakeModelScored.metrics = { test: { value: 1, unit: 'pp', role: 'verified_fact', evidence_grade: 'E2' } };
  fakeModelScored.comparison.scored = true;
  assert(compareModels(fixtures['model_main.json'], fakeModelScored) === false, 'comparison gate: should not compare models with different scored status');

  const fakeModelSame = JSON.parse(JSON.stringify(fixtures['model_main.json']));
  fakeModelSame.scored = true;
  fakeModelSame.prediction_commitment = true;
  fakeModelSame.metrics = { test: { value: 1, unit: 'pp', role: 'verified_fact', evidence_grade: 'E2' } };
  fakeModelSame.comparison.scored = true;
  const mainForCompare = JSON.parse(JSON.stringify(fixtures['model_main.json']));
  mainForCompare.scored = true;
  mainForCompare.prediction_commitment = true;
  mainForCompare.metrics = { test: { value: 1, unit: 'pp', role: 'verified_fact', evidence_grade: 'E2' } };
  mainForCompare.comparison.scored = true;
  assert(compareModels(mainForCompare, fakeModelSame) === true, 'comparison gate: should compare identical scored models');
  results.push('PASS: Comparison gate accepts identical scored models');

  // Different comparison field should reject
  const fakeModelDiffField = JSON.parse(JSON.stringify(fakeModelSame));
  fakeModelDiffField.comparison.data_version = 'v2';
  fakeModelDiffField.comparison.scored = true;
  assert(compareModels(mainForCompare, fakeModelDiffField) === false, 'comparison gate: should reject different data_version');
  results.push('PASS: Comparison gate rejects mismatched comparison fields');

  // Empty strings should reject
  const fakeModelEmpty = JSON.parse(JSON.stringify(fakeModelSame));
  fakeModelEmpty.comparison.data_version = '';
  fakeModelEmpty.comparison.scored = true;
  assert(compareModels(mainForCompare, fakeModelEmpty) === false, 'comparison gate: should reject empty comparison fields');
  results.push('PASS: Comparison gate rejects empty comparison fields');

  // 5. Workbench scenarios validation
  const scenarios = fixtures['workbench_scenarios.json'];
  assert(scenarios.scenarios && Array.isArray(scenarios.scenarios), 'workbench_scenarios has scenarios array');
  assert(scenarios.scenarios.length >= 3, 'at least 3 scenarios');

  for (const s of scenarios.scenarios) {
    assert(s.nominal_interval_level !== undefined, 'scenario ' + s.id + ' has nominal_interval_level');
    assert(!s.diagnostic_interval_coverage, 'scenario ' + s.id + ' should not have diagnostic_interval_coverage');
    assert(['predict', 'fallback', 'reject'].includes(s.decision), 'scenario ' + s.id + ' has valid decision');
    assert(typeof s.temperature_c === 'number', 'scenario ' + s.id + ' has temperature_c');
    assert(typeof s.soc_fraction === 'number', 'scenario ' + s.id + ' has soc_fraction');
    assert(typeof s.prefix_count === 'number', 'scenario ' + s.id + ' has prefix_count');
    assert(typeof s.horizon_days === 'number', 'scenario ' + s.id + ' has horizon_days');
    assert(typeof s.data_quality === 'string', 'scenario ' + s.id + ' has data_quality');
    assert(typeof s.domain_support === 'string', 'scenario ' + s.id + ' has domain_support');
  }

  results.push('PASS: All workbench scenarios have all 6 selector fields');

  // 6. Update sequence validation
  const updateSeq = fixtures['update_demo.json'];
  assert(updateSeq.sequence && Array.isArray(updateSeq.sequence), 'update_demo has sequence array');
  assert(updateSeq.sequence.length >= 4, 'at least 4 steps');

  for (const step of updateSeq.sequence) {
    assert(step.nominal_interval_level !== undefined, 'step ' + step.label + ' has nominal_interval_level');
    assert(!step.diagnostic_interval_coverage, 'step ' + step.label + ' should not have diagnostic_interval_coverage');
    assert(step.checkpoints && step.checkpoints.length > 0, 'step ' + step.label + ' has checkpoints');
  }

  results.push('PASS: Update sequence uses nominal_interval_level');

  // 7. HTML embedded data vs fixture drift check
  const html = fs.readFileSync(HTML_PATH, 'utf8');
  const scriptMatch = html.match(/<script id="demo-data" type="application\/json">([\s\S]*?)<\/script>/);
  assert(scriptMatch, 'HTML contains embedded demo-data script');
  const embeddedData = JSON.parse(scriptMatch[1]);

  assert(JSON.stringify(embeddedData.mainModel) === JSON.stringify(fixtures['model_main.json']), 'mainModel: embedded matches fixture');
  assert(JSON.stringify(embeddedData.independentModel) === JSON.stringify(fixtures['model_independent.json']), 'independentModel: embedded matches fixture');
  assert(JSON.stringify(embeddedData.unavailableModel) === JSON.stringify(fixtures['model_unavailable.json']), 'unavailableModel: embedded matches fixture');
  assert(JSON.stringify(embeddedData.updateSequence) === JSON.stringify(fixtures['update_demo.json']), 'updateSequence: embedded matches fixture');
  assert(JSON.stringify(embeddedData.workbenchScenarios) === JSON.stringify(fixtures['workbench_scenarios.json']), 'workbenchScenarios: embedded matches fixture');

  results.push('PASS: No data drift between HTML embedded data and fixture files');

  // 8. Schema SHA validation
  const expectedSchemaSha = 'sha256-' + crypto.createHash('sha256').update(schemaContent).digest('hex').slice(0, 16);
  assert(embeddedData.schemaSha === expectedSchemaSha, 'schemaSha matches actual schema file hash');
  results.push('PASS: schemaSha is real SHA of schema file');

  // 9. Forbidden pattern scan in HTML
  assert(!html.includes('xMaxRaw'), 'HTML should not contain xMaxRaw');
  assert(!html.includes('eval('), 'HTML should not contain eval(');
  results.push('PASS: No forbidden patterns in HTML');

  // 10. JS syntax check
  const scriptMatch2 = html.match(/<script[^>]*>([\s\S]*?)<\/script>/g);
  const lastScript = scriptMatch2[scriptMatch2.length - 1].replace(/<script[^>]*>/g, '').replace(/<\/script>/g, '');
  const tmpPath = path.join(ROOT, 'docs/demo/_check_temp.js');
  fs.writeFileSync(tmpPath, lastScript, 'utf8');
  const { execSync } = require('child_process');
  try {
    execSync('node --check ' + tmpPath, { stdio: 'pipe' });
    results.push('PASS: Embedded JavaScript syntax valid');
  } catch (e) {
    throw new Error('JS syntax error: ' + e.message);
  } finally {
    fs.unlinkSync(tmpPath);
  }

  // 11. Check no gages in any fixture
  for (const file of jsonFiles) {
    const content = fs.readFileSync(path.join(FIXTURE_DIR, file), 'utf8');
    assert(!content.includes('"gages"'), file + ' should not contain gages field');
  }
  results.push('PASS: No gages field in any fixture');

  // 12. Check release_manifest.json is not self-referential
  const manifest = JSON.parse(fs.readFileSync(path.join(ROOT, 'release_manifest.json'), 'utf8'));
  assert(!manifest.frozen_files_sha256['release_manifest.json'], 'release_manifest.json should not be in its own frozen_files_sha256');
  results.push('PASS: release_manifest.json is not self-referential');

  // 13. Prefix cutoff line uses last prefix checkpoint, not xMin
  assert(html.includes('checkpoints[checkpoints.length - 1].day'), 'HTML boundaryX uses last prefix checkpoint day');
  results.push('PASS: Prefix cutoff line at last prefix checkpoint');

  // Output results
  console.log('\n=== VALIDATION RESULTS ===\n');
  for (const r of results) {
    console.log(r);
  }
  console.log('\nTotal: ' + results.length + ' checks');
  console.log('Failures: ' + failures);

  if (failures > 0) {
    console.log('\n=== VALIDATION FAILED ===\n');
    process.exit(1);
  } else {
    console.log('\n=== ALL VALIDATIONS PASSED ===\n');
    process.exit(0);
  }

} catch (error) {
  console.error('\n=== VALIDATION ERROR ===');
  console.error(error.message);
  process.exit(1);
}
