#!/usr/bin/env bash
# Roll the dark-deploy service back to the deployment recorded before the
# release, then verify the restored code, campaign state and write gate.
# Run by the dark-deploy workflow; its step env lists the inputs. Earlier
# steps leave previous-deployments.json and railway-up.log in the work dir.
set -eo pipefail
umask 077
work_dir=${DARK_DEPLOY_WORK_DIR:-/tmp}
# The response bodies are Railway errors or booleans, never the token.
railway_graphql() {
  http_status=$(curl --silent --show-error \
    --request POST \
    --url https://backboard.railway.com/graphql/v2 \
    --header "Project-Access-Token: ${RAILWAY_TOKEN}" \
    --header "Content-Type: application/json" \
    --data-binary @"$1" \
    --output "$2" \
    --write-out '%{http_code}')
  if [ "$http_status" != "200" ] || ! jq -e '.errors == null' "$2" > /dev/null 2>&1; then
    echo "Railway GraphQL request failed with HTTP ${http_status}:"
    jq -c '{errors}' "$2" 2>/dev/null || head -c 2000 "$2"
    echo
    return 1
  fi
}
list_deployments() {
  railway deployment list --json --limit 50 --service "$RAILWAY_SERVICE" --environment "$RAILWAY_ENVIRONMENT" > "$1"
  jq -c '[.[:5][] | {id, status, createdAt, cliMessage: .meta.cliMessage}]' "$1"
}

# Every deployment that is not stopped yet, as {id, status}; an id
# missing from the list is not known to be stopped.
unstopped() {
  jq -c --argjson dark "$dark_ids" '
    . as $list
    | [$dark[] as $id
        | ($list | map(select(.id == $id)) | .[0] // {id: $id, status: "NOT_LISTED"})
        | {id, status}
        | select(.status | IN("REMOVED", "FAILED", "CRASHED", "SKIPPED") | not)]
  ' "$1"
}

list_deployments "$work_dir/pre-rollback-deployments.json"
# A cancelled or timed-out upload step never wrote its id output, but
# tee already wrote the Build Logs line. Any upload of this release
# that the recorded list lacks is also a dark deployment.
if [ -z "$DARK_DEPLOYMENT_ID" ] && [ -f "$work_dir/railway-up.log" ]; then
  DARK_DEPLOYMENT_ID=$(grep -oE '[?&]id=[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' "$work_dir/railway-up.log" | head -n 1 | cut -d= -f2 || true)
fi
dark_ids=$(jq -c \
  --arg id "$DARK_DEPLOYMENT_ID" \
  --arg message "dark release ${RELEASE_SHA}" \
  --slurpfile recorded "$work_dir/previous-deployments.json" '
    ($recorded[0] | map(.id)) as $known
    | [.[] | select(.meta.cliMessage == $message) | .id
        | select(. as $d | $known | index($d) | not)]
      + (if $id == "" then [] else [$id] end)
    | unique
  ' "$work_dir/pre-rollback-deployments.json")
echo "Dark deployments: ${dark_ids}"
for dark_id in $(jq -r '.[]' <<< "$dark_ids"); do
  dark_status=$(jq -r --arg id "$dark_id" '.[] | select(.id == $id) | .status' "$work_dir/pre-rollback-deployments.json")
  case "$dark_status" in
    QUEUED|WAITING|NEEDS_APPROVAL|INITIALIZING|BUILDING|DEPLOYING)
      # An in-flight dark deployment could still pass its health
      # check after the rollback and take traffic back.
      jq -n --arg id "$dark_id" \
        '{query: "mutation deploymentCancel($id: String!) { deploymentCancel(id: $id) }", variables: {id: $id}}' \
        > "$work_dir/cancel-request.json"
      if railway_graphql "$work_dir/cancel-request.json" "$work_dir/cancel.json" \
        && jq -e '.data.deploymentCancel == true' "$work_dir/cancel.json" > /dev/null; then
        echo "Cancelled in-flight dark deployment ${dark_id} (${dark_status})."
      else
        # Still roll back; success below requires it to have stopped.
        echo "::warning::Could not cancel dark deployment ${dark_id} (${dark_status})."
      fi
      ;;
  esac
done

jq -n \
  --rawfile query "$(dirname "$0")/railway-rollback.graphql" \
  --arg id "$PREVIOUS_DEPLOYMENT_ID" \
  '{query: $query, variables: {id: $id}}' \
  > "$work_dir/rollback-request.json"
railway_graphql "$work_dir/rollback-request.json" "$work_dir/rollback.json"
# deploymentRollback returns Boolean!, not a deployment.
jq -e '.data.deploymentRollback == true' "$work_dir/rollback.json" > /dev/null

rollback_id=""
running=""
for attempt in $(seq 1 "$ROLLBACK_POLL_ATTEMPTS"); do
  echo "Rollback poll ${attempt}/${ROLLBACK_POLL_ATTEMPTS}:"
  list_deployments "$work_dir/rollback-deployments.json"
  if [ -z "$rollback_id" ]; then
    rollback_id=$(jq -r \
      --slurpfile before "$work_dir/pre-rollback-deployments.json" \
      --argjson dark "$dark_ids" '
        (($before[0] | map(.id)) + $dark) as $known
        | [.[] | select(.id as $id | $known | index($id) | not)] as $new
        | if ($new | length) == 1 then $new[0].id
          elif ($new | length) > 1 then "AMBIGUOUS"
          else ""
          end
      ' "$work_dir/rollback-deployments.json")
    if [ "$rollback_id" = "AMBIGUOUS" ]; then
      echo "AMBIGUOUS: more than one deployment appeared after the rollback request."
      exit 1
    fi
    # If Railway reactivates the recorded deployment instead of
    # creating a new one, accept it only after a grace period.
    if [ -z "$rollback_id" ] && [ "$attempt" -gt "$ROLLBACK_NEW_DEPLOYMENT_GRACE_POLLS" ]; then
      previous_status=$(jq -r --arg id "$PREVIOUS_DEPLOYMENT_ID" '.[] | select(.id == $id) | .status' "$work_dir/rollback-deployments.json")
      if [ "$previous_status" = "SUCCESS" ]; then
        rollback_id="$PREVIOUS_DEPLOYMENT_ID"
      fi
    fi
  fi
  status=$(jq -r --arg id "$rollback_id" '.[] | select(.id == $id) | .status' "$work_dir/rollback-deployments.json")
  case "$status" in
    SUCCESS)
      # A dark deployment that has not stopped can still take
      # traffic back, so the rollback is not done until it has.
      running=$(unstopped "$work_dir/rollback-deployments.json")
      if [ "$running" != "[]" ]; then
        echo "Restored deployment ${rollback_id} is up; waiting for dark deployments to stop: ${running}"
        sleep "$DEPLOYMENT_POLL_INTERVAL_SEC"
        continue
      fi
      echo "id=$rollback_id" >> "$GITHUB_OUTPUT"
      curl --fail --silent --show-error "${BASE_URL}/api/health" > "$work_dir/rollback-health.json"
      jq -e '.ok == true' "$work_dir/rollback-health.json" > /dev/null
      # Railway records no commit for `railway up` uploads, so the
      # public release SHA is the restored-code check.
      curl --fail --silent --show-error "${BASE_URL}/api/release" > "$work_dir/rollback-release.json"
      jq -e --arg sha "$PREVIOUS_SHA" '.git_sha == $sha' "$work_dir/rollback-release.json" > /dev/null
      curl --fail --silent --show-error \
        "${BASE_URL}/api/paper-trades/summary" \
        > "$work_dir/rollback-paper-summary.json"
      jq -e '.ok == true' "$work_dir/rollback-paper-summary.json" > /dev/null
      observed_status=$(jq -r '.campaign_health.status // "absent"' "$work_dir/rollback-paper-summary.json")
      if [ "$EXPECTED_CAMPAIGN_STATUS" = "absent" ]; then
        test "$observed_status" = "draft"
      else
        test "$observed_status" = "$EXPECTED_CAMPAIGN_STATUS"
      fi
      observed_write_state=$(jq -er '.campaign_health.write_state | select(. == "enabled" or . == "paused")' "$work_dir/rollback-paper-summary.json")
      test "$observed_write_state" = "$EXPECTED_CAMPAIGN_WRITE_STATE"
      railway variable list --json --service "$RAILWAY_SERVICE" --environment "$RAILWAY_ENVIRONMENT" > "$work_dir/rollback-variables.json"
      observed_gate=$(jq -er '.TRADER_KOO_PAPER_TRADE_ENABLED | select(. == "0" or . == "1")' "$work_dir/rollback-variables.json")
      test "$observed_gate" = "$EXPECTED_PAPER_TRADE_ENABLED"
      echo "observed_campaign_status=$observed_status" >> "$GITHUB_OUTPUT"
      echo "observed_write_state=$observed_write_state" >> "$GITHUB_OUTPUT"
      echo "paper_trade_enabled=$observed_gate" >> "$GITHUB_OUTPUT"
      exit 0
      ;;
    FAILED|CRASHED|REMOVED|REMOVING|SKIPPED)
      echo "Restored deployment ${rollback_id} ended ${status}."
      exit 1
      ;;
  esac
  sleep "$DEPLOYMENT_POLL_INTERVAL_SEC"
done
if [ "$status" = "SUCCESS" ]; then
  echo "Dark deployments still running after ${ROLLBACK_POLL_ATTEMPTS} polls: ${running}"
else
  echo "Restored deployment ${rollback_id:-unidentified} did not reach SUCCESS within ${ROLLBACK_POLL_ATTEMPTS} polls."
fi
exit 1
