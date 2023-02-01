import json

# create causes_en.csv from ConceptNet (conceptnet-assertions-5.7.0.csv.gz) data:
# grep "/r/Causes/" ./assertions.csv > causes.csv
# grep "/c/en/" ./causes.csv > causes_en.csv
# python extract.py

source_file_name = "./causes_en.csv"
filter_file_name = "./conceptnet_cause_sets.txt"

with open(source_file_name, "r") as f:
    lines = [line.rstrip() for line in f]

# "/a/[/r/Causes/,/c/en/acting_in_play/,/c/en/attention/]	/r/Causes	/c/en/acting_in_play	/c/en/attention ...


def sanitize(s):
  if s[:6] == "/c/en/":
    s = s[6:]
  if s[-1] == "/":
    s = s[:-1]
  return s


causes = {}
for line in lines:
  line = line.split("]")[0].split("[")[1]
  try:
    cause, effect = line.split(",")[1:]
  except:
    print(f"skipping (1): {line}")
    continue
  
  cause = sanitize(cause)
  effect = sanitize(effect)
  
  if "/" in cause: # catch "/v/wn/" "/n/wn/"
    #cause_o = cause
    #cause = cause.split("/",1)[0]
    #print(cause_o,"->", cause)
    print(f"skipping (2): {line}")
    continue
  #if "/" in effect:
  #  effect_o = effect
  #  effect = effect.split("/",1)[0]
  #  print("|", effect_o,"->", effect)
  
  if not cause in causes:
    container = []
    causes[cause] = container
  else:
    container = causes[cause]
  
  container.append(effect)

causes_count = 0
total_count = 0
with open(filter_file_name, "w+") as f:
  for cause, effects in causes.items():
    causes_count += 1
    total_count += len(effects)
    f.write(f"{cause},{json.dumps(effects)}\n")

print(f"found {causes_count} causes")
print(f"prepared {total_count} cause-effect pairs")
print("done.")
