def generate_mix_explanation(feature_map):
    binder = feature_map["Binder"]
    wb_ratio = feature_map["WBRatio"]
    age = feature_map["AGE"] if "AGE" in feature_map else feature_map["age"]
    scm_content = feature_map["SCMContent"]
    admixture = feature_map["admixture"] if "admixture" in feature_map else feature_map["Admixture"]
    total_aggregate = feature_map["TotalAggregate"]

    positive_points = []
    caution_points = []

    # Binder effect
    if binder >= 350:
        positive_points.append("higher binder content improved the expected strength")
    elif binder >= 280:
        positive_points.append("moderate binder content supported strength development")
    else:
        caution_points.append("lower binder content may limit strength gain")

    # Water-to-binder ratio effect
    if 0.20 <= wb_ratio <= 0.40:
        positive_points.append("the water-to-binder ratio is favorable for stronger concrete")
    elif 0.40 < wb_ratio <= 0.55:
        positive_points.append("the water-to-binder ratio remains within an acceptable range")
    else:
        caution_points.append(
            f"the water-to-binder ratio is {wb_ratio:.3f}, which is outside the recommended 0.20–0.55 range"
        )

    # Age effect
    if age >= 28:
        positive_points.append("the curing age supports mature strength development")
    elif age >= 7:
        positive_points.append("the curing age contributes to ongoing strength gain")
    else:
        caution_points.append("early-age concrete may not yet reach full strength")

    # SCM effect
    if scm_content > 0:
        positive_points.append("supplementary cementitious materials support sustainability and mix efficiency")

    # Admixture effect
    if admixture > 0:
        positive_points.append("admixture dosage helps improve mix workability and control")

    # Aggregate effect
    if total_aggregate > 0:
        positive_points.append("aggregate proportion contributes to structural stability")

    explanation = []

    if positive_points:
        explanation.append("Key strength drivers in this mix are " + ", ".join(positive_points[:3]) + ".")

    if caution_points:
        explanation.append("A point to watch is that " + ", ".join(caution_points[:2]) + ".")

    if not explanation:
        return "The prediction is driven by the combined effect of binder content, water ratio, curing age, and aggregate proportions."

    return " ".join(explanation)