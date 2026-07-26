from model.troika_cfm import Troika as Troika_cfm


def get_model(config, attributes, classes, offset):
    if config.model_name == 'troika_cfm':
        model = Troika_cfm(config, attributes=attributes, classes=classes, offset=offset)
    else:
        raise NotImplementedError(
            "Error: Unrecognized Model Name {:s}.".format(
                config.model_name
            )
        )


    return model
