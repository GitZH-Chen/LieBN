from Network.SPDNetRBN import SPDNetLieBN,SPDNet


def get_model(args):
    if args.model_type in args.total_BN_model_types:
        model = SPDNetLieBN(args)
    elif args.model_type=='SPDNet':
        model = SPDNet(args)
    else:
        raise Exception('unknown model {} or metric {}'.format(args.model_type,args.metric))
    return model