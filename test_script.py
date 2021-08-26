import gin
import torch
from network.backbone import build_backbone
# from network.transformer_full import build_transformer
from network.transformer_decoder import build_transformer_decoder
#from network.quester import build
# from network.query_generate import build_query
from network.query_generate_dct_phoc import encode_input_string

config_file = "./config/ic15.gin"

def gin_value_getter(n):

    return gin.query_parameter(f'%{n}')

def init_gin():

    gin.parse_config_file(config_file)


def test_backbone():

    model = build_backbone()

    x = torch.rand((2, 1, 550, 550), requires_grad=True)
    print(x.shape)

    output = model(x)

    # N, 1100, 80
    print(output[0].shape) # 1, 1100, 80
    print(output[1].shape) # 1, 1100, 80


def test_transformer():

    model = build_transformer_decoder()
    # input :(src, tgt, query_embed, pos_embed)
    src = torch.rand((2, 1100, 80), requires_grad=True)
    tgt = torch.rand((2, 100, 200, 80))
    query_embed = torch.rand((100, 200, 80))
    pos_embed = torch.rand((2, 1100, 80))

    output = model(src, tgt, query_embed, pos_embed)

    print(output.shape) # ([1, 1, 100, 80])
    # print(output[1].shape) # ([1, 80, 1100])

def test_quester():
    pass


def test_query_gen():

    # query_gen = build_query()

    input = ["hello;:world", "nice;:to;:meet;:you", "sdfa;:sa", "asdf", "asdfasfdfasdfas"]

    # out = query_gen(input, 10)
    out = encode_input_stringi(input)
    print(out[0].shape)
    print(out[1].shape)




if __name__ == "__main__":

    init_gin()

    # test_backbone()
    test_transformer()
    test_query_gen()
