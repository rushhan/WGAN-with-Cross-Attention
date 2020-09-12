import torch
import torch.nn as nn
import torch.nn.parallel

class DCGAN_D(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN_D, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial:{0}-{1}:conv'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial:{0}:relu'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid:{0}-{1}:conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid:{0}:relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        #main.add_module('final:{0}-{1}:conv'.format(cndf, 1),
        #                nn.Conv2d(cndf, cndf, 4, 1, 0, bias=False))

        # first check the dimesion for the cndf as well as each layer in the disc and gen
        self.main = main
        #self.linear_1 = nn.Linear(cndf, nz)
        #self.relu1 = nn.ReLU(True)
        #self.norm1 = nn.BatchNorm1d(nz)

        #self.linear_2 = nn.Linear(nz,nz)

        #self.linear_3 = nn.Linear(nz,1)

        self.final= nn.Conv2d(cndf, cndf, 4, 1, 0, bias=False)
        #self.final_norm = nn.BatchNorm2d(cndf)
        #self.final_relu = nn.LeakyReLU(0.2, inplace=True)

        self.final_2= nn.Linear(cndf,nz)
        #self.final_2_norm = nn.BatchNorm1d(nz)
        #self.final_2_relu = nn.ReLU(True)
        

        self.final_1= nn.Linear(nz,1)



        #self.to_linear = nn.Conv2d(cndf, 1, 4, 1, 0, bias=False)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else: 
            output = self.main(input)
        output = self.final(output)
        output = torch.squeeze(output)
        #output = self.norm1(self.relu1(self.linear_1(output)))
        fd = self.final_2(output)
        out_fin = self.final_1(fd)

        out_fin = out_fin.mean(0)
        return out_fin.view(1),fd

class reshape(nn.Module):
    """docstring for reshape"""
    def __init__(self):
        super(reshape, self).__init__()
    def forward(self,input):
        #print (input.size())
        a,b = input.size()
        #print (a,b,1,1)
        return  input.view(a,b,1,1)
        
class normalz(nn.Module):
    """docstring for normalz"""
    def __init__(self):
        super(normalz, self).__init__()
        

    def forward(self,input):
        return nn.functional.normalize(input)
        
class DCGAN_G(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu,view, n_extra_layers=0):
        super(DCGAN_G, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf//2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2
        
        self.layer_0 =nn.Linear(nz, nz)
        #main.add_module('layer_0',nn.Linear(nz, nz).clamp(min=-1,max =1)) # this will also work for clmaping
        self.linear_norm_0=nn.BatchNorm1d(nz)
        self.linear_ac_0=nn.ReLU(True)


        #feedback layer\

        self.query = nn.Linear(nz,nz)
        self.query_norm = nn.BatchNorm1d(nz)
        self.query_act = nn.ReLU(True)

        self.key = nn.Linear(nz,nz)
        self.key_norm = nn.BatchNorm1d(nz)
        self.key_act = nn.ReLU(True)

        self.value = nn.Linear(nz,nz)
        self.value_norm = nn.BatchNorm1d(nz)
        self.value_act = nn.ReLU(True)

        self.fin = nn.Linear(nz,nz)
        self.fin_norm= nn.BatchNorm1d(nz)
        self.fin_act =nn.ReLU(True)

        self.softmax = nn.Softmax(dim=-1)
        self.layer_1=nn.Linear(nz, nz)
        self.normalize_distribution=normalz()
        self.rehape_0=reshape()

        main = nn.Sequential()
        # input is Z, going into a convolution
        #main.add_module('layer_0',
        #                nn.Linear(nz, nz)
        

        main.add_module('initial:{0}-{1}:convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial:{0}:batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial:{0}:relu'.format(cngf),
                        nn.ReLU(True))

        csize, cndf = 4, cngf
        while csize < isize//2:
            main.add_module('pyramid:{0}-{1}:convt'.format(cngf, cngf//2),
                            nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:batchnorm'.format(cngf//2),
                            nn.BatchNorm2d(cngf//2))
            main.add_module('pyramid:{0}:relu'.format(cngf//2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final:{0}-{1}:convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final:{0}:tanh'.format(nc),
                        nn.Tanh())
        self.main = main


    def forward(self, input,fdback=None,use_fdback=False):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            if not use_fdback:
                in1 = input
                in1 = self.layer_0(in1)
                in1 = self.linear_norm_0(in1)
                in1 = self.linear_ac_0(in1)
                in1 = self.layer_1(in1)
                in1=self.normalize_distribution(in1)
                in1 = self.rehape_0(in1)
                output = self.main(in1)

            else:
                in1 = input
                in1 = self.layer_0(in1)
                in1 = self.linear_norm_0(in1)
                in1 = self.linear_ac_0(in1)
                
                query = self.query_act(self.query_norm(self.query(in1))).unsqueeze(-1).permute(0,2,1)

                key = self.key_act(self.key_norm(self.key(fdback))).unsqueeze(-1)

                #print (key.shape,query.shape)
                mult = (torch.bmm(key,query)) ## should the order be reversed, but the output will be single tensor

                value = self.value_act(self.value_norm(self.value(fdback))).unsqueeze(-1)

                #print (mult.shape,value.shape)
                feedback= torch.bmm(mult,value).squeeze(dim=-1)


                #fin = self.fin(feedback)
                #fin = self.fin_norm(fin)
                #fin = self.fin_act(fin)

                in1 = feedback +in1

                in1 = self.layer_1(in1)
                #in1=self.normalize_distribution(in1)
                in1 = self.rehape_0(in1)
                output = self.main(in1)                
        return output 
###############################################################################
class DCGAN_D_nobn(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN_D_nobn, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        # input is nc x isize x isize
        main.add_module('initial:{0}-{1}:conv'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial:{0}:conv'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid:{0}-{1}:conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module('final:{0}-{1}:conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main


    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else: 
            output = self.main(input)
            
        output = output.mean(0)
        return output.view(1)

class DCGAN_G_nobn(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN_G_nobn, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf//2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        main.add_module('initial:{0}-{1}:convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial:{0}:relu'.format(cngf),
                        nn.ReLU(True))

        csize, cndf = 4, cngf
        while csize < isize//2:
            main.add_module('pyramid:{0}-{1}:convt'.format(cngf, cngf//2),
                            nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:relu'.format(cngf//2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final:{0}-{1}:convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final:{0}:tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,  range(self.ngpu))
        else: 
            output = self.main(input)
        return output 
