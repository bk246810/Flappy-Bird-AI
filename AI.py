import pygame
import neat
import time
import os
import random
from pygame import mixer

mixer.init()
pygame.font.init()
pygame.mixer.pre_init(44100, -16, 2, 512)
mixer.music.load("airtone_-_forgottenland2.mp3")
mixer.music.play(-1)
mixer.music.play(-1)

#Sound
jump_sound = pygame.mixer.Sound("jump2.wav")
jump_sound.set_volume(0.3)
#VARS
WIN_WIDTH=500
WIN_HEIGHT=700

#IMAGES
bird1 = pygame.image.load("bird1.png")
bird1 = pygame.transform.scale2x(bird1)
bird2 = pygame.image.load("bird2.png")
bird2 = pygame.transform.scale2x(bird2)
bird3 = pygame.image.load("bird3.png")
bird3 = pygame.transform.scale2x(bird3)
pipe_img = pygame.image.load("pipe.png")
background_img = pygame.image.load("bg.png")
base_img = pygame.image.load("base.png")

BIRD_IMGS = [bird1,bird1,bird1,bird1,bird1,bird2,bird2,bird2,bird2,bird2,bird3,bird3,bird3,bird3,bird3]
PIPE_IMG = pygame.transform.scale2x(pipe_img)
BG_IMG = pygame.transform.scale(background_img,(500,800))
BASE_IMG = pygame.transform.scale2x(base_img)

stat_font = pygame.font.SysFont("comicsans",50)

#Bird Class
class Bird:
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25
    ROTATION_VEL = 20
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]
    def jump(self):
        self.vel = -9
        self.tick_count = 0
        self.height = self.y
        jump_sound.play()
    def move(self):
        self.tick_count +=0.2
        displacement = self.vel+1.5*self.tick_count**2
        if displacement >=10:
            displacement = 10
        if displacement<0:
            displacement-= 1
        self.y = self.y+displacement
        if displacement<0:
            if self.tilt<self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt >-70:
                self.tilt-= self.ROTATION_VEL
    def draw(self, window):
        if self.img_count<len(self.IMGS):
            self.img = self.IMGS[self.img_count]
            self.img_count += 1
        else:
            self.img_count = 0

        if self.tilt <=-80:
            self.img_count= 6

        rotated_image = pygame.transform.rotate(self.img,self.tilt)
        new_rect = rotated_image.get_rect(center = self.img.get_rect(topleft = (self.x,self.y)).center)
        window.blit(rotated_image,new_rect.topleft)
    def get_mask(self):
        return pygame.mask.from_surface(self.img)

#Pipe Class
class Pipe:
    GAP = 140
    VEL = 5

    def __init__(self,x):
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG,False,True)
        self.PIPE_BOTTOM = PIPE_IMG

        self.passedPipe = False

        self.set_height()
    def set_height(self):
        self.height = random.randrange(25,400)
        self.top = self.height-self.PIPE_TOP.get_height()
        self.bottom = self.height+self.GAP
    def move(self):
        self.x-=self.VEL
    def draw(self,window):
        window.blit(self.PIPE_TOP,(self.x,self.top))
        window.blit(self.PIPE_BOTTOM,(self.x,self.bottom))
    def collide(self,bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x-bird.x,self.top-round(bird.y))
        bottom_offset = (self.x-bird.x,self.bottom-round(bird.y))

        bottom_collision_point = bird_mask.overlap(bottom_mask,bottom_offset)
        top_collision_point = bird_mask.overlap(top_mask, top_offset)

        if top_collision_point or bottom_collision_point:
            return True
        return False

#Base class
class Base:
    VEL = 5
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG
    def __init__(self,y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH
    def move(self):
        self.x1-= self.VEL
        self.x2 -= self.VEL

        if self.x1+self.WIDTH<0:
            self.x1 = self.x2+self.WIDTH
        if self.x2 +self.WIDTH<0:
            self.x2 = self.x1+self.WIDTH
    def draw(self,window):
        window.blit(BASE_IMG,(self.x1,self.y))
        window.blit(BASE_IMG,(self.x2,self.y))

def draw_window(window,birds,pipes,base,score):
    window.blit(BG_IMG,(0,0))
    for pipe in pipes:
        pipe.draw(window)
    text = stat_font.render("Score: "+str(score),1,(255,255,255))
    window.blit(text,(WIN_WIDTH-10-text.get_width(),10))
    base.draw(window)
    for bird in birds:
        bird.draw(window)

    pygame.display.update()

def main(genomes,config):
    nets = []
    ge = []
    birds = []

    for _,g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g,config)
        nets.append(net)
        birds.append(Bird(230,250))
        g.fitness = 0
        ge.append((g))

    base = Base(630)
    pipes = [Pipe(600)]
    screen = pygame.display.set_mode((WIN_WIDTH,WIN_HEIGHT))
    clock = pygame.time.Clock()
    running = True
    endscreen = False
    score = 0

    while running:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                quit()

        pipe_index = 0
        if len(birds)>0:
            if len(pipes)>1 and birds[0].x > pipes[0].x +int(pipes[0].PIPE_TOP.get_width()):
                pipe_index = 1
        else:
            break

        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness+= 0.1

            output = nets[x].activate((bird.y,abs(bird.y -pipes[pipe_index].height),abs(bird.y-pipes[pipe_index].bottom)))
            if output[0]>0.5:
                bird.jump()

        remove_pipes  = []
        add_Pipe = False
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    birds.pop(x)
                    ge.pop(x)
                    nets.pop(x)

                if not pipe.passedPipe and pipe.x <bird.x:
                    pipe.passedPipe = True
                    add_Pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                remove_pipes.append(pipe)

            pipe.move()

        if add_Pipe:
            score+=1
            for g in ge:
                g.fitness+=5
            pipes.append(Pipe(600))
        for i in remove_pipes:
            pipes.remove(i)
        for x,bird in enumerate(birds):
            if bird.y +bird.img.get_height()>=630 or bird.y <0:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        bird.move()
        base.move()
        draw_window(screen,birds,pipes,base,score)

def run(config_path):
    config=neat.config.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,neat.DefaultStagnation,config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main,50)
if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir,'Neat config.txt')
    run(config_path)