from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section6Scene(TeachingScene):
    def construct(self):
        # SETUP
        title = "Real-World Application: The Rainbow Effect"
        lines = [
            "Prisms split white light by bending colors differently.",
            "This dispersion creates the vibrant spectrum of a rainbow.",
            "Lenses use multiple glass types to correct color blur."
        ]
        self.setup_layout(title, lines)
        
        rainbow_colors = ["#FF0000", "#FFA500", "#FFFF00", "#008000", "#0000FF", "#4B0082", "#EE82EE"]

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Prism Asset
        prism = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/prism.svg")
        prism.set_color(WHITE)
        self.place_in_area(prism, "B2", "D4", scale_factor=1.5)
        
        # Rays
        # Incoming white ray
        start_pt = self.grid["C1"] + LEFT * 1.5
        hit_pt = prism.get_left() + RIGHT * 0.1
        incoming_ray = Line(start_pt, hit_pt, color=WHITE, stroke_width=3)
        
        # Dispersing rays inside and outside
        exit_rays = VGroup()
        for i, color in enumerate(rainbow_colors):
            # Internal spread
            target_exit = prism.get_right() + UP * (0.3 - i * 0.1)
            # External ray
            final_target = target_exit + (RIGHT * 1.5 + DOWN * (0.5 - i * 0.15))
            ray = Line(hit_pt, target_exit, color=color, stroke_width=2)
            ray_exit = Line(target_exit, final_target, color=color, stroke_width=2)
            exit_rays.add(VGroup(ray, ray_exit))

        self.play(Create(incoming_ray))
        self.play(FadeIn(prism))
        self.play(Create(exit_rays), run_time=2)
        self.wait(1)

        # Cleanup for next stage
        self.play(FadeOut(incoming_ray), FadeOut(prism), FadeOut(exit_rays), self.lecture[0].animate.set_color(WHITE))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE_C)
        
        # Raindrop Asset
        raindrop = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/ra.svg")
        raindrop.set_color("#ADD8E6")
        self.place_in_area(raindrop, "B2", "E4", scale_factor=2.0)
        
        # Raindrop Ray logic
        drop_center = raindrop.get_center()
        entry_pt = drop_center + LEFT * 0.8 + UP * 0.4
        reflection_pt = drop_center + RIGHT * 0.8
        
        sun_ray = Line(entry_pt + LEFT * 2 + UP * 0.5, entry_pt, color=WHITE)
        
        internal_rainbow = VGroup()
        external_rainbow = VGroup()
        
        for i, color in enumerate(rainbow_colors):
            # Simple simulation of reflection and exit
            bend_offset = i * 0.05
            exit_pt = drop_center + LEFT * 0.7 + DOWN * (0.4 + bend_offset)
            final_pt = exit_pt + LEFT * 1.5 + DOWN * (0.3 + bend_offset)
            
            refracted = Line(entry_pt, reflection_pt, color=color, stroke_width=1.5)
            reflected = Line(reflection_pt, exit_pt, color=color, stroke_width=1.5)
            exited = Line(exit_pt, final_pt, color=color, stroke_width=1.5)
            
            internal_rainbow.add(VGroup(refracted, reflected))
            external_rainbow.add(exited)

        self.play(FadeIn(raindrop))
        self.play(Create(sun_ray))
        self.play(Create(internal_rainbow), run_time=1.5)
        self.play(Create(external_rainbow), run_time=1.5)
        self.wait(1)
        
        # Cleanup
        self.play(FadeOut(raindrop), FadeOut(sun_ray), FadeOut(internal_rainbow), FadeOut(external_rainbow), self.lecture[1].animate.set_color(WHITE))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(ORANGE)

        # 1. Chromatic Aberration Diagram
        lens_1 = Intersection(Circle(radius=1.5), Circle(radius=1.5).shift(RIGHT*1.8), color=WHITE, fill_opacity=0.2)
        self.place_in_area(lens_1, "A2", "C4", scale_factor=0.3)
        label_1 = Text("Chromatic Aberration", font_size=16, color=WHITE)
        self.place_at_grid(label_1, "A3", scale_factor=1)
        
        # Blue vs Red rays
        ray_y = 0.2
        blue_focus = lens_1.get_center() + RIGHT * 1.0
        red_focus = lens_1.get_center() + RIGHT * 1.5
        
        rays_aberration = VGroup()
        for sign in [1, -1]:
            # Blue
            rays_aberration.add(Line(lens_1.get_center() + LEFT*1.5 + UP*sign*ray_y, lens_1.get_center() + UP*sign*ray_y, color=WHITE))
            rays_aberration.add(Line(lens_1.get_center() + UP*sign*ray_y, blue_focus, color=BLUE))
            # Red
            rays_aberration.add(Line(lens_1.get_center() + UP*sign*ray_y, red_focus, color=RED))

        # 2. Achromatic Doublet Diagram
        lens_2a = Intersection(Circle(radius=1.5), Circle(radius=1.5).shift(RIGHT*1.8), color=WHITE, fill_opacity=0.3)
        lens_2b = Difference(Circle(radius=1.5).shift(LEFT*0.1), Circle(radius=1.5).shift(RIGHT*0.5), color=BLUE_E, fill_opacity=0.3)
        lens_doublet = VGroup(lens_2a, lens_2b).arrange(RIGHT, buff=0)
        self.place_in_area(lens_doublet, "D2", "F4", scale_factor=0.3)
        label_2 = Text("Achromatic Doublet", font_size=16, color=WHITE)
        self.place_at_grid(label_2, "D3", scale_factor=1)

        shared_focus = lens_doublet.get_center() + RIGHT * 1.2
        rays_corrected = VGroup()
        for sign in [1, -1]:
            rays_corrected.add(Line(lens_doublet.get_center() + LEFT*1.5 + UP*sign*ray_y, lens_doublet.get_center() + UP*sign*ray_y, color=WHITE))
            rays_corrected.add(Line(lens_doublet.get_center() + UP*sign*ray_y, shared_focus, color=BLUE))
            rays_corrected.add(Line(lens_doublet.get_center() + UP*sign*ray_y, shared_focus, color=RED))

        self.play(FadeIn(lens_1), FadeIn(label_1))
        self.play(Create(rays_aberration))
        self.wait(0.5)
        self.play(FadeIn(lens_doublet), FadeIn(label_2))
        self.play(Create(rays_corrected))
        
        self.wait(2)
