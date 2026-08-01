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

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup the basic layout
        self.setup_layout(
            "The Macroscopic Mystery", 
            [
                "Light changes speed and direction when entering a medium.", 
                "We call this phenomenon refraction.", 
                "The refractive index n is the ratio c over v.", 
                "Curiously, different colors bend at different angles.", 
                "This separation of light is known as dispersion."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        
        # Load rectangle asset as the medium
        medium = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/rectangle.svg")
        medium.set_color("#444444")
        medium.set_fill("#444444", opacity=0.5)
        self.place_in_area(medium, 'C1', 'F6')
        
        intersection_pt = self.grid['C3']
        incoming_start = self.grid['A1']
        outgoing_end = self.grid['E4']
        
        ray_in = Line(incoming_start, intersection_pt, color=WHITE)
        ray_out = Line(intersection_pt, outgoing_end, color=WHITE)
        
        self.play(Create(medium))
        self.play(Create(ray_in))
        self.play(Create(ray_out))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFD700")
        
        flash_circle = Circle(radius=0.15, color="#FFFF00", fill_opacity=0.8)
        flash_circle.move_to(intersection_pt)
        
        refraction_label = Text("Refraction", font_size=24, color="#FFD700")
        # Issue 28: Move label to B4 to avoid obstruction by white ray
        self.place_at_grid(refraction_label, 'B4')
        
        self.play(FadeIn(flash_circle, scale=0.5))
        self.play(Write(refraction_label))
        self.play(FadeOut(flash_circle))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        
        formula = Text("n = c / v", font_size=32, color="#00FF00", t2c={"c": YELLOW, "v": BLUE})
        # Issue 30: Use place_in_area for formula
        self.place_in_area(formula, 'A5', 'B6', scale_factor=0.8)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(WHITE)
        
        # Replace single ray with divergent colored rays
        red_ray = Line(intersection_pt, self.grid['D5'], color="#FF0000")
        blue_ray = Line(intersection_pt, self.grid['F4'], color="#0000FF")
        
        self.play(
            ReplacementTransform(ray_out, VGroup(red_ray, blue_ray))
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF69B4")
        
        dispersion_label = Text("Dispersion", font_size=24, color="#FF69B4")
        # Issue 29: Move label to E5 to prevent edge cut-off
        self.place_at_grid(dispersion_label, 'E5')
        
        self.play(Write(dispersion_label))
        self.wait(2)
