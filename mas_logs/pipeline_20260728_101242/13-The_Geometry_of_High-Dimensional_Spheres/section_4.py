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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "Surface Concentration: The 'Thin Crust' Effect",
            [
                "High-dimensional volume concentrates near the surface.",
                "The interior of the sphere is mostly empty.",
                "Imagine peeling a hundred-dimensional orange.",
                "The thin peel contains nearly all the mass.",
                "There is no 'meat' in a high-D orange."
            ]
        )

        # Colors for lecture lines and elements
        c1 = "#FFFFFF" # White
        c2 = "#FF4500" # OrangeRed
        c3 = "#FFA500" # Orange
        c4 = "#FFFFFF" # White
        c5 = "#FF0000" # Red

        # === Animation for Lecture Line 1 ===
        # "High-dimensional volume concentrates near the surface."
        # "Draw a white unit circle (#FFFFFF) and a smaller concentric circle (#FF4500) at radius 0.95."
        self.play(self.lecture[0].animate.set_color(c1))
        
        outer_circle = Circle(radius=2, color=c1, stroke_width=2)
        inner_circle = Circle(radius=2*0.95, color=c2, stroke_width=2)
        circles_group = VGroup(outer_circle, inner_circle)
        self.place_in_area(circles_group, "B2", "E5")
        
        self.play(Create(outer_circle), Create(inner_circle))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "The interior of the sphere is mostly empty."
        # "Gradually fade the center area of the circle to #000000 to show the 'empty' interior."
        self.play(self.lecture[1].animate.set_color(c2))
        
        center_fill = Circle(radius=2*0.95, fill_color=BLACK, fill_opacity=0, stroke_width=0).move_to(inner_circle)
        self.add(center_fill)
        self.play(center_fill.animate.set_fill(opacity=0.9), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Imagine peeling a hundred-dimensional orange."
        # "Fill the region between the circles with a bright orange color (#FFA500) to represent the 'crust' and incorporate the orange icon."
        self.play(self.lecture[2].animate.set_color(c3))
        
        crust = Annulus(inner_radius=2*0.95, outer_radius=2, fill_color=c3, fill_opacity=0.8, stroke_width=0)
        crust.move_to(outer_circle)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/orange.svg]
        orange_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/orange.svg")
        self.place_at_grid(orange_icon, "A5", scale_factor=0.6)
        
        self.play(FadeIn(crust), FadeIn(orange_icon))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "The thin peel contains nearly all the mass."
        # "Animate an arrow (#FFFFFF) pointing to the orange crust with the text 'Mass Concentration' in #FFFFFF."
        self.play(self.lecture[3].animate.set_color(c4))
        
        mass_label = Text("Mass Concentration", font_size=18, color=c4)
        # Fix: Issue 31 - place_in_area('F3', 'F4')
        self.place_in_area(mass_label, 'F3', 'F4', scale_factor=1.0)
        
        # Arrow pointing from label area to the crust
        arrow = Arrow(start=mass_label.get_top(), end=crust.get_bottom(), color=c4, buff=0.1)
        
        self.play(GrowArrow(arrow), Write(mass_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "There is no 'meat' in a high-D orange."
        # "Display the text 'No Meat at Center' in red (#FF0000) along with the meat icon and then make it disappear."
        self.play(self.lecture[4].animate.set_color(c5))
        
        no_meat_text = Text("No Meat at Center", font_size=20, color=c5)
        # Fix: Issue 30 - place_in_area('C3', 'D4')
        self.place_in_area(no_meat_text, 'C3', 'D4', scale_factor=0.8)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/meat.svg]
        meat_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/meat.svg")
        meat_icon.scale(0.5).next_to(no_meat_text, DOWN, buff=0.2)
        
        meat_group = VGroup(no_meat_text, meat_icon)
        
        self.play(Write(no_meat_text), FadeIn(meat_icon))
        self.wait(2)
        self.play(FadeOut(meat_group))
        self.wait(1)
