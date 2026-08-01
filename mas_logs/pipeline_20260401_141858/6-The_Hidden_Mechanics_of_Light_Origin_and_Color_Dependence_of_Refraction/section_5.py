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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "Real-World Application: The Rainbow and the Mantis Shrimp", 
            [
                "Dispersion creates rainbows as sunlight splits in droplets.", 
                "Prisms visualize the full spectrum of visible light.", 
                "Mantis shrimp eyes use specialized filters for hyperspectral vision."
            ]
        )
        
        # Define Colors for consistency
        RAINBOW_COLORS = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE]
        DROPLET_COLOR = "#00FFFF"
        EYE_COLOR = "#00FF00"

        # === Animation for Lecture Line 1 ===
        # Rainbow in a droplet
        self.play(self.lecture[0].animate.set_color(DROPLET_COLOR))
        
        droplet = Circle(radius=1.2, color=DROPLET_COLOR, stroke_width=4, fill_opacity=0.1)
        # Issue 43: Adjusted placement to avoid lecture text
        self.place_in_area(droplet, 'B3', 'D6', scale_factor=0.7)
        
        # Ray start point adjusted to avoid lecture text overlap
        ray_in = Line(start=self.grid["B2"] + LEFT * 0.5, end=droplet.get_left() + UP * 0.4, color=WHITE)
        # Inside droplet representation
        refract_in = Line(start=ray_in.get_end(), end=droplet.get_right() + DOWN * 0.2, color=WHITE)
        
        rainbow_rays = VGroup()
        for i, color in enumerate(RAINBOW_COLORS):
            # Light reflects off back and exits front lower part
            r = Line(
                start=refract_in.get_end(), 
                end=self.grid["D6"] + DOWN * (i * 0.15), 
                color=color
            )
            rainbow_rays.add(r)
            
        self.play(Create(droplet))
        self.play(Create(ray_in))
        self.play(Create(refract_in))
        self.play(Create(rainbow_rays))
        self.wait(1)
        
        self.play(FadeOut(droplet), FadeOut(ray_in), FadeOut(refract_in), FadeOut(rainbow_rays))

        # === Animation for Lecture Line 2 ===
        # Prism splitting light
        self.play(self.lecture[1].animate.set_color(WHITE))
        
        # Issue 33: Integrate SVG prism asset
        prism = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/prism.svg").set_color(WHITE)
        # Issue 44: Adjusted placement for prism to avoid overlap
        self.place_in_area(prism, 'B3', 'D5', scale_factor=0.9)
        
        # Incident ray starting from grid B2 to avoid text
        white_in = Line(start=self.grid["B2"] + LEFT * 0.5, end=prism.get_left(), color=WHITE)
        
        dispersion_group = VGroup()
        for i, color in enumerate(RAINBOW_COLORS):
            p_start = prism.get_left()
            # Simple representation of path through and out of prism
            p_end_inner = prism.get_right() + DOWN * 0.2 + UP * (i * 0.05)
            inner = Line(p_start, p_end_inner, color=color, stroke_width=2)
            outer = Line(p_end_inner, self.grid["D6"] + DOWN * (i * 0.2), color=color, stroke_width=4)
            dispersion_group.add(VGroup(inner, outer))
            
        self.play(Create(prism))
        self.play(Create(white_in))
        self.play(Create(dispersion_group))
        self.wait(1)
        
        self.play(FadeOut(prism), FadeOut(white_in), FadeOut(dispersion_group))

        # === Animation for Lecture Line 3 ===
        # Mantis Shrimp Eye
        self.play(self.lecture[2].animate.set_color(EYE_COLOR))
        
        # Compound eye representation
        eye_base = Ellipse(width=3, height=2, color=EYE_COLOR, fill_opacity=0.2)
        self.place_in_area(eye_base, "B2", "E5", scale_factor=1.0)
        
        # Mid-band strips
        mid_band = Rectangle(width=3, height=0.6, color=EYE_COLOR, stroke_width=2).move_to(eye_base.get_center())
        
        # Representations of multi-spectral filters
        filter_box = VGroup(*[
            Square(side_length=0.4, fill_opacity=0.8, stroke_width=1).set_color(c) 
            for c in [PURPLE, BLUE, GREEN, YELLOW, RED]
        ]).arrange(RIGHT, buff=0.1).move_to(mid_band.get_center())
        
        label_pointer = Text("Multispectral Mid-band Filters", font_size=18, color=EYE_COLOR)
        # Issue 45: Position label higher to avoid being cut off
        self.place_at_grid(label_pointer, 'E4', scale_factor=0.8)
        
        self.play(Create(eye_base))
        self.play(Create(mid_band), Create(filter_box))
        self.play(Write(label_pointer))
        self.wait(2)
