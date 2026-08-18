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
        self.setup_layout("Generating Newton Fractals", [
            "Color roots based on final convergence.",
            "Boundaries reveal intricate, infinite patterns.",
            "Chaos emerges from simple iteration rules."
        ])
        
        # Create a container for the visual
        visual_area = Rectangle(width=4.0, height=4.0, color=WHITE)
        # Fix 35: visual area layout
        self.place_in_area(visual_area, 'A4', 'E6', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        # Color roots based on final convergence.
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        # Use assets as requested
        grid_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        pixel_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pixel.svg")
        self.place_in_area(grid_img, 'B4', 'D6', scale_factor=0.5)
        self.add(grid_img)
        
        # Visualize basins - Fix 34
        basin1 = Circle(radius=0.4, color="#00FF00", fill_opacity=0.6)
        basin2 = Circle(radius=0.4, color="#0000FF", fill_opacity=0.6)
        basin3 = Circle(radius=0.4, color="#FF0000", fill_opacity=0.6)
        
        self.place_at_grid(basin1, 'B4', scale_factor=0.6)
        self.place_at_grid(basin2, 'D4', scale_factor=0.6)
        self.place_at_grid(basin3, 'C6', scale_factor=0.6)
        
        self.play(FadeIn(basin1), FadeIn(basin2), FadeIn(basin3))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Boundaries reveal intricate, infinite patterns.
        self.play(self.lecture[1].animate.set_color("#0000FF"))
        
        # Create a stylized representation of fractal boundary - Fix 36
        boundary = VGroup(*[Dot(color=WHITE, radius=0.03) for _ in range(10)])
        self.place_in_area(boundary, 'B5', 'D6', scale_factor=0.4)
        
        self.play(Create(boundary))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Chaos emerges from simple iteration rules.
        self.play(self.lecture[2].animate.set_color("#FF0000"))
        
        # Flash the screen/geometry to simulate chaos
        self.play(Flash(visual_area.get_center(), color=YELLOW, line_length=0.2, num_lines=12))
        self.wait(2)
