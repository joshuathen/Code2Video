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
        title = "Visual Proof: The Cylinder Projection"
        lines = [
            "Imagine a sphere perfectly encased inside a cylinder.",
            "Project the sphere's surface horizontally onto the cylinder.",
            "The cylinder wall becomes a rectangle of area 4πr².",
            "Both surfaces cover the exact same total area.",
            "This confirms the surface area is four π r squared."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#0000FF") 
        circle = Circle(radius=1.0, color="#FFFFFF", stroke_width=4)
        cylinder_outline = Rectangle(height=2.0, width=2.0, color="#0000FF", stroke_width=4)
        
        initial_vgroup = VGroup(circle, cylinder_outline)
        # Fix Issue 40: Increased scale and adjusted area for visibility
        self.place_in_area(initial_vgroup, "B2", "D5", scale_factor=1.4)
        
        self.play(Create(circle), Create(cylinder_outline), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFA500")
        
        projections = VGroup()
        for i in range(-5, 6):
            y_val = i * 0.18
            if abs(y_val) < 1.0:
                x_circle = np.sqrt(1.0 - y_val**2)
                l1 = Line(start=[-x_circle, y_val, 0], end=[-1.0, y_val, 0], color="#FFA500")
                l2 = Line(start=[x_circle, y_val, 0], end=[1.0, y_val, 0], color="#FFA500")
                projections.add(l1, l2)
        
        # Adjusted scale to match initial_vgroup
        projections.scale(1.4).move_to(initial_vgroup.get_center())
        
        self.play(Create(projections), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        
        yellow_rect = Rectangle(height=2.0, width=4.0, color="#FFFF00", fill_opacity=0.3)
        # Fix Issue 41: Moved starting column to avoid lecture notes
        self.place_in_area(yellow_rect, "B2", "E6", scale_factor=1.0)
        
        self.play(
            ReplacementTransform(cylinder_outline, yellow_rect),
            FadeOut(circle),
            FadeOut(projections),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FFFF")
        
        h_label = Text("2r", color="#00FFFF", font_size=24)
        w_label = Text("2πr", color="#00FFFF", font_size=24)
        
        h_label.next_to(yellow_rect, LEFT, buff=0.1)
        w_label.next_to(yellow_rect, DOWN, buff=0.1)
        
        self.play(Write(h_label), Write(w_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFFFF")
        
        calc = Text("2r × 2πr = 4πr²", color="#FFFFFF", font_size=32)
        # Fix Issue 42: Adjusted positioning and scale for better centering
        self.place_in_area(calc, "F2", "F5", scale_factor=1.1)
        
        self.play(Write(calc))
        self.wait(2)
