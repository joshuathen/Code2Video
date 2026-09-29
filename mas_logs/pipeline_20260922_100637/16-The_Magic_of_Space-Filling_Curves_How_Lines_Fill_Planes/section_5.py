from manim import *

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
        self.setup_layout("Summary and Geometric Intuition", [
            "These curves reconcile different dimensions.",
            "We see folded origami patterns appearing.",
            "Geometry reveals the hidden filling logic."
        ])
        
        # Load assets
        origami = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/origami.svg")
        
        # 1. Reveal unit square filled with curve (#FFFFFF)
        curve = VMobject(color="#FFFFFF", stroke_width=2)
        curve.set_points_smoothly([self.grid['F4'], self.grid['D5'], self.grid['B6'], self.grid['D5'], self.grid['F4']])
        self.place_at_grid(origami.copy(), 'D5', scale_factor=0.5)

        # 2. Highlight density (#FF0000)
        highlight = Rectangle(color="#FF0000", stroke_width=4, height=1.5, width=1.5)
        self.place_at_grid(highlight, 'C5', scale_factor=0.6)
        
        # 3. Final coverage (#00FF00)
        cover = Square(fill_color="#00FF00", fill_opacity=0.3, stroke_color="#00FF00")
        self.place_in_area(cover, 'B4', 'E6', scale_factor=0.5)
        self.place_at_grid(origami, 'D5', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(Create(curve), FadeIn(origami), self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(Create(highlight), self.lecture[1].animate.set_color("#FF0000"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(cover), FadeIn(origami), self.lecture[2].animate.set_color("#00FF00"))
        self.wait(2)
