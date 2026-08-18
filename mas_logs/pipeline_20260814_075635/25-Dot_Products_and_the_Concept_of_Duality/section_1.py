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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Dot products measure how vectors align.",
            "Formula: magnitude times magnitude times cos(theta).",
            "Think of projection: how much A aligns B.",
            "Visualize hiker walking along a sloped path.",
            "Projection determines the hiker's total forward progress."
        ]
        self.setup_layout("Geometric Intuition of the Dot Product", lecture_lines)
        
        # Define elements
        u = Vector([1.5, 1, 0], color="#FFD700")
        v = Vector([2, -0.5, 0], color="#00CED1")
        label_u = MathTex(r"u", color="#FFD700")
        label_v = MathTex(r"v", color="#00CED1")
        
        # Hiker and Path icons
        hiker = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hiker.svg", color=RED)
        path = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/path.svg", color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        # Position u and v
        self.place_at_grid(u, 'C2')
        self.place_at_grid(v, 'C3')
        self.play(Create(u), Create(v))
        self.lecture[0].set_color("#FFD700")

        # === Animation for Lecture Line 2 ===
        formula = MathTex(r"u \cdot v = |u||v|\cos(\theta)", font_size=32)
        self.place_in_area(formula, 'B4', 'C6', scale_factor=0.9)
        self.play(Write(formula))
        self.lecture[1].set_color("#FFD700")

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(label_u, 'B2', scale_factor=0.7)
        self.place_at_grid(label_v, 'C4', scale_factor=0.7)
        self.play(FadeIn(label_u), FadeIn(label_v))
        self.lecture[2].set_color("#FFD700")

        # === Animation for Lecture Line 4 ===
        self.place_at_grid(hiker, 'D3', scale_factor=0.5)
        self.play(FadeIn(hiker))
        self.lecture[3].set_color("#FFD700")

        # === Animation for Lecture Line 5 ===
        projection_visual = DashedLine(u.get_end(), [u.get_end()[0], v.get_start()[1], 0], color=WHITE)
        self.place_in_area(projection_visual, 'D3', 'E5', scale_factor=0.8)
        self.play(Create(projection_visual), FadeIn(path))
        self.lecture[4].set_color("#FFD700")
        self.wait(2)
